"""Transactional, replayable tools. This module has no reference-label access."""

from __future__ import annotations
from dataclasses import asdict
import copy
import json
from pathlib import Path
import time
import threading
import numpy as np
from .view import View, render
from .sampling import metadata as paired_sampling_metadata, grids as sampling_grids
from .volume import vector


class Session:
    def __init__(self, dataset, output, *, seeded_split_policy="preserve_sides", split_algorithm="connected_min_cut"):
        if seeded_split_policy not in {"preserve_sides", "connected_components"}:
            raise ValueError("Unknown seeded split policy")
        self.seeded_split_policy = seeded_split_policy
        if split_algorithm not in {"min_cut", "connected_min_cut"}:
            raise ValueError("Unknown split algorithm")
        self.split_algorithm = split_algorithm
        self.lock = threading.RLock()
        self.dataset = dataset
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "frames").mkdir()
        self.partition = dataset.initial.copy()
        self.selected, self.hidden = set(), set()
        self.annotations = []
        self.view = View(
            ((dataset.atoms.origin + dataset.atoms.upper) / 2).tolist(),
            pixel_size_nm=float(max(dataset.atoms.upper - dataset.atoms.origin) / 512),
        )
        self.scene = {"target_nm": ((dataset.atoms.origin + dataset.atoms.upper)/2).tolist(),
                      "azimuth": 35., "elevation": 25.,
                      "view_height_nm": float(max(dataset.atoms.upper-dataset.atoms.origin)*1.2)}
        if hasattr(dataset,"initial_view"):
            self.view=View(**dataset.initial_view).validate()
            self.scene.update(target_nm=list(self.view.center_nm),
                              view_height_nm=self.view.pixel_size_nm*self.view.height*1.2)
        self.revision, self.serial, self.next_id = (
            0,
            0,
            (dataset.max_segment_id if hasattr(dataset, "max_segment_id") else max(self.partition.values())) + 1,
        )
        self.views, self.previews, self.undo_stack, self.redo_stack = {}, {}, [], []
        self.scene_views = {}
        self.action_history = []
        self.preview_serial = 0
        self.finished = False
        self._graph = None
        self.events = []
        (self.output / "session.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "seeded_split_policy": self.seeded_split_policy,
                    "split_algorithm": self.split_algorithm,
                    "partition_format": getattr(dataset, "partition_format", "sparse-overlay-v1" if hasattr(dataset, "save_partition") else "full-v1"),
                    "identity_kind": getattr(dataset, "identity_kind", "content-sha256"),
                    "dataset_sha256": dataset.fingerprint,
                    "created_at_unix": time.time(),
                    "reference_access": False,
                    "name": dataset.name,
                },
                indent=2,
            )
        )
        self._save_partition()

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.dataset.graph()
        return self._graph

    def _save_partition(self):
        tmp = self.output / "partition.tmp.npz"
        if hasattr(self.dataset, "save_partition"):
            self.dataset.save_partition(self.partition, tmp)
            tmp.replace(self.output / "partition.npz")
            return
        np.savez_compressed(
            tmp,
            atoms=self.dataset.ids,
            segments=np.array(
                [self.partition[int(a)] for a in self.dataset.ids], dtype=np.uint64
            ),
        )
        tmp.replace(self.output / "partition.npz")

    def state(self):
        return {
            "revision": self.revision,
            "selected": sorted(map(str, self.selected)),
            "hidden": sorted(map(str, self.hidden)),
            "view": self.view.metadata(),
            "em_sampling": paired_sampling_metadata(self.dataset, self.view)[0],
            "segmentation_sampling": paired_sampling_metadata(self.dataset, self.view)[1],
            "scene": copy.deepcopy(self.scene),
            "annotations": copy.deepcopy(self.annotations),
            "finished": self.finished,
            **self._history_status(),
            "undo_depth": len(self.undo_stack),
            "redo_depth": len(self.redo_stack),
            "bounds_nm": [
                self.dataset.atoms.origin.tolist(),
                self.dataset.atoms.upper.tolist(),
            ],
            "resolution_nm": self.dataset.atoms.resolution.tolist(),
            "edit_granularity": "immutable atoms; splits cannot divide an atom",
            "split_capacity": getattr(self.dataset, "split_capacity", "supplied graph")
            if self.dataset.edges is not None
            else "physical contact area (not EM confidence)",
        }

    def execute(self, name, arguments=None, *, replay_legacy=False, refresh_selection=True):
        with self.lock:
            return self._execute(name, arguments, replay_legacy=replay_legacy, refresh_selection=refresh_selection)

    def _execute(self, name, arguments=None, *, replay_legacy=False, refresh_selection=True):
        from .tools import SCHEMAS, LEGACY_SCHEMAS
        available = {**SCHEMAS, **LEGACY_SCHEMAS} if replay_legacy else SCHEMAS

        arguments = {} if arguments is None else arguments
        started = time.monotonic()
        self.serial += 1
        try:
            if name not in available:
                raise ValueError(f"Unknown tool {name}")
            import jsonschema

            try:
                jsonschema.validate(arguments, available[name]["parameters"])
            except jsonschema.ValidationError as exc:
                raise ValueError(exc.message) from exc
            if self.finished and name not in [
                "state",
                "look",
                "look_stack",
                "inspect_segment",
                "list_segments",
                "mesh",
                "undo_action",
            ]:
                raise ValueError("Episode is finished")
            editing = name in {"commit", "undo", "redo", "undo_edit", "redo_edit"}
            selecting = name == "select" or (name == "click" and arguments.get("mode", "select") != "pick")
            tracked = editing or selecting or name in {"set_em_resolution", "set_view", "set_scene", "annotate", "preview_merge", "preview_split", "finish"}
            restore_partition = editing or (name == "undo_action" and bool(self.action_history) and self.action_history[-1]["before"]["partition"] is not None)
            checkpoint = self._action_snapshot(restore_partition) if tracked or name == "undo_action" else None
            old_history, old_views = list(self.action_history), dict(self.views)
            old_scene_views = dict(self.scene_views)
            try:
                result = getattr(self, name)(**arguments)
                if selecting and refresh_selection:
                    result["observation"] = self.look()
            except Exception:
                if checkpoint is not None:
                    self._restore_action(checkpoint, rollback=True)
                    self.action_history, self.views = old_history, old_views
                self.scene_views = old_scene_views
                raise
            changed = tracked and (editing or
                checkpoint["selected"] != self.selected or checkpoint["hidden"] != self.hidden or
                checkpoint["view"] != self.view or checkpoint["scene"] != self.scene or
                checkpoint["annotations"] != self.annotations or checkpoint["finished"] != self.finished or
                checkpoint["previews"] != self.previews)
            if changed:
                self.action_history.append({"step": self.serial, "tool": name, "before": checkpoint})
            if "state" in result:
                result["state"].update(self._history_status())
            if "undo_depth" in result:
                result.update(self._history_status())
            response = {"ok": True, "result": result}
        except (ValueError, TypeError, KeyError) as exc:
            response = {"ok": False, "error": str(exc)}
        event = {
            "step": self.serial,
            "tool": name,
            "arguments": arguments,
            "response": response,
            "revision": self.revision,
            "elapsed_seconds": time.monotonic() - started,
        }
        self.events.append(event)
        with (self.output / "events.jsonl").open("a") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")
        return response

    def look(self, **changes):
        if "offsets_nm" in changes:
            return self.look_stack(changes.pop("offsets_nm"), **changes)
        # Transient orthogonal views must not change navigation or undo history.
        if changes:
            original = self.view
            self.view = View(**{**asdict(original), **changes}).validate()
            try:
                return self.look()
            finally:
                self.view = original
        # Repeated clicks/look calls often leave the EM plane unchanged. Keep
        # one rendered plane; camera-only moves need no resampling/recoloring.
        plane_key=(self.revision,json.dumps(asdict(self.view),sort_keys=True),
                   tuple(sorted(self.selected)),tuple(sorted(self.hidden)))
        cached=getattr(self,'_em_plane_cache',None)
        ident = f"view-{len(self.views) + 1:06d}"
        frame = f"frames/{ident}.png"
        scene_path=f"frames/{ident}-scene.png"
        pool=getattr(self.dataset,'observation_pool',None)
        if cached is not None and cached[0]==plane_key:
            _,image,visible,intensity_diagnostics=cached
            intensity_diagnostics=copy.deepcopy(intensity_diagnostics)
            scene=self._paired_scene(scene_path)
        else:
            intensity_diagnostics = {}
            def render_plane():
                return render(self.dataset,self.partition,self.selected,self.hidden,self.view,
                              diagnostics=intensity_diagnostics)
            if pool is None:
                image,visible=render_plane()
                scene=self._paired_scene(scene_path)
            else:
                # Keep EGL on its owning thread; overlap EM/source work with
                # mesh loading and 3D rendering. Actions remain transactional.
                future=pool.submit(render_plane)
                try:
                    scene=self._paired_scene(scene_path)
                    image,visible=future.result()
                finally:
                    from concurrent.futures import wait
                    wait([future])  # No source work may outlive action rollback.
            self._em_plane_cache=(plane_key,image,visible,copy.deepcopy(intensity_diagnostics))
        image.save(self.output / frame, compress_level=1)
        self.views[ident] = (copy.deepcopy(self.view), self.revision)
        return {
            "view_id": ident,
            "revision": self.revision,
            "image": frame,
            "scene_observation": scene,
            "selected": sorted(map(str, self.selected - self.hidden)),
            "hidden": sorted(map(str, self.hidden)),
            "transform": self.view.metadata(),
            "em_sampling": paired_sampling_metadata(self.dataset, self.view)[0],
            "segmentation_sampling": paired_sampling_metadata(self.dataset, self.view)[1],
            "em_intensity": {k:v for k,v in intensity_diagnostics.items() if k!="segmentation_coverage"},
            "segmentation_coverage": intensity_diagnostics.get("segmentation_coverage", {}),
            "visible_segments": visible[:1024],
            "visible_segment_count": len(visible),
            "visible_segments_truncated": len(visible) > 1024,
            "selection_note": "Click any visible pixel to identify its segment, including IDs omitted from this bounded summary.",
        }

    def look_stack(self, offsets_nm, **changes):
        """Read consecutive planes without changing the persistent camera."""
        offsets = np.asarray(offsets_nm, dtype=float)
        if offsets.ndim != 1 or not 1 <= len(offsets) <= 9 or not np.isfinite(offsets).all():
            raise ValueError("Provide one to nine finite plane offsets in nanometers")
        base = View(**{**asdict(self.view), **changes, "projection": "slice", "slab_nm": 0.0}).validate()
        normal = np.cross(base.right, base.down)
        original_views = dict(self.views)
        slices = []
        try:
            for offset in offsets:
                center = (np.asarray(base.center_nm) + offset * normal).tolist()
                observation = self.look(**{**asdict(base), "center_nm": center})
                slices.append({"offset_nm": float(offset), "observation": observation})
        except Exception:
            self.views = original_views
            raise
        return {"slices": slices, "normal": normal.tolist(),
                "base_center_nm": base.center_nm,
                "note": "Separate single-plane observations in requested order; no slab projection. Each slice has its own clickable view ID and paired 3D plane. Persistent camera and selection are unchanged."}

    def _paired_scene(self, path):
        from .scene import render_scene
        rendering=render_scene(self,self.output/path)
        picking={} if Path(path).name.startswith('edit-') else self._capture_scene(path)
        return {**picking,"image":path,"scene":copy.deepcopy(self.scene),
                "em_transform":self.view.metadata(),"revision":self.revision,
                "selected":list(map(str,sorted(self.selected-self.hidden))),
                "rendering":rendering,
                "note":"EM center-plane outline; slab thickness is not depicted."}

    def _capture_scene(self, path):
        from .scene import geometry
        meshes=geometry(self)
        bounds=self.dataset.scene_bounds(self) if hasattr(self.dataset,'scene_bounds') else (self.dataset.atoms.origin,self.dataset.atoms.upper)
        ident='scene:'+path
        self.scene_views[ident]=(copy.deepcopy(self.scene),np.array(bounds),self.revision,meshes)
        def size():
            arrays={id(a):a for _,_,_,parts in self.scene_views.values() for _,v,f in parts for a in (v,f)}
            return sum(a.nbytes for a in arrays.values())
        while len(self.scene_views)>1 and (len(self.scene_views)>16 or size()>256*1024**2):
            del self.scene_views[next(iter(self.scene_views))]
        return {'view_id':ident,'width':900,'height':700,
                'click_semantics':'click(view_id,x,y) picks the frontmost segment surface; returned xyz_nm is a verified interior voxel seed. Pixel centers are zero-based. Preview surfaces are not clickable.'}

    def _click_scene(self, view_id, x, y, mode):
        from .picking import surface, PickMiss
        camera,bounds,revision,meshes=self.scene_views[view_id]
        if revision!=self.revision:raise ValueError('Stale image after edit; request a fresh view')
        try:
            sid,point,direction=surface(meshes,camera,bounds,x,y)
        except PickMiss:
            return {'hit':False,'mode':mode,'state':self.state(),
                    'note':'No segment surface at this pixel. Selection unchanged. Click inside a visible surface or zoom in.'}
        em,grid=sampling_grids(self.dataset,self.view)
        # Mesh simplification can place the surface just outside segmentation.
        # Probe only a small distance into the visible surface, never a hidden
        # object behind it, and verify the seed against current segmentation.
        offsets=np.linspace(0,2*max(grid.resolution),17)
        candidates=point+offsets[:,None]*direction
        labels,inside=grid.nearest(candidates)
        _,em_inside=em.indices(candidates)
        for candidate,label,valid in zip(candidates,labels,inside & em_inside):
            if not valid or not label:continue
            atom=int(label)
            from .ownership_cache import MissingOwnership
            try:owner=self.partition[atom]
            except MissingOwnership:continue
            if owner!=sid:continue
            indices,valid=grid.indices(candidate)
            xyz=grid.origin+(indices+.5)*grid.resolution
            check_atom,check_owner=self._at(xyz)
            if check_atom!=atom or check_owner!=sid:continue
            if mode!='pick':self.select([str(sid)],mode)
            return {'hit':True,'atom':str(atom),'segment':str(sid),'xyz_nm':xyz.tolist(),
                    'surface_xyz_nm':point.tolist(),'mode':mode,'state':self.state()}
        raise ValueError('Surface has no matching interior seed nearby at current resolution; zoom in or use EM')

    def set_em_resolution(self, xy_nm):
        """Choose matching EM and segmentation source detail, independently of zoom."""
        return self.set_view(em_resolution_nm=xy_nm)

    def set_view(self, **changes):
        unknown = set(changes) - set(asdict(self.view))
        if unknown:
            raise ValueError(f"Unknown view parameters: {sorted(unknown)}")
        new = View(**{**asdict(self.view), **changes}).validate()
        old = self.view
        self.view = new
        try:
            return self.look()
        except Exception:
            self.view = old
            raise

    def _segment(self, segment, *, complete=False):
        sid = int(segment)
        if hasattr(self.dataset,"ensure_segment"):
            self.dataset.ensure_segment(self.partition,sid)
        if complete and hasattr(self.partition, "complete_atoms_for"):
            atoms = self.partition.complete_atoms_for(sid)
            if not atoms:
                raise ValueError(f"No live segment {sid}")
            return sid, atoms
        atoms = (self.partition.atoms_for(sid) if hasattr(self.partition, "atoms_for")
                 else {a for a, s in self.partition.items() if s == sid})
        if not atoms:
            raise ValueError(f"No live segment {sid}")
        return sid, atoms

    def _at(self, xyz, view=None):
        xyz = vector(xyz, "xyz_nm")
        em, atoms = sampling_grids(self.dataset, self.view if view is None else view)
        value, valid = atoms.nearest(xyz)
        _, em_valid = em.indices(xyz)
        if not valid or not em_valid or not value:
            from .picking import PickMiss
            raise PickMiss("Point is outside foreground segmentation")
        atom = int(value)
        return atom, self.partition[atom]

    def click(self, view_id, x, y, mode="select"):
        if view_id in self.scene_views:return self._click_scene(view_id,x,y,mode)
        if view_id not in self.views:
            raise ValueError("Unknown view_id; obtain a view first")
        view, revision = self.views[view_id]
        if revision != self.revision:
            raise ValueError("Stale image after edit; request a fresh view")
        if (
            not np.isfinite([x, y]).all()
            or not 0 <= x < view.width
            or not 0 <= y < view.height
        ):
            raise ValueError("Click outside image")
        xyz = view.xyz(x, y)
        from .ownership_cache import MissingOwnership
        from .picking import PickMiss
        try:
            atom, sid = self._at(xyz, view)
        except (MissingOwnership,PickMiss) as exc:
            return {'hit':False,'mode':mode,'state':self.state(),
                    'reason':'unavailable_historical_ownership' if isinstance(exc,MissingOwnership) else 'background',
                    'note':'No selectable segment at this pixel. Selection unchanged; use another pixel.'}
        if mode != "pick":
            self.select([str(sid)], mode)
        return {
            "hit": True,
            "atom": str(atom),
            "segment": str(sid),
            "xyz_nm": xyz.tolist(),
            "mode": mode,
            "state": self.state(),
        }

    def select(self, segments, mode="select"):
        def selected_id(value):
            sid=int(value)
            if hasattr(self.dataset,'can_select_unobserved') and self.dataset.can_select_unobserved(self.partition,sid):
                return sid
            return self._segment(sid)[0]
        ids = {selected_id(s) for s in segments}
        if mode == "select":
            self.selected |= ids
            self.hidden -= ids
        elif mode == "replace":
            self.selected = ids
            self.hidden -= ids
        elif mode == "deselect":
            self.selected -= ids
        elif mode == "hide":
            self.hidden |= ids
            self.selected -= ids
        elif mode == "show":
            self.hidden -= ids
        else:
            raise ValueError("mode must be select, replace, deselect, hide, or show")
        return self.state()

    def annotate(self, xyz_nm, text):
        xyz = vector(xyz_nm, "xyz_nm").tolist()
        if not isinstance(text, str) or len(text) > 10000:
            raise ValueError("Invalid annotation text")
        self.annotations.append({"xyz_nm": xyz, "text": text})
        return self.state()

    def list_segments(self, offset=0, limit=100):
        if (
            type(offset) is not int
            or type(limit) is not int
            or offset < 0
            or not 1 <= limit <= 1000
        ):
            raise ValueError("offset >= 0 and limit in [1,1000] required")
        from .view import visible_segments, color
        _,_,visible,counts=visible_segments(self.dataset,self.partition,self.selected,self.hidden,self.view)
        rows=sorted(visible,key=lambda sid:(-counts[sid],sid))
        return {
            "scope":"current_em_view", "revision":self.revision,
            "total":len(rows),
            "segments":[{"id":str(sid),"pixels":counts[sid],
                         "color_rgb":color(sid).round().astype(int).tolist(),
                         "selected":sid in self.selected} for sid in rows[offset:offset+limit]],
            "next_offset":offset+limit if offset+limit<len(rows) else None,
        }

    def inspect_segment(self, segment, offset=0, limit=1000):
        sid, atoms = self._segment(segment)
        if (
            type(offset) is not int
            or type(limit) is not int
            or offset < 0
            or not 1 <= limit <= 10000
        ):
            raise ValueError("Invalid atom pagination")
        if hasattr(self.dataset, "inspect"):
            return self.dataset.inspect(self.partition, sid, atoms, offset, limit)
        neighbors = {
            self.partition[b] for a in atoms for b in self.graph.neighbors(a)
        } - {sid}
        mask = np.isin(self.dataset.atoms.data, list(atoms))
        positions = [
            np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i)))
            for i in range(3)
        ]
        lo = (
            self.dataset.atoms.origin
            + np.array([p[0] for p in positions]) * self.dataset.atoms.resolution
        )
        hi = (
            self.dataset.atoms.origin
            + (np.array([p[-1] for p in positions]) + 1) * self.dataset.atoms.resolution
        )
        return {
            "segment": str(sid),
            "n_atoms": len(atoms),
            "bounds_nm": [lo.tolist(), hi.tolist()],
            "center_nm": ((lo + hi) / 2).tolist(),
            "voxels": int(mask.sum()),
            "atoms": list(map(str, sorted(atoms)[offset : offset + limit])),
            "next_offset": offset + limit if offset + limit < len(atoms) else None,
            "neighbors": sorted(map(str, neighbors)),
        }

    def preview_merge(self, segments):
        parts = [self._segment(s) for s in segments]
        if len(parts) < 2 or len({s for s, _ in parts}) != len(parts):
            raise ValueError("Merge requires at least two distinct live segments")
        return self._preview(
            "merge", [s for s, _ in parts], [set.union(*(a for _, a in parts))]
        )

    def preview_split(self, segment, source_nm=None, sink_nm=None, groups=None, algorithm=None):
        import networkx as nx

        sid, atoms = self._segment(segment, complete=True)
        diagnostics = {}
        if groups is not None:
            if algorithm is not None:
                raise ValueError("Algorithm applies to seeded splits, not explicit groups")
            if source_nm is not None or sink_nm is not None:
                raise ValueError("Use groups OR seeds")
            children = [set(map(int, g)) for g in groups]
            if (
                len(children) < 2
                or any(not g for g in children)
                or sum(map(len, children)) != len(atoms)
                or set.union(*children) != atoms
            ):
                raise ValueError("Groups must partition every parent atom exactly once")
        else:
            if not source_nm or not sink_nm:
                raise ValueError("Provide nonempty source_nm and sink_nm seed lists")
            source = {self._at(p)[0] for p in source_nm}
            sink = {self._at(p)[0] for p in sink_nm}
            if not source <= atoms or not sink <= atoms or source & sink:
                raise ValueError("Seeds must lie in distinct atoms of this segment")
            from .seeded_split import split
            graph = self.graph.subgraph(atoms)
            sides, diagnostics = split(graph, source, sink, algorithm or self.split_algorithm)
            components = [list(nx.connected_components(self.graph.subgraph(side))) for side in sides]
            unsupported = [c for c in nx.connected_components(self.graph.subgraph(atoms))
                           if not c & (source | sink)]
            diagnostics.update({
                "seeded_split_policy": self.seeded_split_policy,
                "source_sink_component_counts": list(map(len, components)),
                "unseeded_component_atom_counts": sorted(map(len, unsupported), reverse=True),
                "assignment_note": "Disconnected components without seeds have no source/sink identity evidence. Their side is determined by the graph cut, not biological continuity; inspect and seed them explicitly if needed.",
            })
            children = (sides if self.seeded_split_policy == "preserve_sides" else
                        [set(c) for side_components in components for c in side_components])
            if len(children) < 2:
                raise ValueError("Split did not produce two nonempty pieces")
        result = self._preview("split", [sid], children)
        result.update(diagnostics)
        return result

    def _preview(self, kind, parents, groups):
        ident = f"edit-{self.preview_serial + 1:06d}"
        groups = sorted(groups, key=lambda a: (-len(a), min(a)))
        pending_preview = {
            "kind": kind,
            "parents": parents,
            "groups": groups,
            "revision": self.revision,
        }
        # Preview labels are ephemeral; committing creates fresh IDs.
        trial = self.partition.copy()
        if hasattr(trial, "apply_groups"):
            trial.apply_groups(kind,parents,groups,[self.next_id+i for i in range(len(groups))])
        else:
            for i, group in enumerate(groups):
                for a in group:
                    trial[a] = self.next_id + i
        preview_selected = (self.selected - set(parents)) | {self.next_id+i for i in range(len(groups))}
        image, _ = render(self.dataset, trial, preview_selected, self.hidden, self.view)
        path = f"frames/{ident}.png"
        image.save(self.output / path)
        preview_session=copy.copy(self)
        preview_session.partition=trial
        preview_session.selected=preview_selected
        preview_session.__dict__.pop('_scene_mesh_cache',None)
        scene=preview_session._paired_scene(f"frames/{ident}-scene.png")
        self.previews[ident] = pending_preview
        self.preview_serial += 1
        return {
            "scene_observation":scene,
            "preview_id": ident,
            "em_sampling": paired_sampling_metadata(self.dataset, self.view)[0],
            "segmentation_sampling": paired_sampling_metadata(self.dataset, self.view)[1],
            "kind": kind,
            "parents": list(map(str, parents)),
            "child_atom_counts": list(map(len, groups)),
            "revision": self.revision,
            "image": path,
            "note": "Preview only. No reference labels or correctness feedback.",
        }

    def _history_status(self):
        last = self.action_history[-1] if self.action_history else None
        return {"undo_edit_depth": len(self.undo_stack),
                "redo_edit_depth": len(self.redo_stack),
                "undo_action_depth": len(self.action_history),
                "undo_action_target": {"step": last["step"], "tool": last["tool"]} if last else None}

    def _action_snapshot(self, include_partition):
        return {"partition": self.partition.copy() if include_partition else None,
                "selected": set(self.selected), "hidden": set(self.hidden),
                "view": copy.deepcopy(self.view), "scene": copy.deepcopy(self.scene),
                "annotations": copy.deepcopy(self.annotations), "finished": self.finished,
                "undo": list(self.undo_stack), "redo": list(self.redo_stack),
                "previews": dict(self.previews), "revision": self.revision, "next_id": self.next_id}

    def _restore_action(self, saved, rollback=False):
        self.__dict__.pop("_em_plane_cache",None)
        self.selected, self.hidden = set(saved["selected"]), set(saved["hidden"])
        self.view, self.scene = copy.deepcopy(saved["view"]), copy.deepcopy(saved["scene"])
        self.annotations, self.finished = copy.deepcopy(saved["annotations"]), saved["finished"]
        self.undo_stack, self.redo_stack = list(saved["undo"]), list(saved["redo"])
        self.previews = dict(saved["previews"])
        if rollback:
            self.revision, self.next_id = saved["revision"], saved["next_id"]
        elif saved["partition"] is not None:
            self.revision += 1  # Old clickable images and previews remain stale.
        if saved["partition"] is not None:
            self.partition = saved["partition"].copy()
            self.__dict__.pop("_scene_mesh_cache", None)
            self._save_partition()

    def undo_action(self):
        if not self.action_history:
            raise ValueError("No state-changing action to undo")
        last = self.action_history.pop()
        self._restore_action(last["before"])
        return {"undone_action": {"step": last["step"], "tool": last["tool"]},
                "observation": self.look(), "state": self.state()}

    def undo_edit(self):
        return self.undo()

    def redo_edit(self):
        return self.redo()

    def _snapshot(self):
        return (self.partition.copy(), set(self.selected), set(self.hidden))

    def commit(self, preview_id):
        edit = self.previews.get(preview_id)
        if edit is None or edit["revision"] != self.revision:
            raise ValueError("Unknown or stale preview")
        if self.next_id + len(edit["groups"]) > np.iinfo(np.uint64).max:
            raise ValueError("Segment ID space exhausted")
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()
        children = []
        for group in edit["groups"]:
            sid = self.next_id
            self.next_id += 1
            children.append(str(sid))
            if not hasattr(self.partition, "apply_groups"):
                for a in group:
                    self.partition[a] = sid
        if hasattr(self.partition, "apply_groups"):
            self.partition.apply_groups(edit["kind"],edit["parents"],edit["groups"],list(map(int,children)))
        self.selected -= set(edit["parents"])
        self.selected |= set(map(int, children))
        self.hidden -= set(edit["parents"])
        self.revision += 1
        self._save_partition()
        return {"children": children, "observation": self.look(), "state": self.state()}

    def undo(self):
        if not self.undo_stack:
            raise ValueError("Nothing to undo")
        self.redo_stack.append(self._snapshot())
        partition, selected, hidden = self.undo_stack.pop()
        self.partition, self.selected, self.hidden = partition.copy(), set(selected), set(hidden)
        self.revision += 1
        self._save_partition()
        return {"observation": self.look(), "state": self.state()}

    def redo(self):
        if not self.redo_stack:
            raise ValueError("Nothing to redo")
        self.undo_stack.append(self._snapshot())
        partition, selected, hidden = self.redo_stack.pop()
        self.partition, self.selected, self.hidden = partition.copy(), set(selected), set(hidden)
        self.revision += 1
        self._save_partition()
        return {"observation": self.look(), "state": self.state()}

    def finish(self, summary):
        if not isinstance(summary, str):
            raise ValueError("summary must be text")
        self.finished = True
        return {"finished": True, "summary": summary, "state": self.state()}

    def mesh(self, segments=None, elevation=25.0, azimuth=35.0, stride=1):
        from .mesh import render_mesh

        ids = [
            self._segment(s)[0]
            for s in (segments if segments is not None else sorted(self.selected))
        ]
        if not ids:
            raise ValueError("Select or specify segments to mesh")
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        if not np.isfinite([elevation, azimuth]).all():
            raise ValueError("Camera angles must be finite")
        path = f"frames/mesh-{self.serial:06d}-{time.time_ns()}.png"
        rendering = render_mesh(self, ids, elevation, azimuth, stride, self.output / path)
        return {
            "image": path,
            "segments": list(map(str, ids)),
            "rendering": rendering,
            "elevation": elevation,
            "azimuth": azimuth,
            "stride": stride,
            "note": "Surface extraction; stride > 1 may omit thin processes. Use stride 1 for full resolution.",
        }

    def set_scene(self, target_nm=None, azimuth=None, elevation=None, view_height_nm=None):
        from .scene import validate, render_scene
        changes = {k:v for k,v in locals().copy().items() if k in ("target_nm","azimuth","elevation","view_height_nm") and v is not None}
        new = validate({**self.scene, **changes})
        old = self.scene
        self.scene = new
        path = f"frames/scene-{self.serial:06d}.png"
        try:
            rendering = render_scene(self, self.output / path)
        except Exception:
            self.scene = old
            raise
        return {**self._capture_scene(path), "image": path, "scene": copy.deepcopy(self.scene), "state": self.state(), "rendering": rendering}
