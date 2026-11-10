"""Optional native Neuroglancer mirror, with shift-click selecting in the session.

Uses only public EM and the current editable partition. Native camera navigation
is unconstrained. `adopt_view` captures its camera back into the shared tool state.
"""

import threading
import numpy as np
from scipy.spatial.transform import Rotation
from .view import color


class NeuroglancerBridge:
    def __init__(self, session, lock=None):
        import neuroglancer as ng

        self.ng, self.session = ng, session
        self.lock = lock or session.lock
        self.viewer = ng.Viewer()
        self.revision = -1
        self.last_state = None
        self.stopped = threading.Event()
        self.error = None
        em = session.dataset.em
        self.em = em.source if hasattr(em, "source") else self._volume(em, em.data, "image")
        self.viewer.actions.add("workbench-select", self._select)
        self.viewer.actions.add("workbench-adopt-view", self._adopt)
        with self.viewer.config_state.txn() as config:
            config.input_event_bindings.data_view["shift+mousedown0"] = (
                "workbench-select"
            )
            config.input_event_bindings.data_view["keyv"] = "workbench-adopt-view"
        self.sync()
        self.worker = threading.Thread(target=self._poll, daemon=True)
        self.worker.start()

    def _volume(self, grid, data, kind):
        offset = grid.origin / grid.resolution
        if not np.allclose(offset, np.round(offset)):
            raise ValueError(
                "Native Neuroglancer bridge requires grid-aligned physical origins"
            )
        dims = self.ng.CoordinateSpace(
            names=["x", "y", "z"], units="nm", scales=grid.resolution
        )
        return self.ng.LocalVolume(
            data,
            dimensions=dims,
            voxel_offset=np.round(offset).astype(int),
            volume_type=kind,
        )

    def _select(self, action):
        with self.lock:
            point = action.mouse_voxel_coordinates
            if point is None:
                return
            _, sid = self.session._at(point)
            self.session.execute("select", {"segments": [str(sid)]})
            self.sync()

    def _adopt(self, action):
        with self.lock:
            state = self.viewer.state
            matrix = Rotation.from_quat(state.cross_section_orientation).as_matrix()
            result = self.session.execute(
                "set_view",
                {
                    "center_nm": list(state.position),
                    "right": matrix[:, 0].tolist(),
                    "down": matrix[:, 1].tolist(),
                    "pixel_size_nm": float(state.cross_section_scale),
                },
            )
            if not result["ok"]:
                raise ValueError(result["error"])
            self.sync()

    def sync(self):
        import json

        with self.lock:
            public = self.session.state()
            encoded = json.dumps(public, sort_keys=True)
            if encoded == self.last_state:
                return
            changed = self.revision != self.session.revision
            if changed:
                raw = self.session.dataset.atoms.data
                keys = np.array([0, *sorted(self.session.partition)], dtype=np.uint64)
                values = np.array(
                    [0, *[self.session.partition[int(a)] for a in keys[1:]]],
                    dtype=np.uint64,
                )
                labels = values[np.searchsorted(keys, raw)]
                self.seg = self._volume(
                    self.session.dataset.atoms, labels, "segmentation"
                )
            with self.viewer.txn() as state:
                state.dimensions = self.ng.CoordinateSpace(
                    names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
                )
                state.layers["em"] = self.ng.ImageLayer(
                    source=self.em,
                    shader="#uicontrol invlerp normalized\nvoid main(){emitGrayscale(normalized());}",
                )
                state.layers["segments"] = self.ng.SegmentationLayer(
                    source=self.seg, segments=list(self.session.selected-self.session.hidden),
                    segment_colors={str(sid): '#%02x%02x%02x' % tuple(color(sid).astype(int))
                                    for sid in set(self.session.partition.values())},
                    selected_alpha=self.session.view.overlay,
                    not_selected_alpha=0 if self.session.selected else self.session.view.overlay
                )
                state.position = self.session.view.center_nm
                right, down = (
                    np.array(self.session.view.right),
                    np.array(self.session.view.down),
                )
                state.cross_section_orientation = Rotation.from_matrix(
                    np.stack([right, down, np.cross(right, down)], axis=1)
                ).as_quat()
                state.cross_section_scale = self.session.view.pixel_size_nm
                state.layout = "4panel"
            self.revision = self.session.revision
            self.last_state = encoded

    def _poll(self):
        while not self.stopped.wait(0.25):
            try:
                self.sync()
            except Exception as exc:
                self.error = str(exc)
                import sys

                print(
                    "Neuroglancer synchronization failed: " + self.error,
                    file=sys.stderr,
                    flush=True,
                )
                return

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=2)

    @property
    def url(self):
        return self.viewer.get_viewer_url()
