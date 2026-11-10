"""Public, memory-mapped XYZ data. Reference answers are never loaded here."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def vector(value, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or (positive and np.any(a <= 0)):
        raise ValueError(
            f"{name} must be three finite {'positive ' if positive else ''}numbers"
        )
    return a


class Grid:
    def __init__(self, data, resolution, origin):
        if data.ndim != 3 or not all(data.shape):
            raise ValueError("Expected a nonempty XYZ volume")
        self.data = data
        self.resolution = vector(resolution, "resolution", True)
        self.origin = vector(
            origin, "origin"
        )  # physical lower corner, not first center
        self.upper = self.origin + np.asarray(data.shape) * self.resolution

    def indices(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("Coordinates must be finite XYZ nanometers")
        indices = np.floor((xyz - self.origin) / self.resolution).astype(np.int64)
        valid = ((indices >= 0) & (indices < self.data.shape)).all(axis=-1)
        return indices, valid

    def nearest(self, xyz):
        indices, valid = self.indices(xyz)
        safe = np.clip(indices, 0, np.asarray(self.data.shape) - 1)
        values = self.data[tuple(np.moveaxis(safe, -1, 0))]
        return np.where(valid, values, np.zeros_like(values)), valid

    def linear(self, xyz):
        from scipy.ndimage import map_coordinates

        _, valid = self.indices(xyz)
        # Voxel centers are origin + (index + .5)*resolution.
        points = (np.asarray(xyz) - self.origin) / self.resolution - 0.5
        values = map_coordinates(
            self.data,
            np.moveaxis(points, -1, 0),
            order=1,
            mode="nearest",
            output=np.float64,
            prefilter=False,
        )
        return np.where(valid, values, 0), valid


class Dataset:
    def __init__(
        self, atoms: Grid, em: Grid, partition=None, edges=None, name="volume"
    ):
        if atoms.data.dtype.kind not in "iu" or np.any(atoms.data < 0):
            raise ValueError("Atom labels must be nonnegative integers")
        if em.data.dtype.kind not in "uif" or not np.isfinite(em.data).all():
            raise ValueError("EM must contain finite numeric intensities")
        self.atoms, self.em, self.name = atoms, em, str(name)
        self.ids, self.counts = np.unique(atoms.data, return_counts=True)
        keep = self.ids != 0
        self.ids, self.counts = self.ids[keep], self.counts[keep]
        if not len(self.ids):
            raise ValueError("Dataset has no foreground atoms")
        partition = partition or {int(a): int(a) for a in self.ids}
        if set(map(int, self.ids)) != set(partition) or any(
            not isinstance(v, (int, np.integer))
            or not 0 < int(v) <= np.iinfo(np.uint64).max
            for v in partition.values()
        ):
            raise ValueError(
                "Initial partition must assign every foreground atom exactly once"
            )
        self.initial = {int(k): int(v) for k, v in partition.items()}
        self.edges = edges
        # Content fingerprint is independent of pathname and excludes any reference.
        h = hashlib.sha256()
        for grid in [atoms, em]:
            h.update(str(grid.data.dtype).encode())
            h.update(
                json.dumps(
                    [
                        list(grid.data.shape),
                        grid.resolution.tolist(),
                        grid.origin.tolist(),
                    ]
                ).encode()
            )
            for slab in grid.data:
                h.update(np.ascontiguousarray(slab).tobytes())
        h.update(json.dumps(sorted(self.initial.items())).encode())
        if edges is not None:
            h.update(json.dumps(edges).encode())
        self.fingerprint = h.hexdigest()

    @classmethod
    def load(cls, manifest):
        path = Path(manifest).resolve()
        config = json.loads(path.read_text())
        if config.get("backend") == "streamed-world-v1":
            from .streamed_dataset import StreamedDataset
            return StreamedDataset(path)
        if config.get("backend") == "indexed-world-v1":
            from .indexed import IndexedDataset
            return IndexedDataset(path)

        from .storage import data_path

        def grid(name):
            c = config[name]
            data = np.load(data_path(c["path"], path.parent), mmap_mode="r", allow_pickle=False)
            axes = c.get("axes", "XYZ").upper()
            if sorted(axes) != ["X", "Y", "Z"]:
                raise ValueError("axes must be a permutation of XYZ")
            return Grid(
                data.transpose([axes.index(a) for a in "XYZ"]),
                c["resolution_nm"],
                c.get("origin_nm", [0, 0, 0]),
            )

        initial = None
        if config.get("partition"):
            with np.load(data_path(config["partition"], path.parent), allow_pickle=False) as z:
                aa, ss = z["atoms"], z["segments"]
                if (
                    aa.dtype.kind not in "iu"
                    or ss.dtype.kind not in "iu"
                    or aa.shape != ss.shape
                    or aa.ndim != 1
                    or len(np.unique(aa)) != len(aa)
                ):
                    raise ValueError("Invalid atom partition")
                initial = dict(zip(map(int, aa), map(int, ss)))
        edges = None
        if config.get("edges"):
            with np.load(data_path(config["edges"], path.parent), allow_pickle=False) as z:
                edges = [
                    (int(a), int(b), float(c))
                    for a, b, c in zip(z["u"], z["v"], z["capacity"], strict=True)
                ]
        dataset = cls(
            grid("atoms"), grid("em"), initial, edges, config.get("name", path.stem)
        )
        if "em_pyramid" in config:
            from .multiscale_em import MultiscaleEM
            dataset.geometry_fingerprint = dataset.fingerprint
            dataset.fingerprint = hashlib.sha256(
                (dataset.fingerprint + json.dumps(config["em_pyramid"], sort_keys=True)).encode()
            ).hexdigest()
            dataset.em = MultiscaleEM(dataset.em, config["em_pyramid"])
        from .sampling import load_levels
        load_levels(dataset, config, path.parent)
        if config.get('segmentation_pyramid'):
            h = hashlib.sha256(dataset.fingerprint.encode())
            for g in dataset.segmentation_levels[1:]:
                h.update(json.dumps([g.resolution.tolist(), g.origin.tolist()]).encode())
                h.update(str(g.data.dtype).encode())
                for slab in g.data:
                    h.update(np.ascontiguousarray(slab).tobytes())
            dataset.fingerprint = h.hexdigest()
        dataset.manifest = path
        return dataset

    def graph(self):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(map(int, self.ids))
        if self.edges is not None:
            for a, b, c in self.edges:
                if (
                    a not in self.initial
                    or b not in self.initial
                    or a == b
                    or not np.isfinite(c)
                    or c <= 0
                ):
                    raise ValueError(
                        "Edges require known distinct atoms and positive finite capacities"
                    )
                g.add_edge(a, b, capacity=c)
        else:
            # Geometric mincut only; this is not a membrane-confidence estimate.
            v = self.atoms.data
            for axis in range(3):
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                a, b = v[tuple(lo)], v[tuple(hi)]
                valid = (a != b) & (a != 0) & (b != 0)
                pairs = np.sort(np.stack([a[valid], b[valid]], axis=1), axis=1)
                pairs, counts = np.unique(pairs, axis=0, return_counts=True)
                area = np.prod(np.delete(self.atoms.resolution, axis))
                for (a, b), count in zip(pairs, counts):
                    a, b = int(a), int(b)
                    old = g.get_edge_data(a, b, {}).get("capacity", 0.0)
                    g.add_edge(a, b, capacity=old + float(count * area))
        return g
