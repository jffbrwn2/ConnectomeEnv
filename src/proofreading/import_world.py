"""Import an explicit bounded crop of an existing materialized world, without GT."""

import json
from pathlib import Path
import numpy as np


def _exact_ids(values):
    """Keep NumPy set operations in uint64, never a mixed int64/uint64 float cast."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("World IDs must be nonnegative integers")
    return values.astype(np.uint64, copy=False)


def import_world(world, output, bounds):
    world, output = Path(world), Path(output)
    m = json.loads((world / "manifest.json").read_text())
    seg = np.load(
        world / m.get("seg_file", "seg.npy"), mmap_mode="r", allow_pickle=False
    )
    low, high = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    if np.any(low < 0) or np.any(high > seg.shape) or np.any(low >= high):
        raise ValueError("Crop outside local XYZ voxel bounds")
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    crop = np.asarray(seg[slices])
    ids = np.unique(_exact_ids(crop))
    ids = ids[ids != 0]
    if not len(ids):
        raise ValueError("Empty foreground crop")
    rootdir = world / "root_atoms_csr"
    partition = {}
    if (rootdir / "initial_roots.npy").exists():
        roots = _exact_ids(np.load(rootdir / "root_ids.npy", mmap_mode="r"))
        ptr = np.load(rootdir / "indptr.npy", mmap_mode="r")
        atoms = np.load(rootdir / "atoms.npy", mmap_mode="r")
        initial = _exact_ids(np.load(rootdir / "initial_roots.npy"))
        allowed = np.isin(roots, initial)
        for start in range(0, len(atoms), 1_000_000):
            values = _exact_ids(atoms[start : start + 1_000_000])
            hits = np.flatnonzero(np.isin(values, ids))
            positions = np.searchsorted(ptr, start + hits, side="right") - 1
            for index, pos in zip(hits, positions):
                if allowed[pos]:
                    atom, root = int(values[index]), int(roots[pos])
                    if atom in partition and partition[atom] != root:
                        raise ValueError("Overlapping initial roots")
                    partition[atom] = root
    else:
        # This may be expensive; prefer the CSR for large worlds.
        roots = json.loads((world / "roots.json").read_text())
        wanted = set(map(int, ids))
        for root in roots["initial_roots"]:
            for atom in roots["root_to_atoms"][str(root)]:
                atom = int(atom)
                if atom in wanted:
                    if atom in partition and partition[atom] != int(root):
                        raise ValueError("Overlapping initial roots")
                    partition[atom] = int(root)
    wanted = set(map(int, ids))
    if set(partition) != wanted:
        raise ValueError(
            "Historical partition does not match crop: "
            f"{len(wanted - set(partition))} missing atoms, "
            f"{len(set(partition) - wanted)} unexpected atoms"
        )
    resolution = np.asarray(m["resolution_xyz_nm"], float)
    origin = (np.asarray(m["bounds_min_xyz"]) + low) * resolution
    upper = (np.asarray(m["bounds_min_xyz"]) + high) * resolution
    em = np.load(world / m.get("em_file", "em.npy"), mmap_mode="r", allow_pickle=False)
    grid = m.get("extra", {}).get("em_grid")
    if grid:
        er = np.asarray(grid["resolution_xyz_nm"], float)
        eo = np.asarray(grid["bounds_min_xyz"]) * er
    elif em.shape == seg.shape:
        er = resolution
        eo = np.asarray(m["bounds_min_xyz"]) * resolution
    else:
        raise ValueError("Different EM shape requires explicit extra.em_grid metadata")
    el = np.floor((origin - eo) / er).astype(int)
    eh = np.ceil((upper - eo) / er).astype(int)
    if np.any(el < 0) or np.any(eh > em.shape):
        raise ValueError("EM does not cover requested crop")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "atoms.npy", crop)
    np.save(output / "em.npy", em[tuple(slice(int(a), int(b)) for a, b in zip(el, eh))])
    np.savez(
        output / "initial.npz",
        atoms=ids,
        segments=np.array([partition[int(a)] for a in ids], dtype=np.uint64),
    )
    config = {
        "name": m.get("name", "materialized world") + " / explicit crop",
        "atoms": {
            "path": "atoms.npy",
            "resolution_nm": resolution.tolist(),
            "origin_nm": origin.tolist(),
        },
        "em": {
            "path": "em.npy",
            "resolution_nm": er.tolist(),
            "origin_nm": (eo + el * er).tolist(),
        },
        "partition": "initial.npz",
        "provenance": {
            "world": str(world.resolve()),
            "local_bounds_xyz": list(map(int, bounds)),
            "reference_imported": False,
            "split_graph": "physical contact area; upstream affinities are not silently substituted",
        },
    }
    # Real chunkedgraph lazy-affinity fetching is intentionally not implicit.
    # Users can provide an audited explicit capacity graph through the public format.
    (output / "dataset.json").write_text(json.dumps(config, indent=2) + "\n")
    return output / "dataset.json"
