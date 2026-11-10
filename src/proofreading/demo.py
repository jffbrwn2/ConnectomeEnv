"""Deterministic synthetic fixture; never presented as biological/model evidence."""

import json
from pathlib import Path
import numpy as np
from .volume import Dataset
from .session import Session


def create(directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    x, y, z = np.indices((96, 96, 64))
    first = (
        ((y - (42 + 7 * np.sin(x / 18))) ** 2 + (z - 30) ** 2 < 7**2)
        & (x > 8)
        & (x < 88)
    )
    second = ((x - 62) ** 2 + (z - 32) ** 2 < 6**2) & (y > 9) & (y < 87)
    atom = np.zeros(x.shape, np.uint64)
    atom[first & (x < 47)] = 101
    atom[first & (x >= 47)] = 102
    atom[second & ~first] = 103
    # Small off-branch fragment with an erroneous initial union to 103.
    third = (x - 26) ** 2 + (y - 72) ** 2 + (z - 39) ** 2 < 8**2
    atom[third] = 104
    from scipy.ndimage import binary_erosion, gaussian_filter

    boundary = (atom > 0) & ~binary_erosion(atom > 0)
    rng = np.random.default_rng(17)
    em = np.clip(
        170
        - 95 * gaussian_filter(boundary.astype(float), 0.6)
        - 35 * (atom > 0)
        + rng.normal(0, 9, x.shape),
        0,
        255,
    ).astype("uint8")
    truth = np.zeros(atom.shape, np.uint64)
    truth[np.isin(atom, [101, 102])] = 1
    truth[atom == 103] = 2
    truth[atom == 104] = 3
    np.save(out / "atoms.npy", atom)
    np.save(out / "em.npy", em)
    # These files are evaluator-only. The public manifest never points at them.
    private = out / "evaluation"
    private.mkdir()
    np.save(private / "reference.npy", truth)
    np.save(private / "scope.npy", atom > 0)
    np.savez(
        out / "initial.npz",
        atoms=np.array([101, 102, 103, 104], np.uint64),
        segments=np.array([11, 12, 13, 13], np.uint64),
    )
    config = {
        "name": "Synthetic navigation and editing fixture — not biological evidence",
        "atoms": {
            "path": "atoms.npy",
            "resolution_nm": [40, 40, 80],
            "origin_nm": [1000, 2000, 3040],
        },
        "em": {
            "path": "em.npy",
            "resolution_nm": [40, 40, 80],
            "origin_nm": [1000, 2000, 3040],
        },
        "partition": "initial.npz",
    }
    (out / "dataset.json").write_text(json.dumps(config, indent=2) + "\n")
    return out / "dataset.json"


def scripted(dataset, output):
    session = Session(Dataset.load(dataset), output)
    session.execute(
        "set_view",
        {
            "center_nm": [2920, 3920, 5480],
            "pixel_size_nm": 9.0,
            "width": 512,
            "height": 512,
        },
    )
    session.execute("select", {"segments": ["11", "12"]})
    session.execute("look")
    session.execute("mesh")
    p = session.execute("preview_merge", {"segments": ["11", "12"]})["result"][
        "preview_id"
    ]
    session.execute("commit", {"preview_id": p})
    session.execute("mesh")
    session.execute("undo")
    session.execute("redo")
    session.execute("select", {"segments": ["13"], "mode": "replace"})
    session.execute("mesh", {"azimuth": 105})
    p = session.execute(
        "preview_split", {"segment": "13", "groups": [["103"], ["104"]]}
    )["result"]["preview_id"]
    session.execute("commit", {"preview_id": p})
    session.execute("mesh", {"azimuth": 105})
    session.execute(
        "finish",
        {
            "summary": "Scripted API demonstration: merged two fragments, split a false union, and exercised undo/redo. Not an autonomous model result."
        },
    )
    return session
