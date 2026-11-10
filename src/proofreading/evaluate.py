"""Offline evaluator. Never registered as an agent tool or HTTP route."""

import numpy as np


def metrics(atoms, partition, reference, scope):
    if (
        reference.shape != atoms.shape
        or scope.shape != atoms.shape
        or scope.dtype != bool
    ):
        raise ValueError(
            "Reference and boolean frozen scope must match the atom XYZ grid"
        )
    if reference.dtype.kind not in "iu":
        raise ValueError(
            "Reference must contain nonnegative integer labels; 0 means unknown"
        )
    # Count a compact contingency table in bounded chunks. Sorting integer
    # pair codes avoids a giant structured (prediction, reference) voxel array.
    from collections import Counter
    contingency = Counter()
    scored_voxels = scope_voxels = unknown_voxels = 0
    iterator = np.nditer([atoms, reference, scope],
        flags=['external_loop','buffered','zerosize_ok'],
        op_flags=[['readonly']]*3, order='C', buffersize=1<<20)
    for aa, rr, ss in iterator:
        if rr.dtype.kind=="i" and np.any(rr<0):
            raise ValueError("Reference must contain nonnegative integer labels")
        valid = ss & (rr != 0) & (aa != 0)
        scope_voxels += int(np.count_nonzero(ss))
        unknown_voxels += int(np.count_nonzero(ss & (rr == 0)))
        scored_voxels += int(np.count_nonzero(valid))
        if not valid.any():continue
        unique_atoms, atom_inverse = np.unique(aa[valid],return_inverse=True)
        owners = np.asarray([partition[int(a)] for a in unique_atoms],dtype=np.uint64)
        predicted_labels, atom_prediction = np.unique(owners,return_inverse=True)
        reference_labels, reference_inverse = np.unique(rr[valid],return_inverse=True)
        codes = atom_prediction[atom_inverse]*len(reference_labels)+reference_inverse
        keys, counts = np.unique(codes,return_counts=True)
        for key,count in zip(keys,counts):
            prediction = int(predicted_labels[key//len(reference_labels)])
            truth = int(reference_labels[key%len(reference_labels)])
            contingency[prediction,truth] += int(count)
    if not scored_voxels:
        raise ValueError("No foreground voxels with known reference labels inside scoring scope")
    return contingency_metrics(contingency, scope_voxels, unknown_voxels)


def contingency_metrics(contingency, scope_voxels, unknown_voxels):
    """Shared dense/compact arithmetic so packaging cannot change scores."""
    scored_voxels=sum(map(int,contingency.values()))
    if not scored_voxels:raise ValueError('No foreground voxels with known reference labels inside scoring scope')
    ordered = sorted(contingency)
    pairs = np.asarray(ordered,dtype=np.uint64)
    n = np.asarray([contingency[key] for key in ordered],dtype=np.uint64)
    _, pi = np.unique(pairs[:, 0], return_inverse=True)
    _, ri = np.unique(pairs[:, 1], return_inverse=True)
    pcount = np.bincount(pi, weights=n)
    rcount = np.bincount(ri, weights=n)
    probabilities = n / n.sum()
    merge_vi = float(np.sum(probabilities * np.log2(pcount[pi] / n)))
    split_vi = float(np.sum(probabilities * np.log2(rcount[ri] / n)))
    choose2 = lambda x: sum(int(v) * (int(v) - 1) // 2 for v in x)
    true_positive = choose2(n)
    false_merge_pairs = choose2(pcount) - true_positive
    false_split_pairs = choose2(rcount) - true_positive
    return {
        "vi_merge_bits": merge_vi,
        "vi_split_bits": split_vi,
        "vi_total_bits": merge_vi + split_vi,
        "false_merge_voxel_pairs": false_merge_pairs,
        "false_split_voxel_pairs": false_split_pairs,
        "scored_voxels": scored_voxels,
        "scope_voxels": scope_voxels,
        "unknown_reference_voxels_in_scope": unknown_voxels,
        "scope": "Only non-background voxels with known reference labels in the supplied frozen mask; reference provenance determines what those labels establish. No outside-crop denominator.",
    }


def evaluate(dataset, partition_file, reference_file, scope_file):
    if hasattr(dataset, "save_partition"):
        raise ValueError("Indexed sparse sessions require an indexed evaluator; dense evaluation is not supported")
    reference = np.load(reference_file, mmap_mode="r", allow_pickle=False)
    scope = np.load(scope_file, mmap_mode="r", allow_pickle=False)
    with np.load(partition_file, allow_pickle=False) as z:
        a, s = z["atoms"], z["segments"]
        if (
            a.ndim != 1
            or a.shape != s.shape
            or len(np.unique(a)) != len(a)
            or np.any(s <= 0)
        ):
            raise ValueError("Invalid output partition")
        partition = dict(zip(map(int, a), map(int, s)))
    if set(partition) != set(dataset.initial):
        raise ValueError("Final partition lost or added atoms")
    initial = metrics(dataset.atoms.data, dataset.initial, reference, scope)
    final = metrics(dataset.atoms.data, partition, reference, scope)
    return {
        "initial": initial,
        "final": final,
        "vi_improvement_bits": initial["vi_total_bits"] - final["vi_total_bits"],
        "reference_visible_to_agent": False,
        "scope_attachment": scope_attachment(dataset.atoms.data,dataset.initial,partition,scope,atom_counts=(dataset.ids,dataset.counts)),
    }


def file_hash(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def freeze(dataset, reference_file, scope_file, output):
    """Evaluator-side contract, created before any model decisions."""
    import json, time
    from pathlib import Path

    if hasattr(dataset, "save_partition"):
        raise ValueError("Indexed worlds require an indexed frozen scoring contract")
    # Validate alignment and a nonempty scope before committing the benchmark.
    metrics(
        dataset.atoms.data,
        dataset.initial,
        np.load(reference_file, mmap_mode="r"),
        np.load(scope_file, mmap_mode="r"),
    )
    spec = {
        "schema": 1,
        "frozen_at_unix": time.time(),
        "dataset_sha256": dataset.fingerprint,
        "reference": str(Path(reference_file).resolve()),
        "reference_sha256": file_hash(reference_file),
        "scope": str(Path(scope_file).resolve()),
        "scope_sha256": file_hash(scope_file),
        "contract": "Caller certifies reference provenance and XYZ alignment. Scope must be fixed before inference.",
    }
    with Path(output).open("x") as f:
        f.write(json.dumps(spec, indent=2) + "\n")
    return spec


def evaluate_frozen(dataset, session_dir, spec_file):
    import json
    from pathlib import Path

    directory = Path(session_dir)
    spec = json.loads(Path(spec_file).read_text())
    meta = json.loads((directory / "session.json").read_text())
    if (
        spec["dataset_sha256"] != dataset.fingerprint
        or meta["dataset_sha256"] != dataset.fingerprint
    ):
        raise ValueError("Evaluation dataset fingerprint mismatch")
    if spec["frozen_at_unix"] > meta["created_at_unix"]:
        raise ValueError("Scoring scope was frozen after the episode started")
    from .storage import data_path
    paths = {name: data_path(spec[name], Path(spec_file).parent) for name in ("reference", "scope")}
    for name in ["reference", "scope"]:
        if file_hash(paths[name]) != spec[name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    result = evaluate(
        dataset, directory / "partition.npz", paths["reference"], paths["scope"]
    )
    result["frozen_spec_sha256"] = file_hash(spec_file)
    result["dataset_sha256"] = dataset.fingerprint
    result["frozen_before_episode"] = True
    return result


def scope_attachment(atoms, initial, final, scope, *, atom_counts=None):
    """Track material outside the frozen target joined into its predicted groups.

    This is a scope diagnostic, not a biological wrong-merge judgment. It catches
    a degenerate 'merge everything' solution that a target-only VI score misses.
    Existing attachments and label-only renaming are not counted as new material.
    """
    if scope.shape!=atoms.shape or scope.dtype!=bool:
        raise ValueError('Frozen boolean scope must align with atoms')
    from collections import Counter
    def chunks():
        return np.nditer([atoms,scope],flags=['external_loop','buffered','zerosize_ok'],
                         op_flags=[['readonly']]*2,order='C',buffersize=1<<20)
    has_target=has_outside=False
    for aa,ss in chunks():
        has_target |= bool(np.any(ss & (aa!=0)))
        has_outside |= bool(np.any((~ss) & (aa!=0)))
    if not has_target:raise ValueError('Empty foreground target scope')
    target_counts=Counter();outside_counts=Counter()
    if has_outside:
        for aa,ss in chunks():
            labels,counts=np.unique(aa[ss & (aa!=0)],return_counts=True)
            for atom,count in zip(labels,counts):target_counts[int(atom)]+=int(count)
            if atom_counts is None:
                labels,counts=np.unique(aa[(~ss)&(aa!=0)],return_counts=True)
                for atom,count in zip(labels,counts):outside_counts[int(atom)]+=int(count)
        if atom_counts is not None:
            labels,counts=atom_counts
            for atom,count in zip(labels,counts):
                outside=int(count)-target_counts[int(atom)]
                if outside<0:raise ValueError('Atom count metadata is inconsistent with scope')
                if outside:outside_counts[int(atom)]=outside
    return attachment_counts(initial, final, target_counts, outside_counts)


def attachment_counts(initial, final, target_counts, outside_counts):
    """Scope attachment using per-atom counts, independent of imaging."""
    target_atoms=set(target_counts)
    before_owners={initial[a] for a in target_atoms}
    after_owners={final[a] for a in target_atoms}
    before=after=added=removed=0
    for atom,count in outside_counts.items():
        was=initial[atom] in before_owners;now=final[atom] in after_owners
        if was:before+=count
        if now:after+=count
        if now and not was:added+=count
        if was and not now:removed+=count
    return {'initial_outside_voxels':before,'final_outside_voxels':after,
            'newly_attached_outside_voxels':added,'detached_outside_voxels':removed,
            'interpretation':'Outside the fixed target scope; requires separate certified reference evidence to call biologically wrong.'}
