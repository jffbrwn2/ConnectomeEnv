"""Volume error on a common, explicitly expanded supervoxel universe.

Reference zero is unknown, not background. Missing weights are an error: callers
must not drop added material or substitute unit weights for voxel volumes.
"""
from collections import Counter
from numbers import Integral
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment_error(cells):
    """Minimum reassigned volume, invariant to segment names."""
    cells = Counter(cells)
    cells = Counter({k: v for k, v in cells.items() if v})
    if not cells:
        return 0
    # Independent overlap components cannot compete for the same assignment.
    # Decompose first so a full cube does not allocate a dense matrix across
    # thousands of otherwise unrelated, unchanged segments.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (p, r), n in cells.items():
        if p <= 0 or r <= 0 or n < 0:
            raise ValueError('Positive labels and nonnegative measured counts required')
        a, b = find((0, p)), find((1, r))
        parent[a] = b
    components = {}
    for (p, r), n in cells.items():
        components.setdefault(find((0, p)), []).append((p, r, n))
    matched = 0
    for component in components.values():
        predicted = sorted({p for p, _, _ in component})
        reference = sorted({r for _, r, _ in component})
        if len(predicted) == 1 or len(reference) == 1:
            matched += max(n for _, _, n in component)
            continue
        pi = {p: i for i, p in enumerate(predicted)}
        ri = {r: i for i, r in enumerate(reference)}
        overlaps = np.zeros((len(predicted), len(reference)), dtype=np.int64)
        for p, r, n in component:
            overlaps[pi[p], ri[r]] += n
        a, b = linear_sum_assignment(overlaps, maximize=True)
        matched += int(overlaps[a, b].sum())
    return sum(cells.values()) - matched



def score_cells(cells):
    """Cells are (initial segment, final segment, reference segment, voxels).

    Both partitions are evaluated on identical material. Unknown-reference
    volume is disclosed, and prevents a certified exact/worsened classification.
    """
    before = Counter(); after = Counter(); total = unknown = 0
    for i, f, r, n in cells:
        if any(not isinstance(v, Integral) for v in (i, f, r, n)):
            raise ValueError('Integer labels and measured voxel counts required')
        if i <= 0 or f <= 0 or r < 0 or n < 0:
            raise ValueError('Invalid labels or voxel count')
        total += int(n)
        if not r:
            unknown += int(n)
            continue
        before[int(i), int(r)] += int(n)
        after[int(f), int(r)] += int(n)
    initial = assignment_error(before); final = assignment_error(after)
    outcome = ('exact' if final == 0 else 'partial' if final < initial else
               'unchanged' if final == initial else 'worsened')
    return dict(initial_error_voxels=initial, final_error_voxels=final,
                net_removed_voxels=initial-final, measured_voxels=total,
                scored_voxels=total-unknown, unknown_reference_voxels=unknown,
                known_scope_outcome=outcome,
                outcome='unverified' if unknown or not total else outcome,
                error_reduction_percent=100*(initial-final)/initial if initial else None)


def score_maps(initial, final, reference, voxel_counts):
    if not (set(initial) == set(final) == set(reference) == set(voxel_counts)):
        raise ValueError('Complete aligned memberships, references and measured weights required')
    return score_cells((initial[a], final[a], reference[a], voxel_counts[a]) for a in initial)
