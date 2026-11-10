"""Frozen voxel reassignment scoring; never changes the agent's observations."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from .expanded_volume import assignment_error

from .compact_scoring import read
from .evaluate import file_hash
from .snapshot_scoring import evaluate_snapshot
from .streamed_evaluate import project_rules


def error_volume(data, partition):
    cells = Counter()
    for atom, reference, count in zip(data['cell_atoms'], data['reference'], data['count']):
        cells[int(partition[int(atom)]), int(reference)] += int(count)
    return assignment_error(cells), sum(cells.values())


def evaluate_volume(episode, reference):
    """Initial, model, and optional human edit use identical known-reference voxels."""
    reference = Path(reference).resolve()
    spec = json.loads(reference.read_text())
    result = evaluate_snapshot(episode, reference)  # validates identity and counts digest
    data = read(reference.parent / spec['counts'])
    initial = dict(zip(map(int, data['atoms']), map(int, data['initial'])))
    with np.load(Path(episode) / 'partition.npz', allow_pickle=False) as z:
        rules = json.loads(z['rules'].item())
    final = project_rules(initial, rules)
    human = None
    if 'human_partition' in spec:
        name = spec['human_partition']
        if Path(name).name != name:
            raise ValueError('Human partition must be beside reference')
        path = reference.parent / name
        if file_hash(path) != spec['human_partition_sha256']:
            raise ValueError('Human partition changed')
        with np.load(path, allow_pickle=False) as z:
            human = dict(zip(map(int, z['atoms']), map(int, z['segments'])))
        if set(human) != set(initial):
            raise ValueError('Human partition must cover all scoring atoms')
    full_cube = spec.get('scope_kind') == 'full-cube'
    expanded_roots = []
    roots = set()
    if full_cube:
        # Coverage is the whole cube; the benchmark denominator is task-related
        # material plus the retained edits. Unrelated background errors must not
        # dilute the task score. Cohort roots freeze a common scope across models.
        roots = set(map(int, spec['task_roots'])) | set(map(int, spec.get('benchmark_scope_roots', [])))
        edited = {initial[a] for a in initial if final[a] != initial[a]}
        if human is not None:
            edited.update(initial[a] for a in initial if human[a] != initial[a])
        expanded_roots = sorted(edited - roots)
        roots.update(edited)
        selected = np.isin(data['initial'], np.asarray(sorted(roots), np.uint64))
        cells = np.isin(data['cell_atoms'], data['atoms'][selected])
        measured = sum(map(int, data['total'][selected]))
        known = sum(map(int, data['count'][cells]))
        data = {**data, **{k:data[k][cells] for k in ('cell_atoms','reference','count')},
                'unknown_voxels':np.asarray(measured-known,np.uint64)}
    before, total = error_volume(data, initial)
    after, _ = error_volume(data, final)
    unknown = int(data['unknown_voxels'])
    outcome = ('exact' if after == 0 else 'partial' if after < before else
               'unchanged' if after == before else 'worsened')
    volume = {'initial_error_voxels': before, 'final_error_voxels': after,
              'scored_voxels': total, 'net_removed_voxels': before - after,
              'error_reduction_percent': 100 * (before - after) / before if before else None,
              'unknown_reference_voxels': unknown,
              'known_scope_outcome': outcome,
              'outcome': 'unverified' if unknown or not total else outcome,
              'scope_kind': 'balanced-in-cube' if full_cube else 'target',
              'scope_roots': sorted(roots) if full_cube else None,
              'scope_expanded_roots': expanded_roots,
              'accounts_for_in_cube_additions': full_cube,
              'coverage_warning': None if full_cube else
                  'Legacy target-only reference: wrong additions outside the target mask are unscored. Upgrade the bundle before comparing models.',
              'method': 'Minimum voxel reassignment under one-to-one maximum-overlap segment matching',
              'scope': ('Task and cohort roots plus retained model/human edits inside the original cube'
                        if full_cube else spec['scope_note'])}
    size = spec.get('voxel_size_um3')
    if size is not None:
        if not np.isfinite(size) or size <= 0:
            raise ValueError('voxel_size_um3 must be finite and positive')
        for key in ('initial_error', 'final_error', 'scored', 'net_removed'):
            volume[key + '_um3'] = volume[key + '_voxels'] * size
    if human is not None:
        human_error, _ = error_volume(data, human)
        volume['human_error_voxels'] = human_error
        if size is not None:
            volume['human_error_um3'] = human_error * size
        volume['human_error_reduction_percent'] = 100 * (before-human_error)/before if before else None
        volume['model_vs_human_improvement_percent'] = 100*(before-after)/(before-human_error) if before>human_error else None
    return {**result, 'volume': volume}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('reference', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = evaluate_volume(a.episode, a.reference)
    text = json.dumps(result, indent=2) + '\n'
    if a.out:
        a.out.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
