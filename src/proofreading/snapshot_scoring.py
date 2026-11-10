"""Evaluator-only scoring against a pinned supervoxel-to-root snapshot.

Retrospective references remain separate from preregistered CB2 human-edit
scores. No reference labels are exposed to the agent. No geometry is fetched.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .compact_scoring import read, score
from .evaluate import file_hash
from .streamed_evaluate import project_rules


def timestamp(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Reference timestamp must include a timezone')
    return dt.astimezone(timezone.utc)


def remap(data, roots):
    """Preserve scope and voxel weights; root zero explicitly means unknown."""
    target = data['target'] > 0
    roots = np.asarray(roots)
    if roots.shape != (int(target.sum()),) or roots.dtype.kind not in 'iu':
        raise ValueError('Root assignments must align with scoped supervoxels')
    if roots.dtype.kind == 'i' and np.any(roots < 0):
        raise ValueError('Negative root IDs')
    known = roots != 0
    counts = data['target'][target][known]
    if not len(counts):
        raise ValueError('No known snapshot reference material')
    scope = int(data['scope_voxels'])
    if sum(map(int, counts)) > scope:
        raise ValueError('Target counts exceed scoring scope')
    return {**data, 'cell_atoms': data['atoms'][target][known],
            'reference': roots[known].astype(np.uint64), 'count': counts,
            'unknown_voxels': np.asarray(scope - sum(map(int, counts)), np.uint64)}


def prepare(source_spec, output, client, reference_time, *, materialization_version=None,
            batch_size=10000, scope='target', workers=1):
    """Query one explicit timestamp and create a new, immutable score package."""
    source_spec = Path(source_spec).resolve()
    output = Path(output).resolve()
    when = timestamp(reference_time)
    if when > datetime.now(timezone.utc):
        raise ValueError('Reference timestamp is in the future')
    if batch_size < 1 or not 1 <= workers <= 8:
        raise ValueError('Positive batch size and 1–8 workers required')
    old = json.loads(source_spec.read_text())
    if old.get('schema') != 'streamed-frozen-v2':
        raise ValueError('Expected compact CB2 scoring package')
    name = old['counts']
    if Path(name).name != name:
        raise ValueError('Counts must be beside source contract')
    counts_file = source_spec.parent / name
    if file_hash(counts_file) != old['counts_sha256']:
        raise ValueError('Source scoring counts changed')
    data = read(counts_file)
    task_roots = sorted(set(map(int, data['initial'][data['target'] > 0])))
    if scope not in ('target', 'full-cube'):
        raise ValueError('Scope must be target or full-cube')
    if scope == 'full-cube':
        # total contains every nonzero SV in the original fixed scoring grid,
        # including material outside the old target mask. Never whole-root size.
        data = {**data, 'target': data['total'].copy(),
                'scope_voxels': np.asarray(sum(map(int, data['total'])), np.uint64)}
    atoms = data['atoms'][data['target'] > 0]
    if output.exists() or output.with_suffix('.counts.npz').exists():
        raise FileExistsError(output)
    def lookup(chunk):
        values = np.asarray(client.chunkedgraph.get_roots(chunk, timestamp=when))
        if values.shape != chunk.shape or values.dtype.kind not in 'iu':
            raise ValueError('Invalid root lookup response')
        if values.dtype.kind == 'i' and np.any(values < 0):
            raise ValueError('Negative root lookup response')
        return values.astype(np.uint64)
    chunks = [atoms[start:start + batch_size] for start in range(0, len(atoms), batch_size)]
    if workers == 1:
        roots = list(map(lookup, chunks))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            roots = list(pool.map(lookup, chunks))
    mapped = remap(data, np.concatenate(roots))
    counts = output.with_suffix('.counts.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    with counts.open('xb') as f:
        np.savez_compressed(f, **mapped)
    spec = {'schema': 'timestamped-sv-reference-v1',
            'dataset_sha256': old['dataset_sha256'],
            'counts': counts.name, 'counts_sha256': file_hash(counts),
            'source_spec_sha256': file_hash(source_spec),
            'datastack': client.datastack_name,
            'reference_timestamp': when.isoformat(),
            'materialization_version': materialization_version,
            'prepared_at_unix': time.time(),
            'scope_note': ('All nonzero supervoxels in the original fixed cube'
                           if scope == 'full-cube' else old['scope_note']),
            'scope_kind': scope,
            'task_roots': task_roots,
            'reference_status': 'Timestamped segmentation; full proofreading is not certified.',
            'reference_visible_to_agent': False,
            'weighting': 'Original in-scope voxel counts per supervoxel; no connectivity relabeling.'}
    with output.open('x') as f:
        f.write(json.dumps(spec, indent=2) + '\n')
    return spec


def evaluate_snapshot(episode, spec_file):
    """Score saved final rules; preserve the original episode and its scores."""
    episode = Path(episode)
    path = Path(spec_file).resolve()
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'timestamped-sv-reference-v1':
        raise ValueError('Expected timestamped supervoxel reference')
    meta = json.loads((episode / 'session.json').read_text())
    fingerprint = spec['dataset_sha256']
    if meta['dataset_sha256'] != fingerprint:
        raise ValueError('Episode/reference dataset mismatch')
    name = spec['counts']
    if Path(name).name != name:
        raise ValueError('Counts must be beside reference contract')
    counts = path.parent / name
    if file_hash(counts) != spec['counts_sha256']:
        raise ValueError('Snapshot reference changed')
    data = read(counts)
    initial = dict(zip(map(int, data['atoms']), map(int, data['initial'])))
    with np.load(episode / 'partition.npz', allow_pickle=False) as z:
        if z['format'].item() != 'streaming-rules-v1' or z['dataset_sha256'].item() != fingerprint:
            raise ValueError('Partition/reference identity mismatch')
        rules = json.loads(z['rules'].item())
    result = score(data, initial, project_rules(initial, rules))
    for kind in ('merge', 'split'):
        key = f'vi_{kind}_bits'
        result[f'{kind}_vi_improvement_bits'] = result['initial'][key] - result['final'][key]
    return {**result, 'reference_timestamp': spec['reference_timestamp'],
            'materialization_version': spec['materialization_version'],
            'reference_status': spec['reference_status'],
            'reference_visible_to_agent': False,
            'prepared_before_episode': spec['prepared_at_unix'] <= meta['created_at_unix'],
            'evaluation_kind': 'timestamped-reference-rescore',
            'reference_spec_sha256': file_hash(path),
            'scope_note': spec['scope_note'], 'outside_reference_material': 'unscored'}
