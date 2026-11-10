"""Prepare compact, source-backed CB2 examples without any model inference.

Only native segmentation slabs are downloaded for counting. EM and meshes are
loaded by a preflight render; full temporary EM/segmentation cubes are not saved.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .compact_scoring import read, score
from .evaluate import file_hash
from .snapshot_scoring import prepare as prepare_snapshot, timestamp


class InvalidCase(ValueError):
    """Documented source-data ineligibility, decided without model results."""


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def count_slabs(slabs):
    counts = Counter()
    for slab in slabs:
        values = np.asarray(slab)
        if values.dtype.kind not in 'ui' or (values.dtype.kind == 'i' and np.any(values < 0)):
            raise ValueError('Segmentation must contain nonnegative integer IDs')
        ids, weights = np.unique(values, return_counts=True)
        counts.update({int(a): int(n) for a, n in zip(ids, weights) if a})
    if not counts:
        raise InvalidCase('Scoring crop has no foreground')
    atoms = np.asarray(sorted(counts), np.uint64)
    return atoms, np.asarray([counts[int(a)] for a in atoms], np.uint64)


def native_counts(volume, origin, upper, resolution=(8, 8, 40), slab_depth=8):
    from cloudvolume import Bbox
    resolution = np.asarray(resolution)
    origin, upper = np.asarray(origin), np.asarray(upper)
    if slab_depth < 1 or np.any(upper <= origin) or np.any(origin % resolution) or np.any(upper % resolution):
        raise ValueError('Positive, native-grid-aligned crop and slab depth required')
    mip = next(i for i, s in enumerate(volume.info['scales']) if np.array_equal(s['resolution'], resolution))
    lo = (origin / resolution).astype(int)
    hi = (upper / resolution).astype(int)
    scale = volume.info['scales'][mip]
    start = np.asarray(scale['voxel_offset'])
    if np.any(lo < start) or np.any(hi > start + scale['size']):
        raise InvalidCase('Crop outside source image bounds')
    def slabs():
        for z in range(lo[2], hi[2], slab_depth):
            a, b = lo.copy(), hi.copy()
            a[2], b[2] = z, min(z + slab_depth, hi[2])
            raw = np.asarray(volume.download(Bbox(a, b), mip=mip, agglomerate=False,
                                            coord_resolution=resolution))
            if raw.ndim == 4 and raw.shape[-1] == 1:
                raw = raw[..., 0]
            if raw.shape != tuple(b - a):
                raise ValueError('Source returned an unexpected slab shape')
            yield raw
    return count_slabs(slabs())


def roots_at(client, atoms, when, batch_size=10000):
    roots = []
    for start in range(0, len(atoms), batch_size):
        chunk = atoms[start:start + batch_size]
        values = np.asarray(client.chunkedgraph.get_roots(chunk, timestamp=when))
        if values.shape != chunk.shape or values.dtype.kind not in 'ui':
            raise ValueError('Invalid historical ownership response')
        if np.any(values == 0) or (values.dtype.kind == 'i' and np.any(values < 0)):
            raise InvalidCase('Missing historical ownership')
        roots.append(values.astype(np.uint64))
    return np.concatenate(roots)


def human_counts(atoms, counts, before, after, targets):
    """Equivalent to compact_scoring.pack on the native target mask."""
    values = [np.asarray(x) for x in (atoms, counts, before, after)]
    if any(x.dtype.kind not in 'ui' or (x.dtype.kind == 'i' and np.any(x < 0)) for x in values):
        raise ValueError('Nonnegative integer ownership/count arrays required')
    atoms, counts, before, after = [x.astype(np.uint64) for x in values]
    if not (atoms.shape == counts.shape == before.shape == after.shape) or atoms.ndim != 1:
        raise ValueError('Aligned one-dimensional ownership/count arrays required')
    if not len(atoms) or np.any(atoms == 0) or np.any(atoms[1:] <= atoms[:-1]) or np.any(counts == 0):
        raise ValueError('Sorted positive atoms and measured counts required')
    if np.any(before == 0) or np.any(after == 0):
        raise InvalidCase('Missing historical ownership')
    target = np.isin(before, np.asarray(targets, np.uint64))
    if any(not np.any(before == int(r)) for r in targets) or not target.any():
        raise InvalidCase('Target root missing from scoring crop')
    return dict(atoms=atoms, initial=before, total=counts,
                target=np.where(target, counts, np.uint64(0)), cell_atoms=atoms[target],
                reference=after[target], count=counts[target],
                scope_voxels=np.asarray(sum(map(int, counts[target])), np.uint64),
                unknown_voxels=np.asarray(0, np.uint64))


def prepare_case(row, output, template, reference_time, *, client=None,
                 materialization_version=None, preflight=True):
    """Build a single-case benchmark and return its public index record.

    No provider credentials or model transport are accepted. Preflight invokes
    the environment's explicit preflight-only branch, never its model harness.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if row['sample_type'] not in ('merge_edit', 'split_edit'):
        raise ValueError('Only recorded merge/split operations are supported')
    if row.get('cube_split') != 'val':
        raise ValueError('An authoritative cube-validation selection is required')
    key = row['combined_sample_hash']
    if not isinstance(key, str) or not key or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Expected a hexadecimal source sample hash')
    m = row['metadata']
    point = np.asarray(m.get('interface_point_nm') or m.get('edit_point_nm'), float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise InvalidCase('Missing or invalid target coordinate')
    reference_time = timestamp(reference_time)
    if client is None:
        from caveclient import CAVEclient
        client = CAVEclient('minnie65_public', auth_token=os.environ.get('CAVE_API_TOKEN'))
    operation = str(m['operation_id'])
    details = client.chunkedgraph.get_operation_details([int(operation)])
    detail = details.get(operation, details.get(int(operation)))
    if not detail or detail.get('operation_status', 0) != 0:
        raise InvalidCase('Source operation missing or unsuccessful')
    when = detail.get('operation_ts', detail.get('timestamp'))
    if isinstance(when, str):
        when = datetime.fromisoformat(when.replace('Z', '+00:00'))
    if not isinstance(when, datetime):
        raise InvalidCase('Operation timestamp unavailable')
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    config = json.loads(Path(template).read_text())
    if config.get('backend') != 'streamed-world-v1':
        raise ValueError('Expected a streamed dataset template')
    volume = client.info.segmentation_cloudvolume(agglomerate=False, progress=False, parallel=False)
    if str(volume.cloudpath).rstrip('/') != config['segmentation']['source'].rstrip('/'):
        raise ValueError('Template and source segmentation differ')
    align = np.array([32, 32, 40])
    origin = np.floor((point - 3000) / align) * align
    upper = np.ceil((point + 3000) / align) * align
    atoms, weights = native_counts(volume, origin, upper)
    before = roots_at(client, atoms, when - timedelta(microseconds=1))
    after = roots_at(client, atoms, when + timedelta(microseconds=1))
    targets = list(map(int, m['before_root_ids']))
    target = np.isin(before, np.asarray(targets, np.uint64))
    expected = np.asarray(m['after_root_ids'], np.uint64)
    if not np.all(np.isin(after[target], expected)) or any(not np.any(after[target] == r) for r in expected):
        raise InvalidCase('Recorded correction does not match historical ownership in crop')
    data = human_counts(atoms, weights, before, after, targets)
    initial = dict(zip(map(int, atoms), map(int, before)))
    human = dict(zip(map(int, atoms), map(int, after)))
    metrics = score(data, initial, human)
    if metrics['initial']['vi_total_bits'] <= 1e-10 or metrics['final']['vi_total_bits'] > 1e-10:
        raise InvalidCase('Scoring does not distinguish the recorded error')
    cid = 'cb2-' + row['combined_sample_hash'] + '-streamed'
    case = output / cid
    public, evaluator = case / 'public', case / 'evaluator'
    public.mkdir(parents=True)
    evaluator.mkdir()
    # Match the established 24.576 um initial view and original seed policy.
    candidates, n = np.unique(before[target], return_counts=True)
    seed = int(candidates[np.argmax(n)])
    config = {**config, 'name':'local-proofreading', 'timestamp':(when-timedelta(microseconds=1)).isoformat(),
              'cache_directory':'~/.cache/connectomeenv/streaming',
              'initial_view':{'center_nm':point.tolist(), 'width':768, 'height':768, 'pixel_size_nm':32}}
    identity = {k:v for k,v in config.items() if k != 'cache_directory'}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write(public / 'dataset.json', config)
    write(public / 'task.json', {'center_nm':point.tolist(), 'seed_segment':str(seed), 'extent_nm':24576})
    counts = evaluator / 'human-frozen.counts.npz'
    np.savez_compressed(counts, **data)
    source = evaluator / 'human-frozen.json'
    write(source, {'schema':'streamed-frozen-v2', 'dataset_sha256':fingerprint,
                   'counts':counts.name, 'counts_sha256':file_hash(counts),
                   'frozen_at_unix':time.time(), 'packaged_at_unix':time.time(),
                   'scope_note':'Original target roots inside the fixed native-resolution scoring cube',
                   'provenance':{'operation_id':operation, 'source_sample_hash':row['combined_sample_hash'],
                                 'preparation':'native slab counts; no resampling'}})
    ref = evaluator / 'snapshot-frozen.json'
    spec = prepare_snapshot(source, ref, client, reference_time.isoformat(),
                            materialization_version=materialization_version, scope='full-cube')
    frozen = read(evaluator / spec['counts'])
    known = set(map(int, frozen['cell_atoms']))
    if any(int(a) not in known for a in atoms[target]):
        raise InvalidCase('Incomplete snapshot reference coverage on target')
    from .release_scoring import error_volume
    target_cells = np.isin(frozen['cell_atoms'], atoms[target])
    local = {**frozen, **{k:frozen[k][target_cells] for k in ('cell_atoms','reference','count')}}
    if row['sample_type'] == 'split_edit' and error_volume(local, initial)[0] == 0:
        raise InvalidCase('Snapshot no longer labels this merge as erroneous')
    partition = evaluator / 'human-partition.npz'
    np.savez_compressed(partition, atoms=atoms, segments=after)
    spec.update(voxel_size_um3=8*8*40/1e9, human_partition=partition.name,
                human_partition_sha256=file_hash(partition), benchmark_scope_roots=sorted(set(targets)))
    write(ref, spec)
    record = {'case_id':cid, 'split':'test', 'cube_split':'val', 'cube_id':row['cube_id'],
              'dataset_sha256':fingerprint, 'scorers':['human'],
              'certified_unavailable':'Timestamped full-cube reference supplied separately; no certified proofread labels.',
              'kind':row['sample_type'], 'sample_hash':row['combined_sample_hash'],
              'operation_id':operation, 'timestamp':when.isoformat(), 'snapshot_reference':file_hash(ref),
              'task':'split_error' if row['sample_type']=='merge_edit' else 'merge_error'}
    write(evaluator / 'preparation.json', {'record':record, 'source_record':row,
          'crop_bounds_nm':[origin.tolist(), upper.tolist()], 'resolution_nm':[8,8,40],
          'foreground_atoms':len(atoms), 'foreground_voxels':sum(map(int, weights)),
          'model_calls':0, 'human_initial':metrics['initial'], 'human_oracle':metrics['final']})
    write(output / 'index.json', [record])
    if preflight:
        from .experiment import execute
        cfg = {'schema':1, 'mode':'cb2', 'benchmark':str(output), 'phase':'test', 'cases':[cid],
               'model':{'provider':'responses', 'name':'preflight-only-no-inference'}}
        write(output / 'preflight-config.json', cfg)
        execute(output / 'preflight-config.json', output / 'preflight', preflight=True)
    write(output / 'ready.json', {'record':record, 'preflight_passed':preflight, 'model_calls':0})
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('selection', type=Path, help='Single selected row JSON')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--reference-time', required=True)
    p.add_argument('--materialization-version', type=int)
    a = p.parse_args()
    print(json.dumps(prepare_case(json.loads(a.selection.read_text()), a.out, a.template,
                                 a.reference_time, materialization_version=a.materialization_version)))


if __name__ == '__main__':
    main()
