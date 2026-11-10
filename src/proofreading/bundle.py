"""Export, verify, and unpack portable streamed CB2 benchmark bundles."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from .evaluate import file_hash


def _write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def _beside(directory, name):
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Expected a plain asset filename')
    path = directory / name
    if path.is_symlink():
        raise ValueError('Symlinks are not bundle assets')
    return path


def export(manifest, output):
    """Freeze only allowlisted task/source/reference files, never run logs or secrets.

    Input is the preparation manifest (jobs with config/root and reference). Model
    duplicates are removed. Images and meshes stream from the pinned source at run
    time; this is not an offline volume download.
    """
    import numpy as np
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    jobs = json.loads(Path(manifest).read_text())['jobs']
    output.mkdir(parents=True)
    records = []; seen = set(); objective = None
    for job in jobs:
        cfg = json.loads((Path(job['root']) / 'config.json').read_text())
        benchmark = Path(cfg['benchmark'])
        index = json.loads((benchmark / 'index.json').read_text())
        for record in index:
            cid = record['case_id']
            if cid not in cfg['cases'] or cid in seen:
                continue
            seen.add(cid)
            source = _beside(benchmark, cid)
            dest = _beside(output, cid)
            (dest / 'public').mkdir(parents=True)
            (dest / 'evaluator').mkdir()
            dataset = json.loads((source / 'public/dataset.json').read_text())
            if dataset.get('backend') != 'streamed-world-v1':
                raise ValueError('Bundle export currently requires streamed-world-v1')
            identity = {k:v for k,v in dataset.items() if k != 'cache_directory'}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if fingerprint != record['dataset_sha256']:
                raise ValueError('Source identity changed')
            dataset['cache_directory'] = '~/.cache/connectomeenv/streaming'
            _write(dest / 'public/dataset.json', dataset)
            shutil.copyfile(source / 'public/task.json', dest / 'public/task.json')
            for name in record.get('scorers', ['human', 'certified']):
                if name not in ('human', 'certified'):
                    raise ValueError('Unknown scorer')
                path = source / 'evaluator' / (name + '-frozen.json')
                spec = json.loads(path.read_text())
                if spec['schema'] != 'streamed-frozen-v2':
                    raise ValueError('Export requires compact scoring contracts')
                counts = _beside(path.parent, spec['counts'])
                if file_hash(counts) != spec['counts_sha256']:
                    raise ValueError('Scoring counts changed')
                shutil.copyfile(path, dest / 'evaluator' / path.name)
                shutil.copyfile(counts, dest / 'evaluator' / counts.name)
            path = Path(job['reference']); spec = json.loads(path.read_text())
            counts = _beside(path.parent, spec['counts'])
            if file_hash(counts) != spec['counts_sha256'] or spec['dataset_sha256'] != fingerprint:
                raise ValueError('Snapshot reference changed')
            shutil.copyfile(counts, dest / 'evaluator' / counts.name)
            provenance = json.loads((source / 'evaluator/provenance.json').read_text())
            core = Path(provenance['source_case'])
            grid = json.loads((core / 'public/dataset.json').read_text())['atoms']
            spec['voxel_size_um3'] = float(np.prod(grid['resolution_nm'])) / 1e9
            from .compact_scoring import read
            frozen = read(counts)
            with np.load(core / 'evaluator/before.npz', allow_pickle=False) as z:
                original = dict(zip(map(int,z['atoms']),map(int,z['segments'])))
            if any(original.get(int(a)) != int(r) for a,r in zip(frozen['atoms'], frozen['initial'])):
                raise ValueError('Historical initial partition differs from reference')
            human = dest / 'evaluator/human-partition.npz'
            with np.load(core / 'evaluator/after.npz', allow_pickle=False) as z:
                np.savez_compressed(human, atoms=z['atoms'], segments=z['segments'])
            spec['human_partition'] = human.name
            spec['human_partition_sha256'] = file_hash(human)
            reference = dest / 'evaluator/snapshot-frozen.json'
            _write(reference, spec)
            portable_record = {k: record[k] for k in ('case_id', 'split', 'dataset_sha256', 'scorers',
                'certified_unavailable', 'kind', 'sample_hash', 'cube_split', 'timestamp') if k in record}
            records.append({**portable_record, 'snapshot_reference': file_hash(reference),
                            'task': 'split_error' if job['kind']=='merge_edit' else 'merge_error',
                            'cube_id': job.get('cube_id'), 'operation_id': job.get('operation_id')})
            if objective is not None and cfg.get('objective') != objective:
                raise ValueError('Cannot combine cases with different benchmark objectives')
            objective = cfg.get('objective', objective)
    if not records:
        raise ValueError('No cases selected')
    _write(output / 'index.json', records)
    _write(output / 'example.json', {'schema':1, 'mode':'cb2', 'benchmark':'.',
        'phase':records[0]['split'], 'cases':[r['case_id'] for r in records if r['split']==records[0]['split']],
        'model':{'provider':'responses','name':'gpt-6-astra','options':{'reasoning':{'effort':'high'}}},
        'budget':{'actions':50,'turns':50,'output_tokens':16384}, **({'objective':objective} if objective else {})})
    _write(output / 'bundle.json', {'schema':'connectome-bundle-v1', 'kind':'streamed-cb2',
        'cases':len(records), 'requires':['CAVE access to minnie65_public', 'network access to original imagery and segmentation'],
        'reference_scope':'Fixed original known-reference foreground; source exploration can extend outside scored region.',
        'data_rights':'Source data retain their original terms; the software MIT license does not relicense source data.',
        'files':{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return verify(output)


def verify(directory):
    directory = Path(directory).resolve()
    if any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Symlinks are not bundle assets')
    spec = json.loads((directory / 'bundle.json').read_text())
    if spec.get('schema') != 'connectome-bundle-v1':
        raise ValueError('Unsupported bundle schema')
    for name, digest in spec['files'].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or file_hash(path) != digest:
            raise ValueError('Bundle asset missing, unsafe, or changed: ' + name)
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != set(spec['files']) | {'bundle.json'}:
        raise ValueError('Unlisted files in bundle')
    return {'status':'verified', 'cases':spec['cases'], 'files':len(spec['files'])}


def unpack(archive, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar.getmembers():
            path = output / member.name
            if not (member.isfile() or member.isdir()) or not path.resolve().is_relative_to(output):
                raise ValueError('Unsafe archive entry')
        output.mkdir(parents=True)
        tar.extractall(output, filter='data')
    return verify(output)


def upgrade_reference(source, output, client, reference_time=None, *, workers=1, scope_map=None):
    """Copy a verified bundle and freeze all in-cube SVs at one explicit time.

    The original bundle is untouched. A failed lookup leaves no publishable
    bundle.json in the output; resume by choosing a new output directory.
    """
    from .snapshot_scoring import prepare
    source, output = Path(source).resolve(), Path(output).resolve()
    verify(source)
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output)
    manifest = json.loads((output / 'bundle.json').read_text())
    (output / 'bundle.json').unlink()
    records = json.loads((output / 'index.json').read_text())
    for record in records:
        directory = _beside(output, record['case_id']) / 'evaluator'
        old_path = directory / 'snapshot-frozen.json'
        old = json.loads(old_path.read_text())
        if file_hash(old_path) != record['snapshot_reference']:
            raise ValueError('Snapshot reference contract changed')
        if client.datastack_name != old['datastack']:
            raise ValueError('Reference datastack mismatch')
        source_spec = directory / 'human-frozen.json'
        upgraded = directory / 'full-cube-reference.json'
        spec = prepare(source_spec, upgraded, client,
                       reference_time or old['reference_timestamp'], scope='full-cube', workers=workers,
                       materialization_version=old.get('materialization_version') if reference_time is None else None)
        if scope_map is not None:
            roots = scope_map[record['case_id']]
            if not roots or any(type(x) is not int or not 0 < x < 2**64 for x in roots):
                raise ValueError('Scope map requires positive uint64 root IDs')
            spec['benchmark_scope_roots'] = sorted(set(roots))
        for key in ('voxel_size_um3', 'human_partition', 'human_partition_sha256'):
            if key in old:
                spec[key] = old[key]
        _write(old_path, spec)
        upgraded.unlink()
        record['snapshot_reference'] = file_hash(old_path)
        print(json.dumps({'upgraded_case': record['case_id']}), flush=True)
    _write(output / 'index.json', records)
    manifest['reference_scope'] = 'Task and cohort roots plus retained edits inside the original cube; full-cube reference coverage.'
    manifest['scope_kind'] = 'balanced-in-cube'
    manifest['reference_coverage'] = 'full-cube'
    manifest['upgraded_from_sha256'] = file_hash(source / 'bundle.json')
    manifest['files'] = {str(p.relative_to(output)):file_hash(p)
                         for p in sorted(output.rglob('*')) if p.is_file()}
    _write(output / 'bundle.json', manifest)
    return verify(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('export', 'unpack'):
        q=sub.add_parser(command);q.add_argument('source',type=Path);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('verify');q.add_argument('directory',type=Path)
    q=sub.add_parser('upgrade-reference', help='Freeze surrounding in-cube material for balanced scoring')
    q.add_argument('source', type=Path);q.add_argument('--out', type=Path, required=True)
    q.add_argument('--datastack', required=True);q.add_argument('--timestamp')
    q.add_argument('--workers', type=int, default=1)
    q.add_argument('--scope-map', type=Path, help='Case-ID to shared cohort root IDs for comparable benchmark scores')
    a=p.parse_args()
    if a.command == 'upgrade-reference':
        import os
        from caveclient import CAVEclient
        client = CAVEclient(a.datastack, auth_token=os.environ.get('CAVE_API_TOKEN'))
        print(json.dumps(upgrade_reference(a.source, a.out, client, a.timestamp, workers=a.workers,
                            scope_map=json.loads(a.scope_map.read_text()) if a.scope_map else None)))
        return
    print(json.dumps(verify(a.directory) if a.command=='verify' else
                     export(a.source,a.out) if a.command=='export' else unpack(a.source,a.out)))


if __name__ == '__main__':
    main()
