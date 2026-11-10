"""Data-only CB2 bundle expansion: frozen candidate pool, preparation waves, assembly."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import tarfile

from .bundle import verify
from .evaluate import file_hash
from .preparation import InvalidCase, prepare_case, write

TASKS = {'merge_edit':'split_error', 'split_edit':'merge_error'}


def point(row):
    m = row['metadata']
    return m.get('interface_point_nm') or m.get('edit_point_nm')


def roots(row):
    m = row['metadata']
    values = [m.get('root_id')] + list(m.get('root_ids') or [])
    for key in ('before_root_ids', 'after_root_ids'):
        values += list(m.get(key) or [])
    return {str(v) for v in values if v and str(v) != '0'}


def box(row):
    p = point(row)
    return ([x-3040 for x in p], [x+3040 for x in p])


def overlaps(a, b):
    return all(a[0][k] < b[1][k] and a[1][k] > b[0][k] for k in range(3))


def catalogue(rows, valid, geometry, excluded):
    """Authoritative cube validation overrides the unrelated parquet split."""
    selected = {}
    for row in rows:
        if row['source_archive'] != 'unified_mouse' or row['sample_type'] not in TASKS:
            continue
        m = row['metadata']
        if isinstance(m, str):
            m = json.loads(m)
        if str(m.get('operation_id')) not in valid:
            continue
        r = {**row, 'metadata':m}
        p = point(r)
        if not p or len(p) != 3 or not all(math.isfinite(x) for x in p):
            continue
        cube = tuple(math.floor((x-z)/geometry['edge_nm']) for x,z in zip(p, geometry['bbox_nm']['min']))
        if cube in excluded or not roots(r):
            continue
        r.update(cube_id=list(cube), cube_split='val')
        key = r['combined_sample_hash']
        if not isinstance(key, str) or not key or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Expected a hexadecimal source sample hash')
        if key in selected and selected[key] != r:
            raise ValueError('Conflicting duplicate source sample hash')
        selected[key] = r
    return selected


def make_plan(source, keys, geometry, exclusions, base, output, per_task=500):
    import pyarrow.parquet as pq
    base = Path(base).resolve()
    verify(base)
    validation = json.loads(Path(keys).read_text())
    excluded = json.loads(Path(exclusions).read_text())
    if file_hash(keys) != excluded['validation_keys_sha256']:
        raise ValueError('Authoritative validation keys changed')
    def rows():
        columns = ['metadata','combined_sample_hash','source_archive_sample_hash','source_archive','sample_type','split']
        for chunk in pq.ParquetFile(source).iter_batches(batch_size=4096, columns=columns):
            yield from chunk.to_pylist()
    pool = catalogue(rows(), set(map(str, validation['operation_bank'])),
                     json.loads(Path(geometry).read_text()), set(map(tuple, excluded['excluded_cubes'])))
    old = json.loads((base/'index.json').read_text())
    retained = [pool[r['sample_hash']] for r in old]
    if len({r['combined_sample_hash'] for r in retained}) != len(old):
        raise ValueError('Duplicate retained source samples')
    if any(n > per_task for n in Counter(r['sample_type'] for r in retained).values()):
        raise ValueError('Requested target is smaller than the retained benchmark')
    seed = 'portable-cb2-1000-v1'
    ordered = sorted(pool.values(), key=lambda r:hashlib.sha256((seed+r['combined_sample_hash']).encode()).hexdigest())
    plan = {'schema':'connectome-preparation-v1', 'seed':seed, 'per_task':per_task,
            'target_cases':2*per_task, 'base_bundle':str(base), 'base_bundle_sha256':file_hash(base/'bundle.json'),
            'validation_keys_sha256':file_hash(keys), 'geometry_sha256':file_hash(geometry),
            'exclusions_sha256':file_hash(exclusions), 'retained_rows':retained, 'candidates':ordered,
            'source':str(Path(source).resolve()), 'source_bytes':Path(source).stat().st_size,
            'source_mtime_ns':Path(source).stat().st_mtime_ns, 'model_calls':0,
            'selection_rule':'Prioritize split errors; deduplicate operations, root identities and 6.08um bounding boxes; replace only before inference.'}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, plan)
    return {'retained':len(retained), 'candidate_counts':dict(Counter(r['sample_type'] for r in ordered)),
            'target_cases':plan['target_cases'], 'model_calls':0}


def choose_wave(plan, ready, invalid, limit=8):
    """Failed data preparations release their reservations; no model outcomes used."""
    if limit < 1:
        raise ValueError('Positive wave size required')
    held = plan['retained_rows'] + list(ready)
    seen = {r['combined_sample_hash'] for r in held} | set(invalid)
    ops = {str(r['metadata']['operation_id']) for r in held}
    used = set().union(*(roots(r) for r in held)) if held else set()
    boxes = [box(r) for r in held]
    counts = Counter(r['sample_type'] for r in held)
    chosen = []
    for kind in TASKS:  # Split errors are the scarcer source pool.
        for row in plan['candidates']:
            if counts[kind] >= plan['per_task'] or len(chosen) >= limit:
                break
            if row['sample_type'] != kind or row['combined_sample_hash'] in seen:
                continue
            op = str(row['metadata']['operation_id'])
            rr, bb = roots(row), box(row)
            if op in ops or used & rr or any(overlaps(bb, old) for old in boxes):
                continue
            chosen.append(row)
            seen.add(row['combined_sample_hash']); ops.add(op); used.update(rr); boxes.append(bb)
            counts[kind] += 1
    return chosen


def state(plan, work):
    ready, invalid, failures = [], [], []
    for row in plan['candidates']:
        d = Path(work)/row['combined_sample_hash']
        if (d/'ready.json').exists():
            saved = json.loads((d/'ready.json').read_text())
            if saved['record']['sample_hash'] != row['combined_sample_hash'] or not saved['preflight_passed']:
                raise ValueError('Invalid preparation receipt')
            ready.append(row)
        elif (d/'invalid.json').exists():
            invalid.append(row['combined_sample_hash'])
        elif (d/'failure.json').exists():
            failures.append(row['combined_sample_hash'])
    return ready, invalid, failures


def archive_preflight(directory):
    """Retain audit images/logs in one verified archive instead of many files."""
    directory = Path(directory)
    target = directory.with_suffix('.tar.gz')
    temporary = target.with_suffix('.tmp')
    expected = {str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file()}
    with tarfile.open(temporary, 'w:gz') as archive:
        archive.add(directory, arcname='preflight')
    actual = {}
    with tarfile.open(temporary, 'r:gz') as archive:
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda:stream.read(1024*1024), b''):digest.update(chunk)
                actual[str(Path(member.name).relative_to('preflight'))] = digest.hexdigest()
    if actual != expected:
        raise ValueError('Preflight archive verification failed')
    temporary.replace(target)
    shutil.rmtree(directory)
    return {'path':str(target),'sha256':file_hash(target),'files':len(expected)}


def run_case(wave, index, work, template, reference_time, materialization_version=None):
    document = json.loads(Path(wave).read_text())
    row = document['rows'][index]
    root = Path(work).resolve()/row['combined_sample_hash']
    root.mkdir(parents=True, exist_ok=True)
    if (root/'ready.json').exists() or (root/'invalid.json').exists():
        return {'status':'already_prepared_or_rejected', 'model_calls':0}
    # This job has no model credentials and never calls a model harness.
    for key in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY','OPENROUTER_API_KEY','MOONSHOT_API_KEY','MODEL_API_KEY'):
        os.environ.pop(key, None)
    attempts = sorted(root.glob('attempt-*'))
    attempt = root/f'attempt-{len(attempts)+1:03d}'
    try:
        with tempfile.TemporaryDirectory(prefix='connectome-prepare-') as cache:
            old = {k:os.environ.get(k) for k in ('CONNECTOME_CACHE_DIRECTORY','CONNECTOME_MESH_CACHE','MPLCONFIGDIR')}
            os.environ.update(CONNECTOME_CACHE_DIRECTORY=cache+'/streaming', CONNECTOME_MESH_CACHE=cache+'/meshes', MPLCONFIGDIR=cache+'/matplotlib')
            try:
                record = prepare_case(row, attempt, template, reference_time,
                                      materialization_version=materialization_version)
            finally:
                for k,v in old.items():
                    if v is None:os.environ.pop(k, None)
                    else:os.environ[k] = v
        archive = archive_preflight(attempt/'preflight')
        case_directory = attempt/record['case_id']
        hashes = {str(p.relative_to(case_directory)):file_hash(p) for p in sorted(case_directory.rglob('*')) if p.is_file()}
        write(root/'ready.json', {'record':record, 'case_directory':str(case_directory), 'files':hashes, 'preflight_archive':archive,
                                 'preflight_passed':True, 'model_calls':0})
        return {'status':'ready','case_id':record['case_id'],'model_calls':0}
    except InvalidCase as exc:
        write(root/'invalid.json', {'reason':str(exc),'attempt':str(attempt),'model_calls':0})
        return {'status':'ineligible','reason':str(exc),'model_calls':0}
    except Exception as exc:
        write(root/'failure.json', {'error_type':type(exc).__name__, 'reason':str(exc),
                                   'attempt':str(attempt), 'model_calls':0})
        raise


def assemble(plan, work, output):
    base = Path(plan['base_bundle'])
    verify(base)
    if file_hash(base/'bundle.json') != plan['base_bundle_sha256']:
        raise ValueError('Retained bundle changed')
    ready, _, _ = state(plan, work)
    counts = Counter(r['sample_type'] for r in plan['retained_rows']+ready)
    if any(counts[k] != plan['per_task'] for k in TASKS):
        raise ValueError(f'Not ready to publish: {dict(counts)}')
    # Recheck all deduplication constraints before copying any new artifact.
    held = []
    for row in plan['retained_rows']+ready:
        if any(str(row['metadata']['operation_id']) == str(r['metadata']['operation_id'])
               or roots(row)&roots(r) or overlaps(box(row),box(r)) for r in held):
            raise ValueError('Duplicate operation/root or overlapping scoring crop')
        held.append(row)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(base, output)
    manifest = json.loads((output/'bundle.json').read_text())
    (output/'bundle.json').unlink()
    records = json.loads((output/'index.json').read_text())
    for row in ready:
        receipt = json.loads((Path(work)/row['combined_sample_hash']/'ready.json').read_text())
        record = receipt['record']
        directory = Path(receipt['case_directory'])
        actual = {str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file()}
        if actual != receipt['files']:
            raise ValueError('Prepared case changed after validation')
        shutil.copytree(receipt['case_directory'], output/record['case_id'])
        records.append(record)
    write(output/'index.json', records)
    example = json.loads((output/'example.json').read_text())
    example['cases'] = [r['case_id'] for r in records]
    write(output/'example.json', example)
    write(output/'preparation-summary.json', {'model_calls':0,'cases':len(records),
          'tasks':dict(Counter(r['task'] for r in records)), 'retained_cases':len(plan['retained_rows']),
          'preflight_validated_new_cases':len(ready), 'validation_keys_sha256':plan['validation_keys_sha256'],
          'seed':plan['seed'], 'selection_rule':plan['selection_rule']})
    manifest.update(cases=len(records), expanded_from_sha256=plan['base_bundle_sha256'],
                    scope_kind='balanced-in-cube', reference_coverage='full-cube')
    manifest['files'] = {str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'bundle.json', manifest)
    return verify(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q=sub.add_parser('plan')
    for key in ('source','keys','geometry','exclusions','base','out'):q.add_argument('--'+key,type=Path,required=True)
    q.add_argument('--per-task',type=int,default=500)
    q=sub.add_parser('wave');q.add_argument('plan',type=Path);q.add_argument('--work',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--size',type=int,default=8)
    q=sub.add_parser('case');q.add_argument('wave',type=Path);q.add_argument('--index',type=int,required=True)
    for key in ('work','template'):q.add_argument('--'+key,type=Path,required=True)
    q.add_argument('--reference-time',required=True);q.add_argument('--materialization-version',type=int)
    q=sub.add_parser('assemble');q.add_argument('plan',type=Path);q.add_argument('--work',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='plan':result=make_plan(a.source,a.keys,a.geometry,a.exclusions,a.base,a.out,a.per_task)
    elif a.command=='case':result=run_case(a.wave,a.index,a.work,a.template,a.reference_time,a.materialization_version)
    else:
        plan=json.loads(a.plan.read_text())
        if a.command=='assemble':result=assemble(plan,a.work,a.out)
        else:
            if a.out.exists():raise FileExistsError(a.out)
            ready,invalid,failures=state(plan,a.work)
            if failures:raise ValueError(f'Retry or investigate {len(failures)} infrastructure failures before selecting replacements')
            rows=choose_wave(plan,ready,invalid,a.size)
            write(a.out,{'plan_sha256':file_hash(a.plan),'rows':rows,'model_calls':0})
            result={'wave_cases':len(rows),'new_ready':len(ready),'ineligible':len(invalid),'model_calls':0}
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
