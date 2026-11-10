"""Freeze native-resolution CB2 errors and native controls before model inference.

Selection uses source test membership, stable hash order, disjoint crops and
metadata identities. Invalid geometry is rejected before any model is invoked.
Controls certify their target partition only, not every neuron in the crop.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

KINDS = ('merge_edit', 'adjacent_control', 'synapse_control', 'split_edit', 'junction_control')
ERRORS = {'merge_edit', 'split_edit'}

class InvalidCase(ValueError):
    """Source/geometry failure eligible for a documented pre-inference replacement."""


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def center(row):
    m = row['metadata']
    key = 'synapse_ctr_pt_nm' if row['sample_type'] == 'synapse_control' else 'interface_point_nm'
    point = m.get(key) or m.get('edit_point_nm')
    if point is None or np.asarray(point).shape != (3,) or not np.isfinite(point).all():
        raise InvalidCase('Missing or invalid target coordinate')
    return np.asarray(point, float)


def identities(row):
    m = row['metadata']
    values = [m.get('root_id')] + (m.get('root_ids') or [])
    for k in ('before_root_ids', 'after_root_ids'):
        values += m.get(k) or []
    return {str(x) for x in values if x is not None and str(x) != '0'}


def select(rows, excluded, per_kind=48, extent=6000, seed='cb2-balanced-80-v1'):
    used, operations, boxes = set(), set(), []
    for a in excluded:
        operations.add(str(a['operation_id']))
        used.add(str(a.get('source_root_id')))
        for k in ('before_certified_composition', 'after_certified_composition'):
            used.update(a.get(k, {}))
            for values in a.get(k, {}).values():
                used.update(x for x in values if x != '0')
        boxes.append(np.asarray(a['crop_bounds_nm']))
    selected, counts = [], Counter()
    ordered = sorted(rows, key=lambda r: hashlib.sha256((seed+r['combined_sample_hash']).encode()).hexdigest())
    for r in ordered:
        kind = r['sample_type']; m = r['metadata']; op = str(m['operation_id'])
        if r['split'] != 'test' or kind not in KINDS or counts[kind] >= per_kind:
            continue
        roots = identities(r)
        if not roots or roots & used or op in operations:
            continue
        point = center(r); box = np.array([point-extent/2-40, point+extent/2+40])
        if any(np.all(box[0]<b[1]) and np.all(b[0]<box[1]) for b in boxes):
            continue
        selected.append(r); counts[kind] += 1
        used.update(roots); operations.add(op); boxes.append(box)
    if any(counts[k] != per_kind for k in KINDS):
        raise ValueError(f'Insufficient disjoint held-out candidates: {counts}')
    return selected


def target_reference(atoms, before, after, target_roots, kind):
    target = np.isin(before, np.asarray(target_roots, np.uint64))
    if not target.any() or any(not np.any(before == int(r)) for r in target_roots):
        raise InvalidCase('Target root missing from crop at source timestamp')
    truth = after if kind in ERRORS else before
    return np.where(target, truth, 0), target


def download(volume, origin, upper, resolution, path):
    from cloudvolume import Bbox
    mip = next(i for i,s in enumerate(volume.info['scales']) if s['resolution'] == resolution)
    lo = (origin / resolution).astype(int); hi = (upper / resolution).astype(int)
    scale = volume.info['scales'][mip]
    start = np.asarray(scale['voxel_offset']); stop = start + scale['size']
    if np.any(lo < start) or np.any(hi > stop):
        raise InvalidCase('Crop outside source image bounds')
    dtype = np.dtype(volume.info['data_type'])
    array = np.lib.format.open_memmap(str(path)+'.partial', mode='w+', dtype=dtype, shape=tuple(hi-lo))
    for z in range(lo[2], hi[2], 8):
        end = min(z+8, hi[2]); a = lo.copy(); b = hi.copy(); a[2]=z; b[2]=end
        raw = np.asarray(volume.download(Bbox(a,b), mip=mip, agglomerate=False, coord_resolution=volume.meta.resolution(mip)))
        array[:,:,z-lo[2]:end-lo[2]] = raw[...,0] if raw.ndim == 4 else raw
    array.flush(); del array
    Path(str(path)+'.partial').replace(path)


def owners_at(client, atoms, timestamp):
    owners = np.empty(len(atoms), np.uint64)
    for start in range(0, len(atoms), 50000):
        owners[start:start+50000] = client.chunkedgraph.get_roots(atoms[start:start+50000], timestamp=timestamp)
    if np.any(owners == 0):
        raise InvalidCase('Missing historical ownership')
    return owners


def prepare(row, output, template, reference):
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume
    from proofreading import Dataset
    from proofreading.evaluate import freeze, metrics
    from prepare_cb2_practice import reference_labels
    from prepare_matched_cb2 import coarser_labels
    kind=row['sample_type']; m=row['metadata']; point=center(row)
    case_id='cb2-'+row['combined_sample_hash']
    dest=output/case_id; pub=dest/'public'; ev=dest/'evaluator'
    pub.mkdir(parents=True, exist_ok=False); ev.mkdir()
    client=CAVEclient('minnie65_public')
    seg=client.info.segmentation_cloudvolume(agglomerate=False,progress=False,parallel=False)
    em_config=json.loads(template.read_text())['em_pyramid']
    em=CloudVolume(em_config['source'],progress=False,parallel=False)
    align=np.array([32,32,40]); origin=np.floor((point-3000)/align)*align; upper=np.ceil((point+3000)/align)*align
    download(seg,origin,upper,[8,8,40],pub/'atoms.npy')
    cube=np.load(pub/'atoms.npy',mmap_mode='r')
    ids,counts=np.unique(cube,return_counts=True); atoms=ids[ids!=0]
    op=str(m['operation_id'])
    if kind in ERRORS:
        details=client.chunkedgraph.get_operation_details([int(op)])
        detail=details.get(op,details.get(int(op)))
        if not detail or detail.get('operation_status',0)!=0:
            raise InvalidCase('Source operation missing or unsuccessful')
        timestamp=detail.get('operation_ts',detail.get('timestamp'))
        if isinstance(timestamp,str):timestamp=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
        if not isinstance(timestamp,datetime):raise InvalidCase('Operation timestamp unavailable')
        if timestamp.tzinfo is None:timestamp=timestamp.replace(tzinfo=timezone.utc)
        before=owners_at(client,atoms,timestamp-timedelta(microseconds=1))
        after=owners_at(client,atoms,timestamp+timedelta(microseconds=1))
        target_roots=m['before_root_ids']; expected=np.asarray(m['after_root_ids'],np.uint64)
        affected=np.isin(before,np.asarray(target_roots,np.uint64))
        if not np.all(np.isin(after[affected],expected)) or any(not np.any(after[affected]==r) for r in expected):
            raise InvalidCase('Recorded correction does not match historical ownership in crop')
    else:
        timestamp=datetime.fromtimestamp(m.get('cutout_timestamp') or m['timestamp'],tz=timezone.utc)
        before=owners_at(client,atoms,timestamp); after=before
        target_roots=m['root_ids']
        expected_count=1 if kind=='junction_control' else 2
        if len(set(target_roots))!=expected_count:raise InvalidCase('Invalid control target identities')
    truth,target=target_reference(atoms,before,after,target_roots,kind)
    for name,values in [('before',before),('after',after)]:np.savez(ev/(name+'.npz'),atoms=atoms,segments=values)
    np.savez(pub/'partition.npz',atoms=atoms,segments=before)
    download(em,origin,upper,[32,32,40],pub/'em.npy')
    levels=[]
    for xy in (16,32):
        name=f'atoms-{xy}.npy';np.save(pub/name,coarser_labels(cube,xy//8))
        levels.append({'path':name,'resolution_nm':[xy,xy,40],'origin_nm':origin.tolist()})
    config={'name':'local-proofreading','partition':'partition.npz',
            'atoms':{'path':'atoms.npy','resolution_nm':[8,8,40],'origin_nm':origin.tolist()},
            'em':{'path':'em.npy','resolution_nm':[32,32,40],'origin_nm':origin.tolist()},
            'segmentation_pyramid':levels,'em_pyramid':em_config}
    save(pub/'dataset.json',config)
    # Map each Z plane to avoid a volume-sized inverse index in memory.
    known=np.load(reference/'atoms.npy',mmap_mode='r'); roots=np.load(reference/'reference.npy',mmap_mode='r')
    certified=reference_labels(atoms,known,roots)
    scorers=['human']; certified_count=0
    for name,values in [('human',truth),('certified',certified)]:
        lookup=np.zeros(len(ids),np.uint64);lookup[ids!=0]=values
        ref=np.lib.format.open_memmap(ev/(name+'-reference.npy'),mode='w+',dtype=np.uint64,shape=cube.shape)
        scope=np.lib.format.open_memmap(ev/(name+'-scope.npy'),mode='w+',dtype=bool,shape=cube.shape)
        for z in range(cube.shape[2]):
            ref[:,:,z]=lookup[np.searchsorted(ids,cube[:,:,z])]
            scope[:,:,z]=ref[:,:,z]!=0 if name=='human' else cube[:,:,z]!=0
        if name=='certified':certified_count=int(counts[ids!=0][certified!=0].sum())
        ref.flush();scope.flush();del ref,scope
    if certified_count:scorers.append('certified')
    dataset=Dataset.load(pub/'dataset.json')
    for name in scorers:freeze(dataset,ev/(name+'-reference.npy'),ev/(name+'-scope.npy'),ev/(name+'-frozen.json'))
    ref=np.load(ev/'human-reference.npy',mmap_mode='r'); scope=np.load(ev/'human-scope.npy',mmap_mode='r')
    initial=metrics(cube,dataset.initial,ref,scope)
    oracle=metrics(cube,dict(zip(map(int,atoms),map(int,after))),ref,scope)
    if oracle['vi_total_bits']!=0 or ((initial['vi_total_bits']>0)!=(kind in ERRORS)):
        raise InvalidCase('Scoring does not distinguish the recorded error/control')
    candidates,weights=np.unique(before[target],return_counts=True)
    seed=int(candidates[np.argmax(weights)])
    save(pub/'task.json',{'center_nm':point.tolist(),'seed_segment':str(seed),'extent_nm':float(max(upper-origin))})
    record={'case_id':case_id,'operation_id':op,'sample_hash':row['combined_sample_hash'],'kind':kind,
            'variant':'before' if kind in ERRORS else 'native-control','split':'test',
            'dataset_sha256':dataset.fingerprint,'scorers':scorers,'human_initial':initial,'human_oracle':oracle,
            'certified_voxels':certified_count,'certified_unavailable':None if certified_count else 'No certified reference coverage from the frozen 80um lookup; biological identity score unavailable.',
            'reference_note':'human score means source target-partition agreement: recorded correction for errors, unchanged source target roots for native controls. It does not establish whole-crop correctness.',
            'timestamp':timestamp.isoformat(),'crop_bounds_nm':[origin.tolist(),upper.tolist()],
            'source_record':row}
    save(ev/'provenance.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select');s.add_argument('candidates',type=Path);s.add_argument('exclusions',type=Path);s.add_argument('output',type=Path)
    s.add_argument('--reserve-per-kind',type=int,default=48)
    q=sub.add_parser('prepare');q.add_argument('selection',type=Path);q.add_argument('output',type=Path);q.add_argument('template',type=Path);q.add_argument('reference',type=Path);q.add_argument('--kind',choices=KINDS,required=True)
    f=sub.add_parser('finalize');f.add_argument('output',type=Path)
    a=p.parse_args()
    if a.command=='select':
        if a.output.exists():raise ValueError('Selection already exists')
        rows=json.loads(a.candidates.read_text())['candidates']; exclusions=json.loads(a.exclusions.read_text())
        selected=select(rows,exclusions,a.reserve_per_kind)
        save(a.output,{'seed':'cb2-balanced-80-v1','source':str(a.candidates),'source_sha256':hashlib.sha256(a.candidates.read_bytes()).hexdigest(),'exclusions_sha256':hashlib.sha256(a.exclusions.read_bytes()).hexdigest(),'candidates':selected})
        print(Counter(r['sample_type'] for r in selected))
    elif a.command=='prepare':
        a.output.mkdir(parents=True,exist_ok=True); report=a.output/(a.kind+'-prepared.json')
        if report.exists():raise ValueError('Kind already prepared')
        records=[];rejected=[]
        for row in json.loads(a.selection.read_text())['candidates']:
            if row['sample_type']!=a.kind:continue
            try:record=prepare(row,a.output,a.template,a.reference)
            except InvalidCase as exc:
                rejected.append({'sample_hash':row['combined_sample_hash'],'reason':str(exc)})
                save(a.output/(a.kind+'-rejected.json'),rejected)
                print(json.dumps({'rejected':rejected[-1]}),flush=True);continue
            records.append(record);save(a.output/(a.kind+'-progress.json'),records)
            print(json.dumps({'prepared':record['case_id'],'kind':a.kind,'count':len(records)}),flush=True)
            if len(records)==16:break
        if len(records)!=16:raise ValueError('Fewer than 16 valid cases; no inference allowed')
        save(report,records)
    else:
        records=[]
        for kind in KINDS:
            part=json.loads((a.output/(kind+'-prepared.json')).read_text())
            if len(part)!=16 or any(r['kind']!=kind for r in part):raise ValueError('Unbalanced preparation')
            records+=part
        if len({r['case_id'] for r in records})!=80:raise ValueError('Duplicate cases')
        if (a.output/'index.json').exists():raise ValueError('Already finalized')
        save(a.output/'index.json',records)
        print('Frozen 80 cases',flush=True)

if __name__=='__main__':main()
