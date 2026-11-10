"""Materialize historical CB2 crops and audit their certified identities.

This stage produces candidates, not model results. Freeze scoring contracts before
inference. Test selection requires explicit practice exclusions. Reference files
are never placed in the public crop directory.
"""
import argparse
from datetime import datetime,timedelta,timezone
import json
import hashlib
from pathlib import Path
import time
import numpy as np


def reference_labels(atoms,known_atoms,known_roots):
    pos=np.searchsorted(known_atoms,atoms)
    valid=pos<len(known_atoms)
    valid[valid]&=known_atoms[pos[valid]]==atoms[valid]
    out=np.zeros(len(atoms),dtype=np.uint64);out[valid]=known_roots[pos[valid]]
    return out


def composition(owners,refs,counts,selected):
    result={}
    for sid in selected:
        mask=owners==sid;totals={}
        for ref,n in zip(refs[mask],counts[mask]):
            key=str(int(ref));totals[key]=totals.get(key,0)+int(n)
        result[str(sid)]=totals
    return result


def select_cases(rows,certified,split,per_kind,extent_nm,excluded):
    """Select without model outcomes; reject shared identities and overlapping crops."""
    by_hash={r['combined_sample_hash']:r for r in rows}
    used_ops={str(r['operation_id']) for r in excluded};used_roots=set();boxes=[]
    for audit in excluded:
        original=by_hash[audit['sample_hash']]['metadata']
        used_roots.add(str(original.get('root_id')))
        for key in ('before_certified_composition','after_certified_composition'):
            for composition in audit[key].values():used_roots.update(r for r in composition if r!='0')
        boxes.append(np.asarray(audit['crop_bounds_nm'],float))
    selected=[]
    for kind in ('merge_edit','split_edit'):
        count=0
        for row in sorted(rows,key=lambda x:x['combined_sample_hash']):
            m=row['metadata'];op=str(m['operation_id']);root=str(m.get('root_id') or 0)
            if row['split']!=split or row['sample_type']!=kind or op in used_ops:continue
            if int(root) not in certified or root in used_roots:continue
            point=np.asarray(m['interface_point_nm'],float)
            # Conservative 40nm margin covers voxel rounding on every axis.
            box=np.array([point-extent_nm/2-40,point+extent_nm/2+40])
            if any(np.all(box[0]<b[1]) and np.all(b[0]<box[1]) for b in boxes):continue
            selected.append(row);used_ops.add(op);used_roots.add(root);boxes.append(box)
            count+=1
            if count>=per_kind:break
        if count<per_kind:raise ValueError(f'Only {count} disjoint {split} {kind} cases available')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates',type=Path);p.add_argument('world',type=Path)
    p.add_argument('reference',type=Path);p.add_argument('certified_roots',type=Path)
    p.add_argument('output',type=Path);p.add_argument('--per-kind',type=int,default=2)
    p.add_argument('--extent-nm',type=float,default=6000)
    p.add_argument('--exclude-operation',action='append',default=[])
    p.add_argument('--split',choices=['train','test'],default='train')
    p.add_argument('--exclude-audits',type=Path)
    args=p.parse_args()
    from caveclient import CAVEclient
    client=CAVEclient('minnie65_public')
    source=json.loads(args.candidates.read_text())
    certified=set(map(int,np.load(args.certified_roots)))
    excluded=json.loads(args.exclude_audits.read_text()) if args.exclude_audits else []
    if args.split=='test' and not excluded:raise ValueError('Test preparation requires explicit practice exclusions')
    pool=[r for r in source['candidates'] if str(r['metadata']['operation_id']) not in args.exclude_operation]
    selected=select_cases(pool,certified,args.split,args.per_kind,args.extent_nm,excluded)
    manifest=json.loads((args.world/'manifest.json').read_text())
    res=np.asarray(manifest['resolution_xyz_nm'],float)
    origin=np.asarray(manifest['bounds_min_xyz'])*res
    seg=np.load(args.world/'seg.npy',mmap_mode='r');em=np.load(args.world/'em.npy',mmap_mode='r')
    known=np.load(args.reference/'atoms.npy',mmap_mode='r')
    roots=np.load(args.reference/'reference.npy',mmap_mode='r')
    args.output.mkdir(parents=True,exist_ok=True)
    reports=[]
    blocked_reference_roots=set()
    for audit in excluded:
        for key in ('before_certified_composition','after_certified_composition'):
            for values in audit[key].values():blocked_reference_roots.update(r for r in values if r!='0')
    for row in selected:
        started=time.time();m=row['metadata'];op=str(m['operation_id'])
        dest=args.output/op
        if dest.exists():raise ValueError('Refusing to overwrite historical case '+op)
        dest.mkdir();public=dest/'public';public.mkdir();private=dest/'evaluator';private.mkdir()
        point=np.asarray(m['interface_point_nm'],float)
        lo=np.floor((point-args.extent_nm/2-origin)/res).astype(int)
        hi=np.ceil((point+args.extent_nm/2-origin)/res).astype(int)
        if np.any(lo<0) or np.any(hi>seg.shape):raise ValueError('Crop crosses source bounds')
        slices=tuple(slice(int(l),int(h)) for l,h in zip(lo,hi))
        cube=np.asarray(seg[slices]);ids,inverse,counts=np.unique(cube,return_inverse=True,return_counts=True)
        foreground=ids!=0;atoms=ids[foreground]
        operation_details=client.chunkedgraph.get_operation_details([int(op)])
        details=operation_details.get(op,operation_details.get(int(op)))
        if details is None:raise ValueError('Operation details missing for '+op)
        timestamp=details.get('operation_ts',details['timestamp'])
        if isinstance(timestamp,str):timestamp=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
        if timestamp.tzinfo is None:timestamp=timestamp.replace(tzinfo=timezone.utc)
        if details.get('operation_status',0)!=0:raise ValueError('Unsuccessful source operation')
        partitions=[]
        for label,when in [('before',timestamp-timedelta(microseconds=1)),('after',timestamp+timedelta(microseconds=1))]:
            owners=np.empty(len(atoms),dtype=np.uint64)
            for start in range(0,len(atoms),50000):
                owners[start:start+50000]=client.chunkedgraph.get_roots(atoms[start:start+50000],timestamp=when)
            if np.any(owners==0):
                (private/'rejected.json').write_text(json.dumps({'operation_id':op,'reason':'Historical ownership missing',
                    'partition':label,'missing_atoms':int(np.sum(owners==0)),'timestamp':when.isoformat()},indent=2)+'\n')
                raise ValueError('Historical ownership missing')
            np.savez(private/(label+'.npz'),atoms=atoms,segments=owners)
            partitions.append(owners)
        before,after=partitions
        ref=reference_labels(atoms,known,roots);weights=counts[foreground]
        before_ids=list(map(int,m['before_root_ids']));after_ids=list(map(int,m['after_root_ids']))
        pre=composition(before,ref,weights,before_ids);post=composition(after,ref,weights,after_ids)
        affected=np.isin(before,before_ids)
        unexpected=np.unique(after[affected & ~np.isin(after,after_ids)]).tolist()
        # Background has no partition assignment. Certified reference remains private.
        np.save(public/'atoms.npy',cube);np.save(public/'em.npy',np.asarray(em[slices]))
        np.savez(public/'partition.npz',atoms=atoms,segments=before)
        labels=np.zeros(len(ids),dtype=np.uint64);labels[foreground]=ref
        np.save(private/'reference.npy',labels[inverse].reshape(cube.shape))
        public_config={'name':'cb2-practice-'+op,'partition':'partition.npz'}
        for name in ('atoms','em'):
            public_config[name]={'path':name+'.npy','resolution_nm':res.tolist(),'origin_nm':(origin+lo*res).tolist()}
        (public/'dataset.json').write_text(json.dumps(public_config,indent=2)+'\n')
        # Exact operation metadata is provenance, not exposed as a model hint.
        report={'operation_id':op,'sample_hash':row['combined_sample_hash'],'split':args.split,'source_root_id':m['root_id'],
            'exclusion_audits':str(args.exclude_audits) if args.exclude_audits else None,'excluded_operations':args.exclude_operation,
            'kind':row['sample_type'],'timestamp':timestamp.isoformat(),'crop_bounds_nm':[(origin+lo*res).tolist(),(origin+hi*res).tolist()],
            'source_world':str(args.world),'source_candidates':str(args.candidates),'reference_spec':str(args.reference/'reference-spec.json'),
            'before_certified_composition':pre,'after_certified_composition':post,
            'unexpected_after_roots':list(map(str,unexpected)),
            'missing_before_roots':[str(r) for r in before_ids if not np.any(before==r)],
            'missing_after_roots':[str(r) for r in after_ids if not np.any(after==r)],
            'foreground_atoms':len(atoms),'certified_voxels':sum(map(int,weights[ref!=0])),
            'foreground_voxels':sum(map(int,weights)),'frozen':False,'seconds':time.time()-started}
        identities={r for values in pre.values() for r in values if r!='0'}
        report['reference_identity_overlap']=sorted(identities & blocked_reference_roots)
        report['exclusion_audits_sha256']=hashlib.sha256(args.exclude_audits.read_bytes()).hexdigest() if args.exclude_audits else None
        blocked_reference_roots.update(identities)
        (private/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(report);print(json.dumps(report),flush=True)
    (args.output/'audits.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
