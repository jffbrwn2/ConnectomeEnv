"""Evaluator-only sufficient statistics. No images, mesh, or source clients."""
from collections import Counter
from pathlib import Path
import numpy as np
from .evaluate import contingency_metrics,attachment_counts


def pack(atoms, initial, reference, scope, output):
    if reference.shape!=atoms.shape or scope.shape!=atoms.shape or scope.dtype!=bool:
        raise ValueError('Reference and boolean scope must align with atom grid')
    if reference.dtype.kind not in 'iu':raise ValueError('Reference must contain integer labels')
    cells=Counter();totals=Counter();target=Counter();scope_count=unknown=0
    chunks=np.nditer([atoms,reference,scope],flags=['external_loop','buffered','zerosize_ok'],
                     op_flags=[['readonly']]*3,order='C',buffersize=1<<20)
    for aa,rr,ss in chunks:
        if rr.dtype.kind=='i' and np.any(rr<0):raise ValueError('Negative reference labels')
        scope_count+=int(np.count_nonzero(ss));unknown+=int(np.count_nonzero(ss & (rr==0)))
        for mask,counter in [(aa!=0,totals),(ss & (aa!=0),target)]:
            ids,counts=np.unique(aa[mask],return_counts=True)
            counter.update({int(a):int(n) for a,n in zip(ids,counts)})
        valid=ss & (aa!=0) & (rr!=0)
        if valid.any():
            ai,av=np.unique(aa[valid],return_inverse=True);ri,rv=np.unique(rr[valid],return_inverse=True)
            codes,counts=np.unique(av*len(ri)+rv,return_counts=True)
            cells.update({(int(ai[c//len(ri)]),int(ri[c%len(ri)])):int(n) for c,n in zip(codes,counts)})
    if set(totals)!=set(initial):raise ValueError('Initial partition differs from scoring atoms')
    if not cells:raise ValueError('No scored foreground voxels')
    ids=sorted(initial);keys=sorted(cells)
    with Path(output).open('xb') as f:
        np.savez_compressed(f,atoms=np.asarray(ids,np.uint64),initial=np.asarray([initial[a] for a in ids],np.uint64),
            total=np.asarray([totals[a] for a in ids],np.uint64),target=np.asarray([target[a] for a in ids],np.uint64),
            cell_atoms=np.asarray([a for a,r in keys],np.uint64),reference=np.asarray([r for a,r in keys],np.uint64),
            count=np.asarray([cells[k] for k in keys],np.uint64),scope_voxels=np.asarray(scope_count,np.uint64),
            unknown_voxels=np.asarray(unknown,np.uint64))


def read(path):
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    required={'atoms','initial','total','target','cell_atoms','reference','count','scope_voxels','unknown_voxels'}
    if set(data)!=required or any(a.dtype!=np.uint64 for a in data.values()):raise ValueError('Invalid scoring table')
    atoms=data['atoms'];n=len(atoms)
    if atoms.ndim!=1 or not n or np.any(atoms==0) or np.any(atoms[1:]<=atoms[:-1]):raise ValueError('Invalid scoring atoms')
    if any(data[k].shape!=(n,) for k in ['initial','total','target']):raise ValueError('Misaligned atom counts')
    if np.any(data['initial']==0) or np.any(data['total']==0) or np.any(data['target']>data['total']):raise ValueError('Invalid atom counts')
    m=len(data['count'])
    if any(data[k].shape!=(m,) for k in ['cell_atoms','reference','count']) or not m:raise ValueError('Invalid reference cells')
    if np.any(data['count']==0) or np.any(data['reference']==0) or not np.isin(data['cell_atoms'],atoms).all():raise ValueError('Invalid reference cells')
    if any(data[k].shape!=() for k in ['scope_voxels','unknown_voxels']):raise ValueError('Invalid scope counts')
    return data


def score(data, initial, final):
    def metric(partition):
        cells=Counter()
        for a,r,n in zip(data['cell_atoms'],data['reference'],data['count']):cells[partition[int(a)],int(r)]+=int(n)
        return contingency_metrics(cells,int(data['scope_voxels']),int(data['unknown_voxels']))
    target={int(a):int(n) for a,n in zip(data['atoms'],data['target']) if n}
    outside={int(a):int(n)-int(t) for a,n,t in zip(data['atoms'],data['total'],data['target']) if n>t}
    before,after=metric(initial),metric(final)
    return {'initial':before,'final':after,'vi_improvement_bits':before['vi_total_bits']-after['vi_total_bits'],
            'scope_attachment':attachment_counts(initial,final,target,outside)}
