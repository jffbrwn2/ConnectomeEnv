"""Build compact historical ownership for a pre-indexed world; never read references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('world', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
m = json.loads((a.world / 'manifest.json').read_text())
def read(path):
    return np.load(a.world / path, mmap_mode='r', allow_pickle=False)
native, dense = read('geom_index/native.npy'), read('geom_index/dense.npy')
roots, ip, atoms = (read('root_atoms_csr/' + f + '.npy') for f in ('root_ids','indptr','atoms'))
if len(roots) != m['extra']['n_initial_roots']:
    raise ValueError('Legacy CSR does not exclusively describe historical roots')
if len(atoms) != len(native)-1 or native[0] != 0:
    raise ValueError('Unexpected atom coverage')
counts = read('geom_index/atom_cnt.npy')
owners = np.lib.format.open_memmap(a.output/'owners.npy', mode='w+', dtype=np.int64, shape=counts.shape)
inverse = np.lib.format.open_memmap(a.output/'native_by_dense.npy', mode='w+', dtype=np.int64, shape=counts.shape)
root_counts = np.zeros(len(roots), np.int64)
for start in range(0, len(native), 1_000_000):
    stop = min(start+1_000_000,len(native))
    inverse[dense[start:stop]] = native[start:stop]
for start in range(0,len(atoms),1_000_000):
    stop=min(start+1_000_000,len(atoms))
    aa=np.asarray(atoms[start:stop],dtype=np.int64)
    pos=np.searchsorted(native,aa)
    if np.any(pos>=len(native)) or not np.array_equal(native[pos],aa):
        raise ValueError('CSR contains unknown atoms')
    dd=dense[pos]
    if np.any(owners[dd]) or len(np.unique(dd)) != len(dd):
        raise ValueError('Duplicate ownership')
    rows=np.searchsorted(ip,np.arange(start,stop,dtype=np.int64),side='right')-1
    assigned=roots[rows].copy()
    assigned[assigned==0]=aa[assigned==0]
    owners[dd]=assigned
    np.add.at(root_counts,rows,counts[dd].astype(np.int64))
    if start%10_000_000==0: print('Assigned',stop,'/',len(atoms),flush=True)
for start in range(1,len(owners),1_000_000):
    if np.any(owners[start:start+1_000_000]==0): raise ValueError('Missing ownership')
owners.flush(); inverse.flush()
np.save(a.output/'root_counts.npy',root_counts)
np.save(a.output/'root_order.npy',np.lexsort((roots,-root_counts)))
paths=['manifest.json','seg.npy','em.npy']+['geom_index/'+f+'.npy' for f in ('native','dense','dense_cube','atom_cnt','atom_cent_vox','label_lo','label_hi','atom_edges_dense','atom_inc_eids','atom_inc_indptr')]+['root_atoms_csr/'+f+'.npy' for f in ('root_ids','indptr','atoms')]
# Fast identity binds immutable local files and catches replacement/modification;
# it is explicitly not a content hash of the 100+ GB source volume.
identity={f:{'size':(a.world/f).stat().st_size,'mtime_ns':(a.world/f).stat().st_mtime_ns} for f in paths}
config={'backend':'indexed-world-v1','world':str(a.world.resolve()),'name':m['name'],
        'identity':identity,'identity_kind':'source-file-size-and-mtime',
        'split_capacity':'unit geometric adjacency (not EM confidence or contact area)',
        'prepared_atoms':len(atoms),'prepared_roots':len(roots)}
config['unknown_ownership_atoms']=list(map(int,atoms[ip[0]:ip[1]])) if int(roots[0])==0 else []
config['unknown_ownership_policy']='each root-zero atom starts as an independent segment'
config['fingerprint']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
(a.output/'dataset.json').write_text(json.dumps(config,indent=2)+'\n')
print('READY',a.output/'dataset.json',flush=True)
