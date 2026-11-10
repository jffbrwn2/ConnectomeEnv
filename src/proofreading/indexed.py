"""Memory-mapped historical worlds with sparse, per-session edit overlays.

Only public EM, initial ownership and geometric indices are read. No reference
labels are loaded. Indexed adjacency uses unit capacities, explicitly reported
in every state; it is not a learned membrane graph.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .volume import Grid


class Overlay:
    def __init__(self, dataset, changes=None):
        self.dataset = dataset
        self.changes = {} if changes is None else changes

    def __getitem__(self, atom):
        atom = int(atom)
        if atom in self.changes:
            return self.changes[atom]
        return int(self.dataset.owners[self.dataset.to_dense([atom])[0]])

    def __setitem__(self, atom, segment):
        self.changes[int(atom)] = int(segment)

    def copy(self):
        return Overlay(self.dataset, self.changes.copy())

    def atoms_for(self, sid):
        d = self.dataset
        pos = int(np.searchsorted(d.roots, sid))
        aa = {sid} if sid in d.unknown_ownership else set()
        if sid > 0 and pos < len(d.roots) and int(d.roots[pos]) == sid:
            aa.update(map(int, d.root_atoms[d.root_ip[pos]:d.root_ip[pos+1]]))
        aa.difference_update(self.changes)
        aa.update(a for a, s in self.changes.items() if s == sid)
        return aa


class LocalGraph:
    """Only materialize the requested induced graph, never the whole world."""
    def __init__(self, dataset):
        self.dataset = dataset

    def subgraph(self, atoms):
        import networkx as nx
        d = self.dataset
        aa = np.asarray(sorted(atoms), dtype=np.int64)
        dd = d.to_dense(aa)
        edges = d.incident(dd)
        edges = edges[np.isin(edges[:, 0], dd) & np.isin(edges[:, 1], dd)]
        g = nx.Graph()
        g.add_nodes_from(map(int, aa))
        g.add_edges_from((int(a), int(b), {'capacity': 1.0})
                         for a,b in d.native_by_dense[edges])
        return g


class IndexedDataset:
    def __init__(self, manifest):
        path = Path(manifest).resolve()
        c = json.loads(path.read_text())
        if c['backend'] != 'indexed-world-v1':
            raise ValueError('Unsupported indexed world version')
        identity = dict(c)
        fingerprint = identity.pop('fingerprint')
        if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest() != fingerprint:
            raise ValueError('Indexed manifest fingerprint mismatch')
        import os
        # Relocate read-only world storage without altering the frozen manifest identity.
        # rsync -a / equivalent must preserve timestamps; all existing source checks still run.
        from .storage import data_path
        self.world = data_path(os.environ.get('CONNECTOME_WORLD_ROOT', c['world']), path.parent)
        for name, expected in c['identity'].items():
            stat = (self.world/name).stat()
            if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != expected:
                raise ValueError(f'Indexed source changed: {name}; prepare again')
        def load(base, name):
            return np.load(base/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        m = json.loads((self.world/'manifest.json').read_text())
        resolution = np.asarray(m['resolution_xyz_nm'],float)
        origin = np.asarray(m['bounds_min_xyz'])*resolution
        self.atoms = Grid(load(self.world,'seg'),resolution,origin)
        self.em = Grid(load(self.world,'em'),resolution,origin)
        if self.em.data.shape != self.atoms.data.shape:
            raise ValueError('Indexed EM must explicitly match the segmentation grid')
        for name in ('native','dense','dense_cube','atom_cnt','atom_cent_vox','label_lo',
                     'label_hi','atom_edges_dense','atom_inc_eids','atom_inc_indptr'):
            setattr(self,name,load(self.world/'geom_index',name))
        for attr,name in [('roots','root_ids'),('root_ip','indptr'),('root_atoms','atoms')]:
            setattr(self,attr,load(self.world/'root_atoms_csr',name))
        for name in ('owners','native_by_dense','root_counts','root_order'):
            setattr(self,name,load(path.parent,name))
        self.unknown_ownership = set(c.get('unknown_ownership_atoms', []))
        self.name, self.fingerprint = c['name'], fingerprint
        geometry_identity = dict(identity)
        geometry_identity.pop('em_pyramid', None)
        self.geometry_fingerprint = hashlib.sha256(json.dumps(geometry_identity,sort_keys=True).encode()).hexdigest()
        if 'em_pyramid' in c:
            from .multiscale_em import MultiscaleEM
            self.em = MultiscaleEM(self.em, c['em_pyramid'])
        self.identity_kind = c['identity_kind']
        self.split_capacity = c['split_capacity']
        self.edges = True  # public supplied adjacency, not inferred contact area
        self.initial = Overlay(self)
        self.max_segment_id = int(self.roots[-1])
        from .sampling import load_levels
        load_levels(self, c, path.parent)
        self.manifest = path

    def to_dense(self, atoms):
        aa = np.asarray(atoms,dtype=np.int64)
        pos = np.searchsorted(self.native,aa)
        if np.any(pos>=len(self.native)) or not np.array_equal(self.native[pos],aa) or np.any(aa<=0):
            raise ValueError('Unknown foreground atom')
        return self.dense[pos]

    def graph(self):
        return LocalGraph(self)

    def incident(self, dd):
        starts, ends = self.atom_inc_indptr[dd], self.atom_inc_indptr[dd+1]
        ids = [self.atom_inc_eids[a:b] for a,b in zip(starts,ends) if b>a]
        if not ids:
            return np.empty((0,2),np.int64)
        return self.atom_edges_dense[np.unique(np.concatenate(ids))]

    def geometry(self, atoms):
        dd = self.to_dense(np.asarray(sorted(atoms),dtype=np.int64))
        lo = np.min(self.label_lo[dd-1],axis=0)
        hi = np.max(self.label_hi[dd-1],axis=0)  # exclusive voxel bounds
        return dd,lo,hi

    def inspect(self, partition, sid, atoms, offset, limit):
        dd,lo,hi = self.geometry(atoms)
        neighbors = {partition[int(a)] for a in np.unique(self.native_by_dense[self.incident(dd)])} - {sid}
        lower = self.atoms.origin + lo*self.atoms.resolution
        upper = self.atoms.origin + hi*self.atoms.resolution
        return {'segment':str(sid),'n_atoms':len(atoms),'bounds_nm':[lower.tolist(),upper.tolist()],
                'center_nm':((lower+upper)/2).tolist(),'voxels':int(self.atom_cnt[dd].sum()),
                'atoms':list(map(str,sorted(atoms)[offset:offset+limit])),
                'next_offset':offset+limit if offset+limit<len(atoms) else None,
                'neighbors':sorted(map(str,neighbors))}

    def list_segments(self, partition, offset, limit):
        # Baseline counts are 12 MB; edits touch only their atom-level deltas.
        counts = self.root_counts.copy()
        new = {a: int(self.atom_cnt[self.to_dense([a])[0]]) for a in self.unknown_ownership}
        if self.roots[0] == 0:
            counts[0] = 0
        for a,s in partition.changes.items():
            dd = self.to_dense([a])[0]
            old, n = int(self.owners[dd]), int(self.atom_cnt[dd])
            if old in self.unknown_ownership:
                new[old] -= n
            else:
                counts[np.searchsorted(self.roots,old)] -= n
            if s <= self.max_segment_id:
                i = np.searchsorted(self.roots,s)
                if i<len(self.roots) and int(self.roots[i])==s:
                    counts[i] += n
                    continue
            new[s] = new.get(s,0)+n
        ids = np.concatenate([self.roots,np.asarray(list(new),dtype=np.int64)])
        values = np.concatenate([counts,np.asarray(list(new.values()),dtype=np.int64)])
        valid = values>0
        ids,values = ids[valid],values[valid]
        order = np.lexsort((ids,-values))
        rows = order[offset:offset+limit]
        return {'total':len(order),'segments':[{'id':str(int(ids[i])),'voxels':int(values[i])} for i in rows],
                'next_offset':offset+limit if offset+limit<len(order) else None}

    def save_partition(self, partition, path):
        aa = np.asarray(sorted(partition.changes),dtype=np.uint64)
        np.savez_compressed(path, atoms=aa,
                            segments=np.asarray([partition.changes[int(a)] for a in aa],dtype=np.uint64),
                            format=np.asarray('sparse-overlay-v1'), dataset_sha256=np.asarray(self.fingerprint))

    def mesh_parts(self, atoms, stride):
        """Extract all occupied 64-cell tiles, including halos, without cropping scope.

        Tiles partition marching-cubes cells (not voxels), so internal tile faces
        are never padded and cannot introduce artificial caps.
        """
        from skimage.measure import marching_cubes
        dd, _, _ = self.geometry(atoms)
        # One dense membership lookup per extraction instead of sorting the atom
        # list and every voxel tile repeatedly in np.isin. Exact, including uint64 IDs.
        membership = np.zeros(len(self.native_by_dense), dtype=bool)
        membership[dd] = True
        los = self.label_lo[dd-1]//stride
        his = (self.label_hi[dd-1]-1)//stride
        tiles = set()
        for lo,hi in zip(los,his):
            for x in range(int(lo[0]//64),int((hi[0]+1)//64)+1):
                for y in range(int(lo[1]//64),int((hi[1]+1)//64)+1):
                    for z in range(int(lo[2]//64),int((hi[2]+1)//64)+1):
                        tiles.add((x,y,z))
        shape = (np.asarray(self.atoms.data.shape)+stride-1)//stride
        for tile in sorted(tiles):
            start = np.asarray(tile)*64-1
            end = np.minimum(start+64,shape)
            axes = [np.arange(a,b+1,dtype=np.int64) for a,b in zip(start,end)]
            valid = [(v>=0)&(v<n) for v,n in zip(axes,shape)]
            source = [np.clip(v,0,n-1)*stride for v,n in zip(axes,shape)]
            cube = self.dense_cube[np.ix_(*source)]
            mask = membership[cube]
            mask &= valid[0][:,None,None]&valid[1][None,:,None]&valid[2][None,None,:]
            if not mask.any() or mask.all():
                continue
            vertices,faces,_,_ = marching_cubes(mask.astype(np.float32),0.5)
            vertices = self.atoms.origin+(vertices+start)*self.atoms.resolution*stride+self.atoms.resolution/2
            yield vertices,faces
