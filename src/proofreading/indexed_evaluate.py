"""Evaluator-only weighted atom scoring. Never imported by the agent environment.

A reference label is certified identity, not a claim of complete extension.
Unknown material is reported separately and never treated as a correct merge.
"""
import numpy as np


def _vectors(atoms, reference, voxel_counts):
    a, r, n = map(np.asarray, (atoms, reference, voxel_counts))
    if a.ndim != 1 or a.shape != r.shape or a.shape != n.shape:
        raise ValueError('Atom, reference, and voxel count vectors must align')
    if any(x.dtype.kind not in 'iu' for x in (a,r,n)):
        raise ValueError('IDs and counts must be integers')
    if np.any(a<=0) or np.any(r<0) or np.any(n<=0) or len(np.unique(a))!=len(a):
        raise ValueError('Unique positive atoms/counts and nonnegative reference IDs required')
    return a,r,n


def weighted_metrics(atoms, partition, reference, voxel_counts):
    """Equivalent to dense voxel VI/pair counts, without expanding each voxel.

    Each supplied immutable atom must have a single certified reference identity.
    Caller supplies a frozen scoring scope; weights count only voxels in it.
    """
    a,r,n=_vectors(atoms,reference,voxel_counts)
    known=r!=0
    if not known.any():raise ValueError('No certified foreground atoms in scoring scope')
    labels=np.asarray([partition[int(x)] for x in a[known]],dtype=np.uint64)
    if np.any(labels==0):raise ValueError('Partition must preserve all foreground atoms')
    cells={}
    for p,t,w in zip(labels,r[known],n[known]):
        key=(int(p),int(t));cells[key]=cells.get(key,0)+int(w)
    result = contingency_metrics(cells)
    result.update(scope_voxels=sum(map(int,n)),
                  unknown_reference_voxels_in_scope=sum(map(int,n[~known])),
                  scope='Frozen supplied atom weights; certified identities only. Unknown material is unscored.')
    return result


def contingency_metrics(cells):
    """Exact pair counts and VI from {(predicted, reference): voxel_count}."""
    cells={key:int(w) for key,w in cells.items() if w}
    if not cells or any(w<0 for w in cells.values()):raise ValueError('Invalid contingency counts')
    ps,ts={},{}
    for (p,t),w in cells.items():
        ps[p]=ps.get(p,0)+w;ts[t]=ts.get(t,0)+w
    total=sum(cells.values())
    merge=sum(w/total*np.log2(ps[p]/w) for (p,t),w in cells.items())
    split=sum(w/total*np.log2(ts[t]/w) for (p,t),w in cells.items())
    choose2=lambda values:sum(int(w)*(int(w)-1)//2 for w in values)
    tp=choose2(cells.values())
    return {'vi_merge_bits':float(merge),'vi_split_bits':float(split),'vi_total_bits':float(merge+split),
            'false_merge_voxel_pairs':choose2(ps.values())-tp,
            'false_split_voxel_pairs':choose2(ts.values())-tp,'scored_voxels':total}



def extension_evidence(atoms, reference, voxel_counts, initial_atoms, final_atoms, target_reference,
                       reachable_target_atoms=None):
    """Audit growth/loss of a target; do not score unknown additions as correct.

    All initial/final atoms must appear in the supplied table, including unknowns.
    Recall is omitted unless caller supplies a frozen, crop-reachable target set.
    """
    a,r,n=_vectors(atoms,reference,voxel_counts)
    if int(target_reference)<=0:raise ValueError('A certified target reference is required')
    lookup={int(x):(int(t),int(w)) for x,t,w in zip(a,r,n)}
    initial,final=set(map(int,initial_atoms)),set(map(int,final_atoms))
    if not initial|final <= lookup.keys():raise ValueError('Reference table must include unknown selected atoms too')
    def counts(subset):
        own=wrong=unknown=0
        wrong_roots={}
        for atom in subset:
            target,weight=lookup[atom]
            if target==0:unknown+=weight
            elif target==target_reference:own+=weight
            else:
                wrong+=weight
                wrong_roots[str(target)]=wrong_roots.get(str(target),0)+weight
        return {'target_voxels':own,'other_certified_neuron_voxels':wrong,
                'unknown_voxels':unknown,'other_certified_neurons':wrong_roots}
    out={'initial':counts(initial),'final':counts(final),'added':counts(final-initial),
         'removed':counts(initial-final),'recall':None,
         'interpretation':'Other certified neurons are confirmed contamination; unknown additions remain unresolved.'}
    if reachable_target_atoms is not None:
        reachable=set(map(int,reachable_target_atoms))
        if not reachable or not reachable<=lookup.keys() or any(lookup[x][0]!=target_reference for x in reachable):
            raise ValueError('Reachable target must be nonempty, represented, and certified to the target')
        denominator=sum(lookup[x][1] for x in reachable)
        out['recall']={'reachable_target_voxels':denominator,
                       'initial':sum(lookup[x][1] for x in reachable&initial)/denominator,
                       'final':sum(lookup[x][1] for x in reachable&final)/denominator,
                       'scope':'Only caller-supplied frozen crop-reachable target; excludes outside-crop reconnections.'}
    return out


def load_partition(dataset, path):
    """Restore and validate a sparse saved partition without copying all owners."""
    with np.load(path,allow_pickle=False) as z:
        if str(z['format'])!='sparse-overlay-v1' or str(z['dataset_sha256'])!=dataset.fingerprint:
            raise ValueError('Sparse partition format or dataset fingerprint mismatch')
        atoms,segments=z['atoms'],z['segments']
        _vectors(atoms,segments,np.ones(len(atoms),dtype=np.uint8))
        if np.any(segments==0):raise ValueError('Partition cannot remove foreground atoms')
        dataset.to_dense(atoms)  # Reject invented atoms before applying changes.
        partition=dataset.initial.copy()
        for atom,segment in zip(atoms,segments):partition[int(atom)]=int(segment)
    return partition


class CertifiedScorer:
    """Cached whole-crop identity contingency with sparse edit updates.

    Whole-root split VI is a fragmentation diagnostic, not reachable extension
    recall. Practice-task scores require an independently frozen task scope.
    """
    def __init__(self,dataset,directory):
        import json
        from pathlib import Path
        from .evaluate import file_hash
        self.dataset=dataset;self.directory=Path(directory)
        self.spec_file=self.directory/'reference-spec.json'
        self.spec=json.loads(self.spec_file.read_text())
        if self.spec['geometry_sha256']!=dataset.geometry_fingerprint:
            raise ValueError('Reference geometry identity mismatch')
        self.spec_hash=file_hash(self.spec_file)
        for name in ('atoms','reference','voxel_counts'):
            entry=self.spec['arrays'][name];path=self.directory/entry['file']
            if path.resolve().parent!=self.directory.resolve() or file_hash(path)!=entry['sha256']:
                raise ValueError('Reference array identity mismatch')
            setattr(self,name,np.load(path,mmap_mode='r',allow_pickle=False))
        if self.atoms.dtype != np.dtype('uint64'):
            raise ValueError('Certified atom index must use uint64 for exact ID lookup')
        if not (self.atoms.shape==self.reference.shape==self.voxel_counts.shape) or self.atoms.ndim!=1:
            raise ValueError('Reference arrays must align')
        for start in range(0,len(self.atoms),500000):
            a=self.atoms[max(0,start-1):start+500000]
            if np.any(a[1:]<=a[:-1]) or np.any(a==0):raise ValueError('Reference atoms must be sorted unique foreground')
            if np.any(self.reference[start:start+500000]==0) or np.any(self.voxel_counts[start:start+500000]==0):
                raise ValueError('Certified reference index contains unknown labels or empty atoms')
        self.baseline=None

    def build_baseline(self,output=None):
        import json
        from pathlib import Path
        from .evaluate import file_hash
        cells={}
        for start in range(0,len(self.atoms),500000):
            a=self.atoms[start:start+500000]
            p=self.dataset.owners[self.dataset.to_dense(a)]
            r=self.reference[start:start+500000];w=self.voxel_counts[start:start+500000]
            order=np.lexsort((r,p));p,r,w=p[order],r[order],w[order]
            boundaries=np.r_[0,np.flatnonzero((p[1:]!=p[:-1])|(r[1:]!=r[:-1]))+1]
            # Physical volume size bounds every sum below uint64 overflow.
            if np.prod(self.dataset.atoms.data.shape,dtype=object)>=2**64:raise ValueError('Volume too large for uint64 counts')
            sums=np.add.reduceat(w,boundaries)
            for pp,rr,nn in zip(p[boundaries],r[boundaries],sums):
                key=(int(pp),int(rr));cells[key]=cells.get(key,0)+int(nn)
        self.baseline=cells
        if output:
            output=Path(output)
            rows=np.asarray([(p,r,n) for (p,r),n in sorted(cells.items())],dtype=np.uint64)
            with output.open('xb') as f:np.save(f,rows,allow_pickle=False)
            output.with_suffix('.json').write_text(json.dumps({'reference_spec_sha256':self.spec_hash,
                'baseline_sha256':file_hash(output),'geometry_sha256':self.dataset.geometry_fingerprint,
                'metrics':contingency_metrics(cells)},indent=2)+'\n')
        return contingency_metrics(cells)

    def load_baseline(self,path):
        import json
        from pathlib import Path
        from .evaluate import file_hash
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
        if (meta['reference_spec_sha256']!=self.spec_hash or
            meta['geometry_sha256']!=self.dataset.geometry_fingerprint or file_hash(path)!=meta['baseline_sha256']):
            raise ValueError('Baseline identity mismatch')
        rows=np.load(path,allow_pickle=False)
        if rows.ndim!=2 or rows.shape[1]!=3 or np.any(rows==0):raise ValueError('Invalid baseline rows')
        self.baseline={(int(p),int(r)):int(w) for p,r,w in rows}
        if len(self.baseline)!=len(rows):raise ValueError('Duplicate baseline cells')

    def lookup(self,atoms):
        atoms=np.asarray(atoms,dtype=np.uint64)
        pos=np.searchsorted(self.atoms,atoms)
        valid=pos<len(self.atoms)
        valid[valid]&=self.atoms[pos[valid]]==atoms[valid]
        refs=np.zeros(len(atoms),dtype=np.uint64)
        refs[valid]=self.reference[pos[valid]]
        counts=np.asarray(self.dataset.atom_cnt[self.dataset.to_dense(atoms)])
        if not np.all(counts==np.floor(counts)):raise ValueError('Noninteger voxel weights')
        return refs,counts.astype(np.uint64),pos,valid

    def score(self,partition):
        if self.baseline is None:raise ValueError('Build or load the certified baseline first')
        cells=dict(self.baseline)
        atoms=np.asarray(sorted(partition.changes),dtype=np.uint64)
        reference,counts,pos,known=self.lookup(atoms)
        initial=self.dataset.owners[self.dataset.to_dense(atoms)]
        for a,old,r,n in zip(atoms[known],initial[known],reference[known],self.voxel_counts[pos[known]]):
            new=partition.changes[int(a)]
            if new==old:continue
            key=(int(old),int(r));cells[key]-=int(n)
            if cells[key]<0:raise ValueError('Edit delta exceeds certified baseline')
            key=(int(new),int(r));cells[key]=cells.get(key,0)+int(n)
        before,after=contingency_metrics(self.baseline),contingency_metrics(cells)
        return {'initial':before,'final':after,
                'vi_improvement_bits':before['vi_total_bits']-after['vi_total_bits'],
                'false_merge_pairs_removed':before['false_merge_voxel_pairs']-after['false_merge_voxel_pairs'],
                'false_split_pairs_removed':before['false_split_voxel_pairs']-after['false_split_voxel_pairs'],
                'changed_atoms':len(atoms),'changed_certified_atoms':int(known.sum()),
                'changed_unknown_voxels':sum(map(int,counts[~known])),
                'reference_spec_sha256':self.spec_hash,'retrospective':True,
                'scope_note':'Whole-crop certified identity diagnostic. Split VI includes disconnected reference pieces; it is not crop-reachable recall.'}


def new_merge_evidence(initial, final, reference, counts):
    """Attribute newly joined cross-neuron pairs, excluding pre-existing mergers.

    Inputs must cover every atom of each reported final segment. Unknown reference
    material is reported separately and cannot establish correctness.
    """
    arrays=[np.asarray(x) for x in (initial,final,reference,counts)]
    if any(x.ndim!=1 or x.shape!=arrays[0].shape or x.dtype.kind not in 'iu' for x in arrays):
        raise ValueError('Expected aligned integer vectors')
    if any(np.any(x<=0) for x in (arrays[0],arrays[1],arrays[3])) or np.any(arrays[2]<0):
        raise ValueError('Owners and counts must be positive; reference may be unknown zero')
    groups={}
    for old,new,ref,n in zip(*arrays):
        cells=groups.setdefault(int(new),{})
        key=(int(old),int(ref));cells[key]=cells.get(key,0)+int(n)
    reports=[]
    for segment,cells in sorted(groups.items()):
        rows=sorted(cells.items());pairs=[]
        for i,((left,lr),ln) in enumerate(rows):
            if not lr:continue
            for (right,rr),rn in rows[i+1:]:
                if left!=right and rr and lr!=rr:
                    pairs.append({'initial_segments':[str(left),str(right)],
                                  'certified_neurons':[str(lr),str(rr)],
                                  'voxels':[ln,rn],'new_false_merge_voxel_pairs':ln*rn})
        reports.append({'final_segment':str(segment),
                        'composition':[{'initial_segment':str(old),'certified_neuron':str(ref) if ref else None,'voxels':n} for (old,ref),n in rows],
                        'unknown_voxels':sum(n for (old,ref),n in rows if not ref),
                        'new_false_merge_voxel_pairs':sum(p['new_false_merge_voxel_pairs'] for p in pairs),
                        'new_cross_neuron_joins':pairs})
    return reports
