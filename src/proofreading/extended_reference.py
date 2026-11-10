"""Retrospective edit audit over complete affected supervoxel memberships.

Counts changed pairwise relationships, not voxel volume. A later segmentation
is evidence, not certified biological truth. Original benchmark scores stay intact.
"""
from collections import Counter


def _pairs(counts):
    return sum(n * (n - 1) // 2 for n in counts.values())


def relationship_changes(initial, final, reference):
    """Separate newly joined/separated pairs by later-reference agreement."""
    if set(initial) != set(final) or set(initial) != set(reference):
        raise ValueError('Aligned complete atom maps required')
    if any(int(v) <= 0 for d in (initial, final) for v in d.values()):
        raise ValueError('Positive partition labels required')
    if any(int(v) < 0 for v in reference.values()):
        raise ValueError('Negative reference label')
    return relationship_changes_from_cells((initial[a],final[a],reference[a],1) for a in initial)


def relationship_changes_from_cells(cells):
    """Exact sufficient statistics: (initial, final, reference, atom count)."""
    cells=list(cells)
    if any(x<=0 or y<=0 or r<0 or n<=0 for x,y,r,n in cells):
        raise ValueError('Invalid partition count cell')
    def count(known_only):
        i=Counter();f=Counter();both=Counter();ir=Counter();fr=Counter();br=Counter()
        for x,y,r,n in cells:
            if known_only and not r:continue
            i[x]+=n;f[y]+=n;both[x,y]+=n;ir[x,r]+=n;fr[y,r]+=n;br[x,y,r]+=n
        joined=_pairs(f)-_pairs(both);split=_pairs(i)-_pairs(both)
        joined_same=_pairs(fr)-_pairs(br);split_same=_pairs(ir)-_pairs(br)
        return joined,split,joined_same,split_same
    total=count(False);known=count(True)
    joined,split,js,ss=known
    result={'supported_joined_pairs':js,'contradicted_joined_pairs':joined-js,
            'supported_separated_pairs':split-ss,'contradicted_separated_pairs':ss,
            'unresolved_changed_pairs':total[0]+total[1]-joined-split,
            'changed_pairs':total[0]+total[1],
            'unknown_reference_atoms':sum(n for x,y,r,n in cells if not r)}
    good=js+split-ss;bad=joined-js+ss
    result['net_reference_pair_error_reduction']=good-bad
    result['supported_fraction_of_known_changed_pairs']=good/(good+bad) if good+bad else None
    result['assessment']=('no_net_relationship_change' if not result['changed_pairs'] else
        'unresolved' if result['unresolved_changed_pairs'] else
        'mixed' if good and bad else 'contradicted' if bad else 'supported')
    return result


def human_pair_changes(initial, final, reference, human_roots):
    """Joined relationships between the two human targets versus other edits."""
    return human_pair_changes_from_cells(((initial[a],final[a],reference[a],1) for a in initial),human_roots)


def human_pair_changes_from_cells(cells, human_roots):
    roots=set(map(int,human_roots))
    if len(roots)!=2:raise ValueError('Exactly two original human targets required')
    a,b=sorted(roots);fa=Counter();fb=Counter();ra=Counter();rb=Counter()
    for owner,final,reference,n in cells:
        if owner not in roots:continue
        (fa if owner==a else fb)[final]+=n
        if reference: (ra if owner==a else rb)[final,reference]+=n
    joined=sum(n*fb[k] for k,n in fa.items())
    supported=sum(n*rb[k] for k,n in ra.items())
    return {'human_target_joined_pairs':joined,'human_target_supported_joined_pairs':supported}


def project_reference_cells(root, atoms, reference, rules):
    """Project complete membership through rules and compress without sampling."""
    import numpy as np
    atoms=np.asarray(atoms,dtype=np.uint64);reference=np.asarray(reference,dtype=np.uint64)
    if atoms.ndim!=1 or reference.shape!=atoms.shape:raise ValueError('Aligned membership required')
    if all(rule['assignments'] is None for rule in rules):
        final_root=int(root)
        for rule in rules:
            if final_root in set(map(int,rule['parents'])):final_root=int(rule['children'][0])
        ids,counts=np.unique(reference,return_counts=True)
        return [(int(root),final_root,int(r),int(n)) for r,n in zip(ids,counts)]
    final=np.full(len(atoms),root,dtype=np.uint64)
    for rule in rules:
        mask=np.isin(final,np.asarray(rule['parents'],dtype=np.uint64))
        if not mask.any():continue
        if rule['assignments'] is None:final[mask]=int(rule['children'][0])
        else:
            assignments={int(a):int(c) for a,c in rule['assignments'].items()}
            keys=np.array(sorted(assignments),dtype=np.uint64)
            ix=np.searchsorted(keys,atoms[mask])
            if np.any(ix>=len(keys)) or not np.array_equal(keys[ix],atoms[mask]):
                raise ValueError('Split omitted affected reference atom')
            values=np.array([assignments[int(k)] for k in keys],dtype=np.uint64)
            final[mask]=values[ix]
    pairs=np.empty(len(atoms),dtype=[('final','<u8'),('reference','<u8')])
    pairs['final']=final;pairs['reference']=reference
    keys,counts=np.unique(pairs,return_counts=True)
    return [(int(root),int(k['final']),int(k['reference']),int(n)) for k,n in zip(keys,counts)]


def voxel_weighted_relationship_changes(initial, final, reference, voxel_counts):
    """Changed voxel-pair relationships within the caller's explicit scope.

    Counts must be measured voxel counts, not estimated full-supervoxel sizes.
    Atoms of sizes a and b contribute a*b voxel pairs. Missing atoms are errors;
    unknown reference labels remain unresolved. Publish scope coverage alongside
    results: this function does not infer weights outside the supplied volume.
    """
    from numbers import Integral
    if not (set(initial) == set(final) == set(reference) == set(voxel_counts)):
        raise ValueError('Aligned voxel counts and partition maps required')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0
           for n in voxel_counts.values()):
        raise ValueError('Positive integer voxel counts required')
    result = relationship_changes_from_cells(
        (int(initial[a]), int(final[a]), int(reference[a]), int(n))
        for a, n in voxel_counts.items())
    result = {k.replace('_pairs', '_voxel_pairs').replace(
        'unknown_reference_atoms', 'unknown_reference_voxels').replace(
        'net_reference_pair_error_reduction', 'net_reference_voxel_pair_error_reduction'): v
        for k, v in result.items()}
    result['scoped_voxels'] = sum(map(int, voxel_counts.values()))
    result['scoped_supervoxels'] = len(voxel_counts)
    result['weighting'] = 'voxel pairs; product of measured in-scope voxel counts'
    return result
