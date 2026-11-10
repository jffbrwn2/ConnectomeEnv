import numpy as np
import pytest
from proofreading.evaluate import metrics,scope_attachment


def oracle(atoms,partition,reference,scope):
    valid=scope&(atoms!=0)&(reference!=0)
    predicted=np.asarray([partition[int(a)] for a in atoms[valid]],np.uint64)
    pairs,counts=np.unique(np.stack([predicted,reference[valid]],axis=1),axis=0,return_counts=True)
    _,pi=np.unique(pairs[:,0],return_inverse=True);_,ri=np.unique(pairs[:,1],return_inverse=True)
    pc=np.bincount(pi,weights=counts);rc=np.bincount(ri,weights=counts)
    n=counts/counts.sum()
    choose=lambda a:sum(int(x)*(int(x)-1)//2 for x in a)
    return {'vi_merge_bits':float(np.sum(n*np.log2(pc[pi]/counts))),
            'vi_split_bits':float(np.sum(n*np.log2(rc[ri]/counts))),
            'false_merge_voxel_pairs':choose(pc)-choose(counts),
            'false_split_voxel_pairs':choose(rc)-choose(counts)}


@pytest.mark.parametrize('transpose',[False,True])
def test_chunked_score_matches_voxel_oracle_with_mixed_atom_truth(transpose):
    rng=np.random.default_rng(7)
    atoms=rng.integers(0,100,(129,130,65),dtype=np.uint64)
    reference=rng.integers(0,7,atoms.shape,dtype=np.uint64)
    scope=rng.random(atoms.shape)>.3
    if transpose:atoms,reference,scope=[a.transpose(2,0,1) for a in (atoms,reference,scope)]
    partition={a:2**63+a%11 for a in range(1,100)}
    expected=oracle(atoms,partition,reference,scope);actual=metrics(atoms,partition,reference,scope)
    for k,v in expected.items():assert actual[k]==pytest.approx(v,rel=1e-14)
    assert actual['scored_voxels']==int((scope&(atoms!=0)&(reference!=0)).sum())
    assert actual['unknown_reference_voxels_in_scope']==int((scope&(reference==0)).sum())


def test_attachment_matches_direct_mask():
    rng=np.random.default_rng(9)
    atoms=rng.integers(0,8,(17,19,23));scope=np.zeros_like(atoms,dtype=bool);scope[:2,:2,:2]=True
    initial={a:a for a in range(1,8)};final={a:a%3+10 for a in range(1,8)}
    target=np.unique(atoms[scope&(atoms!=0)])
    def attached(part):
        owners={part[int(a)] for a in target}
        return np.isin(atoms,[a for a,s in part.items() if s in owners])&~scope&(atoms!=0)
    before,after=attached(initial),attached(final)
    actual=scope_attachment(atoms,initial,final,scope)
    assert actual['initial_outside_voxels']==int(before.sum())
    assert actual['final_outside_voxels']==int(after.sum())
    assert actual['newly_attached_outside_voxels']==int((after&~before).sum())
    assert actual['detached_outside_voxels']==int((before&~after).sum())


def test_attachment_uses_immutable_counts_without_changing_partial_atom_scope():
    rng=np.random.default_rng(19)
    atoms=rng.integers(0,50,(51,52,53));scope=rng.random(atoms.shape)>.99
    initial={a:a%7 for a in range(1,50)};final={a:a%5 for a in range(1,50)}
    labels,counts=np.unique(atoms[atoms!=0],return_counts=True)
    assert scope_attachment(atoms,initial,final,scope)==scope_attachment(atoms,initial,final,scope,atom_counts=(labels,counts))
    whole=scope_attachment(atoms,initial,final,atoms!=0,atom_counts=(labels,counts))
    assert whole['newly_attached_outside_voxels']==whole['initial_outside_voxels']==0
