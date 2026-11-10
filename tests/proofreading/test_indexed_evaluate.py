import numpy as np
import pytest
from proofreading.indexed_evaluate import weighted_metrics,extension_evidence
from proofreading.evaluate import metrics


def test_weighted_dense_parity_and_large_ids():
    atoms=np.array([2**63+1,2**63+2,2**63+3,2**63+4],dtype=np.uint64)
    refs=np.array([11,11,12,0],dtype=np.uint64);weights=np.array([3,5,2,7])
    partition={int(atoms[0]):30,int(atoms[1]):31,int(atoms[2]):31,int(atoms[3]):31}
    dense=metrics(np.repeat(atoms,weights),partition,np.repeat(refs,weights),np.ones(weights.sum(),bool))
    sparse=weighted_metrics(atoms,partition,refs,weights)
    for key in dense:
        if key!='scope':assert sparse[key]==pytest.approx(dense[key])


def test_pair_counts_do_not_round_large_integer_weights():
    weights=np.array([2**53+1,3],dtype=np.uint64)
    result=weighted_metrics(np.array([1,2]),{1:7,2:7},np.array([10,20]),weights)
    assert result['false_merge_voxel_pairs']==int(weights[0])*3


def test_unknown_additions_and_incomplete_reference_are_not_success():
    a=np.array([1,2,3,4,5]);r=np.array([10,10,20,0,10]);n=np.array([10,20,30,40,50])
    result=extension_evidence(a,r,n,{1},{1,2,3,4},10)
    assert result['recall'] is None
    assert result['added']['target_voxels']==20
    assert result['added']['other_certified_neuron_voxels']==30
    assert result['added']['unknown_voxels']==40
    # Atom5 leaves the crop and reconnects; it is excluded from reachable recall.
    result=extension_evidence(a,r,n,{1},{1,2,3,4},10,{1,2})
    assert result['recall']['final']==1
    assert result['recall']['initial']==pytest.approx(1/3)
    with pytest.raises(ValueError):extension_evidence(a,r,n,{1},{1,6},10)
    with pytest.raises(ValueError):extension_evidence(a,r,n,{1},{1},10,{1,3})


@pytest.fixture
def scorer_case(tmp_path):
    import json
    from types import SimpleNamespace
    from proofreading.indexed import Overlay
    from proofreading.indexed_evaluate import CertifiedScorer
    from proofreading.evaluate import file_hash
    atoms=np.array([2**56+1,2**56+2,2**56+3,2**56+4],dtype=np.uint64)
    d=SimpleNamespace(geometry_fingerprint='geometry',fingerprint='dataset',
        owners=np.array([30,31,31,31],dtype=np.uint64),atom_cnt=np.array([3,5,2,7]),
        atoms=SimpleNamespace(data=np.zeros((17,1,1),dtype=np.uint8)))
    def dense(values):
        values=np.asarray(values,dtype=np.uint64);pos=np.searchsorted(atoms,values)
        if np.any(pos>=len(atoms)) or not np.array_equal(atoms[pos],values):raise ValueError('Unknown atom')
        return pos
    d.to_dense=dense;d.initial=Overlay(d)
    spec={'geometry_sha256':'geometry','arrays':{}}
    for name,data in [('atoms',atoms[:3]),('reference',np.array([11,11,12])),('voxel_counts',np.array([3,5,2]))]:
        path=tmp_path/(name+'.npy');np.save(path,data)
        spec['arrays'][name]={'file':path.name,'sha256':file_hash(path)}
    (tmp_path/'reference-spec.json').write_text(json.dumps(spec))
    return CertifiedScorer(d,tmp_path),atoms,d,tmp_path


def test_cached_baseline_delta_matches_full_recomputation(scorer_case):
    scorer,atoms,d,path=scorer_case
    baseline=scorer.build_baseline(path/'baseline.npy')
    truth=np.array([11,11,12,0]);counts=np.array([3,5,2,7])
    expected=weighted_metrics(atoms,d.initial,truth,counts)
    for key,value in baseline.items():assert value==pytest.approx(expected[key])
    changed=d.initial.copy();changed[int(atoms[1])]=30;changed[int(atoms[2])]=33;changed[int(atoms[3])]=30
    scorer.baseline=None;scorer.load_baseline(path/'baseline.npy')
    result=scorer.score(changed)
    expected=weighted_metrics(atoms,changed,truth,counts)
    for key,value in result['final'].items():assert value==pytest.approx(expected[key])
    assert result['changed_unknown_voxels']==7
    assert result['changed_certified_atoms']==2
    assert result['final']['vi_total_bits']==0
    assert scorer.score(d.initial)['vi_improvement_bits']==0


def test_sparse_partition_and_reference_integrity(scorer_case):
    from proofreading.indexed_evaluate import load_partition,CertifiedScorer
    scorer,atoms,d,path=scorer_case
    np.savez(path/'partition.npz',atoms=atoms[1:2],segments=np.array([30],dtype=np.uint64),
             format='sparse-overlay-v1',dataset_sha256=d.fingerprint)
    partition=load_partition(d,path/'partition.npz')
    assert partition[int(atoms[0])]==30 and partition[int(atoms[1])]==30
    d.fingerprint='different'
    with pytest.raises(ValueError,match='fingerprint'):load_partition(d,path/'partition.npz')
    np.save(path/'reference.npy',np.array([11,12,12]))
    with pytest.raises(ValueError,match='identity'):CertifiedScorer(d,path)


def test_new_merge_evidence_excludes_preexisting_errors_and_unknown():
    from proofreading.indexed_evaluate import new_merge_evidence
    report=new_merge_evidence(
        [1,1,2,3,4], [9,9,9,9,4], [10,20,10,0,20], [5,3,7,11,2])
    merged=next(r for r in report if r['final_segment']=='9')
    assert merged['new_false_merge_voxel_pairs']==3*7
    assert merged['unknown_voxels']==11
    assert len(merged['new_cross_neuron_joins'])==1
    assert next(r for r in report if r['final_segment']=='4')['new_false_merge_voxel_pairs']==0
    # Exact integer counts beyond floating-point precision.
    large=new_merge_evidence([1,2],[3,3],[10,20],[10**9+7,10**9+9])
    assert large[0]['new_false_merge_voxel_pairs']==(10**9+7)*(10**9+9)


def test_signed_atom_index_rejected_before_rounded_lookup(scorer_case):
    import json
    from proofreading.evaluate import file_hash
    from proofreading.indexed_evaluate import CertifiedScorer
    _, atoms, dataset, path = scorer_case
    np.save(path/'atoms.npy',atoms[:3].astype(np.int64))
    spec=json.loads((path/'reference-spec.json').read_text())
    spec['arrays']['atoms']['sha256']=file_hash(path/'atoms.npy')
    (path/'reference-spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='uint64'):
        CertifiedScorer(dataset,path)
