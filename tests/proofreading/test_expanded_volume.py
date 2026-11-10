import pytest
from proofreading.expanded_volume import score_cells, score_maps


def test_good_repair_plus_larger_wrong_addition_is_net_harm():
    # A=100, true continuation B=10, unrelated C=15. Merge A+B+C.
    r=score_cells([(1,4,9,100),(2,4,9,10),(3,4,8,15)])
    assert (r['initial_error_voxels'],r['final_error_voxels']) == (10,15)
    assert r['net_removed_voxels'] == -5 and r['outcome']=='worsened'


def test_wrong_alternative_without_repair_is_penalized():
    r=score_cells([(1,4,9,100),(2,2,9,10),(3,4,8,15)])
    assert r['initial_error_voxels']==10 and r['final_error_voxels']==25


def test_correct_other_end_is_credited():
    r=score_cells([(1,4,9,100),(2,2,9,10),(3,4,9,15)])
    assert r['initial_error_voxels']==25 and r['final_error_voxels']==10


def test_unknown_material_is_not_silently_exact():
    r=score_cells([(1,4,9,100),(2,4,9,10),(3,4,0,15)])
    assert r['unknown_reference_voxels']==15 and r['outcome']=='unverified'


def test_missing_added_weights_raise():
    with pytest.raises(ValueError,match='Complete'):
        score_maps({1:1,2:2},{1:3,2:3},{1:9,2:8},{1:100})


def test_unchanged_partition_and_renaming_have_identical_error():
    r=score_cells([(1,11,9,100),(2,12,9,10),(3,13,8,15)])
    assert r['net_removed_voxels']==0 and r['outcome']=='unchanged'


def test_fragmenting_clean_material_is_penalized():
    r=score_cells([(1,2,9,100),(1,3,9,15)])
    assert r['initial_error_voxels']==0 and r['final_error_voxels']==15
    assert r['error_reduction_percent'] is None


def test_full_correct_split():
    r=score_cells([(1,2,9,100),(1,3,8,15)])
    assert r['initial_error_voxels']==15 and r['final_error_voxels']==0


def test_zero_sampled_volume_is_not_missing_weight():
    assert score_cells([(1,2,9,0)])['measured_voxels']==0


def test_component_matching_agrees_with_dense_assignment():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from proofreading.expanded_volume import assignment_error
    rng = np.random.default_rng(421)
    for _ in range(100):
        weights = rng.integers(0,100,size=(15,18),dtype=np.int64)
        weights[rng.random(weights.shape)<.9]=0
        rows,cols=linear_sum_assignment(weights,maximize=True)
        cells={(int(i)+1,int(j)+1):int(weights[i,j]) for i,j in zip(*np.nonzero(weights))}
        assert assignment_error(cells)==int(weights.sum()-weights[rows,cols].sum())
    # No quadratic allocation for unrelated correctly assigned material.
    assert assignment_error({(i,i):1 for i in range(1,10001)})==0
