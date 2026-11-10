import numpy as np
from proofreading.evaluate import scope_attachment,metrics


def test_perfect_target_score_does_not_hide_unrequested_extension():
    atoms=np.array([1,1,2,3,3,3],dtype=np.uint64).reshape(3,2,1)
    scope=atoms!=3
    reference=np.where(scope,10,0).astype(np.uint64)
    initial={1:1,2:2,3:3};final={1:9,2:9,3:9}
    assert metrics(atoms,final,reference,scope)['vi_total_bits']==0
    result=scope_attachment(atoms,initial,final,scope)
    assert result['newly_attached_outside_voxels']==3
    assert result['initial_outside_voxels']==0


def test_existing_contamination_and_renaming_are_not_new_extensions():
    atoms=np.array([1,2,3,0],dtype=np.uint64).reshape(2,2,1)
    scope=atoms==1
    result=scope_attachment(atoms,{1:5,2:5,3:6},{1:9,2:9,3:6},scope)
    assert result['initial_outside_voxels']==result['final_outside_voxels']==1
    assert result['newly_attached_outside_voxels']==0
    result=scope_attachment(atoms,{1:5,2:5,3:6},{1:9,2:8,3:6},scope)
    assert result['detached_outside_voxels']==1
