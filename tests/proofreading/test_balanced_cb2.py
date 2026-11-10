import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('balanced',Path(__file__).parents[2]/'scripts/proofreading/prepare_balanced_cb2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(kind,i,split='test'):
    return {'sample_type':kind,'split':split,'combined_sample_hash':str(i),'metadata':{'operation_id':str(i),'root_ids':[str(i+100)],'interface_point_nm':[i*10000,10000,10000],'synapse_ctr_pt_nm':[i*10000+8,10000,10000]}}


def test_selection_balanced_deterministic_heldout_exclusions():
    rows=[row(k,i*10+j) for i,k in enumerate(m.KINDS) for j in range(1,5)]
    rows += [row(k,100+i,'train') for i,k in enumerate(m.KINDS)]
    excluded=[{'operation_id':'1','crop_bounds_nm':[[0,0,0],[1,1,1]]}]
    a=m.select(rows,excluded,per_kind=2)
    assert a==m.select(rows[::-1],excluded,per_kind=2)
    assert len(a)==10 and all(x['split']=='test' and x['metadata']['operation_id']!='1' for x in a)
    assert all(sum(x['sample_type']==k for x in a)==2 for k in m.KINDS)


def test_native_controls_preserve_both_partners_and_error_uses_after():
    atoms=np.array([1,2,3,4]);before=np.array([10,10,20,30]);after=np.array([40,40,40,30])
    for kind in ('synapse_control','adjacent_control'):
        truth,scope=m.target_reference(atoms,before,after,[10,20],kind)
        assert truth.tolist()==[10,10,20,0]
        assert scope.tolist()==[True,True,True,False]
    truth,_=m.target_reference(atoms,before,after,[10,20],'merge_edit')
    assert truth.tolist()==[40,40,40,0]
    truth,_=m.target_reference(atoms,before,after,[10],'junction_control')
    assert truth.tolist()==[10,10,0,0]
    with pytest.raises(m.InvalidCase):m.target_reference(atoms,before,after,[99],'junction_control')


def test_synapse_center_uses_synapse_location():
    r=row('synapse_control',1)
    assert m.center(r).tolist()==r['metadata']['synapse_ctr_pt_nm']


def test_no_silent_training_fallback():
    with pytest.raises(ValueError,match='Insufficient'):
        m.select([row(k,i+1,'train') for i,k in enumerate(m.KINDS)],[],per_kind=1)


def test_missing_certified_coverage_is_explicit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import proofreading.experiment as e
    dataset=SimpleNamespace(fingerprint='abc')
    record={'directory':str(tmp_path),'dataset_sha256':'abc','scorers':['human']}
    with pytest.raises(ValueError,match='coverage explanation'):
        e.scorer_for({'mode':'cb2'},record,dataset)
    record['certified_unavailable']='No frozen lookup coverage'
    monkeypatch.setattr(e,'evaluate_frozen',lambda *args: {'verified':True})
    assert e.scorer_for({'mode':'cb2'},record,dataset)('episode')=={'human':{'verified':True}}
    record['scorers']=['certified']
    with pytest.raises(ValueError,match='Invalid CB2'):
        e.scorer_for({'mode':'cb2'},record,dataset)
