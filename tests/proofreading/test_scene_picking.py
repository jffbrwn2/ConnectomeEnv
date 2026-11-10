import numpy as np
import pytest
from proofreading import Dataset,Session
from proofreading.volume import Grid
from proofreading.picking import surface


def make(tmp_path):
    a=np.zeros((24,20,20),np.uint64)
    a[3:8,5:15,5:15]=1;a[13:18,5:15,5:15]=2
    d=Dataset(Grid(a,[8,8,8],[0,0,0]),Grid(np.full(a.shape,100,np.uint8),[8,8,8],[0,0,0]),{1:10,2:2**63+7})
    s=Session(d,tmp_path/'episode')
    assert s.execute('select',{'segments':['10',str(2**63+7)]})['ok']
    return s


def test_frontmost_scene_pick_saved_camera_and_valid_seed(tmp_path):
    s=make(tmp_path)
    view=s.execute('set_scene',{'target_nm':[80,80,80],'azimuth':0,'elevation':0,'view_height_nm':160})['result']
    hit=s.execute('click',{'view_id':view['view_id'],'x':449.5,'y':349.5,'mode':'pick'})
    assert hit['ok'],hit
    assert hit['result']['segment']==str(2**63+7)
    assert s._at(hit['result']['xyz_nm'])==(2,2**63+7)
    rotated=s.execute('set_scene',{'azimuth':180})['result']
    assert s.execute('click',{'view_id':rotated['view_id'],'x':449.5,'y':349.5,'mode':'pick'})['result']['segment']=='10'
    assert s.execute('click',{'view_id':view['view_id'],'x':449.5,'y':349.5,'mode':'pick'})['result']['segment']==str(2**63+7)
    p=s.execute('preview_merge',{'segments':['10',str(2**63+7)]})
    assert p['ok'],p
    assert s.execute('commit',{'preview_id':p['result']['preview_id']})['ok']
    result=s.execute('click',{'view_id':view['view_id'],'x':449.5,'y':349.5,'mode':'pick'})
    assert not result['ok'] and 'Stale' in result['error']


def test_scene_background_selection_and_snapshot_eviction(tmp_path):
    s=make(tmp_path)
    view=s.execute('set_scene',{'target_nm':[80,80,80],'azimuth':0,'elevation':0,'view_height_nm':160})['result']
    before=set(s.selected)
    bad=s.execute('click',{'view_id':view['view_id'],'x':0,'y':0,'mode':'replace'})
    assert bad['ok'] and bad['result']['hit'] is False and s.selected==before
    hit=s.execute('click',{'view_id':view['view_id'],'x':449.5,'y':349.5,'mode':'deselect'})
    assert hit['ok'] and s.selected=={10} and 'observation' in hit['result']
    for i in range(20):s._capture_scene(f'frames/test-{i}.png')
    assert len(s.scene_views)<=16 and view['view_id'] not in s.scene_views


def test_ray_background_and_bounds():
    camera={'target_nm':[0,0,0],'azimuth':0,'elevation':0,'view_height_nm':100}
    with pytest.raises(ValueError,match='No segment'):surface([],camera,[[-100]*3,[100]*3],450,350)
    with pytest.raises(ValueError,match='outside'):surface([],camera,[[-100]*3,[100]*3],900,350)
