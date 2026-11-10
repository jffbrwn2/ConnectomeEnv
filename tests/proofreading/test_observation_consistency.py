"""Action/observation contract, failure atomicity, and old recording compatibility."""
import json
import numpy as np
import pytest
from proofreading import Dataset, Session
from proofreading.demo import create
from proofreading.harness import image_paths
from proofreading.replay import replay

@pytest.fixture
def session(tmp_path):
    create(tmp_path/'data')
    return Session(Dataset.load(tmp_path/'data/dataset.json'),tmp_path/'episode')

def call(s, name, **args):
    response=s.execute(name,args)
    assert response['ok'],response
    return response['result']

@pytest.mark.parametrize('mode,ids,expected', [
    ('replace',['12'],['12']),('select',['12'],['11','12']),
    ('deselect',['11'],[]),('hide',['11'],[]),('show',['12'],['11']),
    ('replace',[],[])])
def test_all_selection_modes_refresh(session,mode,ids,expected):
    call(session,'select',segments=['11'])
    result=call(session,'select',segments=ids,mode=mode)
    observed=result['observation']
    assert result['selected']==expected
    assert observed['selected']==expected
    assert observed['scene_observation']['selected']==expected
    assert observed['transform']==observed['scene_observation']['em_transform']
    assert len(list(image_paths(result)))==2
    assert session.revision==0

def test_mesh_override_then_single_selection(session):
    call(session,'mesh',segments=['11','12'])
    result=call(session,'select',segments=['11'],mode='replace')
    assert result['observation']['scene_observation']['selected']==['11']
    assert len(list(image_paths(result)))==2

def test_click_pick_vs_selection(session):
    # Choose a foreground voxel directly from the synthetic volume.
    grid=session.dataset.atoms
    index=np.argwhere(grid.data>0)[0]
    center=(grid.origin+(index+.5)*grid.resolution).tolist()
    view=call(session,'set_view',center_nm=center,width=16,height=16)
    pick=call(session,'click',view_id=view['view_id'],x=7.5,y=7.5,mode='pick')
    assert not list(image_paths(pick))
    clicked=call(session,'click',view_id=view['view_id'],x=7.5,y=7.5,mode='replace')
    assert clicked['observation']['selected']==[pick['segment']]

@pytest.mark.parametrize('name',['select','commit','undo_edit','redo_edit'])
def test_render_failure_rolls_back(session,monkeypatch,name):
    call(session,'select',segments=['11'])
    preview=call(session,'preview_merge',segments=['11','12'])
    args={'segments':['12'],'mode':'replace'} if name=='select' else {}
    if name=='commit':args={'preview_id':preview['preview_id']}
    if name in ('undo_edit','redo_edit'):call(session,'commit',preview_id=preview['preview_id'])
    if name=='redo_edit':call(session,'undo_edit')
    before=session.state();partition=session.partition.copy();next_id=session.next_id
    disk=(session.output/'partition.npz').read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(session,'_paired_scene',lambda _: (_ for _ in ()).throw(ValueError('render failed')))
        assert not session.execute(name,args)['ok']
    assert session.state()==before and session.partition==partition
    assert session.next_id==next_id
    assert (session.output/'partition.npz').read_bytes()==disk
    assert session.execute(name,args)['ok']

def test_failed_preview_cannot_commit(session,monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr('proofreading.scene.render_scene',lambda *_: (_ for _ in ()).throw(ValueError('render failed')))
        assert not session.execute('preview_merge',{'segments':['11','12']})['ok']
    assert not session.previews
    assert not session.execute('commit',{'preview_id':'edit-000001'})['ok']

@pytest.mark.parametrize('legacy',[False,True])
def test_selection_replay_retains_image_ids(session,tmp_path,legacy):
    session.execute('select',{'segments':['11']},refresh_selection=not legacy)
    view=call(session,'look')
    # Stable view IDs are critical for recorded click coordinates.
    session.execute('click',{'view_id':view['view_id'],'x':0,'y':0,'mode':'pick'})
    result=replay(session.dataset,session.output,tmp_path/'replayed')
    assert result['partition_equal']
    assert result['exact_images']==(2 if legacy else 4)
