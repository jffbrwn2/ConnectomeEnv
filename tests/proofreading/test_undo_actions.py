import copy
import pytest
from proofreading import Dataset,Session
from proofreading.demo import create
from proofreading.replay import replay

@pytest.fixture
def session(tmp_path):
    return Session(Dataset.load(create(tmp_path/'data')),tmp_path/'episode')

def call(s,name,**args):
    r=s.execute(name,args)
    assert r['ok'],r
    return r['result']

def merge(s):
    p=call(s,'preview_merge',segments=['11','12'])
    return call(s,'commit',preview_id=p['preview_id'])

def test_undo_action_restores_view_without_undoing_edit(session):
    merge(session);edited=session.partition.copy()
    before=session.view.metadata()
    call(session,'set_view',center_nm=[2200,3300,4400])
    result=call(session,'undo_action')
    assert result['undone_action']['tool']=='set_view'
    assert session.view.metadata()==before
    assert session.partition==edited and session.revision==1
    assert result['observation']['transform']==before
    assert session.state()['undo_edit_depth']==1

def test_undo_edit_skips_navigation_and_action_undo_can_reverse_it(session):
    call(session,'select',segments=['11'])
    merge(session);edited=session.partition.copy()
    call(session,'set_scene',azimuth=77)
    call(session,'set_view',center_nm=[2200,3300,4400])
    view=session.view.metadata();camera=copy.deepcopy(session.scene)
    call(session,'undo_edit')
    assert session.partition==session.dataset.initial
    assert session.view.metadata()==view and session.scene==camera
    result=call(session,'undo_action')
    assert result['undone_action']['tool']=='undo_edit'
    assert session.partition==edited
    assert session.state()['undo_edit_depth']==1 and session.state()['redo_edit_depth']==0

def test_action_history_skips_reads_and_failures(session):
    call(session,'select',segments=['11'])
    call(session,'set_scene',azimuth=71)
    call(session,'look');call(session,'list_segments');call(session,'state')
    call(session,'set_scene',azimuth=71)  # No-op requests do not consume history.
    assert not session.execute('set_scene',{'elevation':100})['ok']
    call(session,'undo_action');assert session.scene['azimuth']==35
    call(session,'undo_action');assert session.selected==set()
    assert not session.execute('undo_action')['ok']

def test_undo_preview_and_annotation_and_finish(session):
    p=call(session,'preview_merge',segments=['11','12'])
    call(session,'undo_action')
    assert not session.execute('commit',{'preview_id':p['preview_id']})['ok']
    newer=call(session,'preview_merge',segments=['11','12'])
    assert newer['preview_id']!=p['preview_id']
    call(session,'annotate',xyz_nm=[1,2,3],text='check')
    call(session,'finish',summary='done')
    call(session,'undo_action');assert not session.finished
    call(session,'undo_action');assert session.annotations==[]

def test_action_undo_commit_keeps_ids_monotonic_and_clicks_stale(session):
    old=call(session,'look')
    children=merge(session)['children']
    call(session,'undo_action')
    assert session.partition==session.dataset.initial and session.revision==2
    assert not session.execute('click',{'view_id':old['view_id'],'x':1,'y':1})['ok']
    newer=merge(session)['children']
    assert min(map(int,newer))>max(map(int,children))

def test_action_undo_failure_preserves_history(session,monkeypatch):
    merge(session)
    state=copy.deepcopy(session.state());partition=session.partition.copy()
    with monkeypatch.context() as m:
        m.setattr(session,'_paired_scene',lambda _: (_ for _ in ()).throw(ValueError('render failed')))
        assert not session.execute('undo_action')['ok']
    assert session.state()==state and session.partition==partition
    call(session,'undo_action');assert session.partition==session.dataset.initial

def test_transient_look_does_not_change_camera_or_history(session):
    call(session,'set_scene',azimuth=70)
    before=copy.deepcopy(session.state())
    view=call(session,'look',center_nm=[2200,3300,4400],right=[0,1,0],down=[0,0,1])
    assert session.state()==before
    assert view['transform']['center_nm']==[2200,3300,4400]
    assert session.views[view['view_id']][0].metadata()==view['transform']
    assert view['scene_observation']['em_transform']==view['transform']

def test_mixed_undo_replay_and_legacy_aliases(session,tmp_path):
    merge(session)
    call(session,'set_scene',azimuth=73)
    call(session,'undo_action')
    assert not session.execute('undo')['ok']
    assert session.execute('undo',replay_legacy=True)['ok']
    call(session,'redo_edit')
    call(session,'undo_action')
    r=replay(session.dataset,session.output,tmp_path/'replayed')
    assert r['partition_equal']

def test_edit_history_snapshots_remain_immutable(session):
    merge(session);first=session.partition.copy()
    call(session,'undo_edit')
    # Undoing the undo must restore the first edit even after redoing another action.
    call(session,'undo_action')
    assert session.partition==first
    call(session,'undo_edit');call(session,'redo_edit')
    assert session.partition==first
