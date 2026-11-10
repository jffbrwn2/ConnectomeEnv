import numpy as np
from test_workbench import dataset, session, call
from proofreading.replay import replay


def test_stack_matches_individual_observations_and_keeps_state(session, tmp_path):
    before=session.state()
    args={'center_nm':[110,215,320], 'right':[0,1,0], 'down':[0,0,1],
          'width':16,'height':16,'pixel_size_nm':1,'overlay':0}
    stack=call(session,'look_stack',offsets_nm=[-4,0,4],**args)
    assert session.state()==before
    from proofreading.harness import content
    for provider,image_type in [('responses','input_image'),('anthropic','image')]:
        blocks=content(stack,session.output,provider)
        assert sum(b['type']==image_type for b in blocks)==6
    for offset,item in zip([-4,0,4],stack['slices']):
        obs=item['observation']
        np.testing.assert_allclose(obs['transform']['center_nm'],[110+offset,215,320])
        single=call(session,'look',**{**args,'center_nm':[110+offset,215,320]})
        assert (session.output/obs['image']).read_bytes()==(session.output/single['image']).read_bytes()
        assert (session.output/obs['scene_observation']['image']).read_bytes()==(session.output/single['scene_observation']['image']).read_bytes()
        hit=call(session,'click',view_id=obs['view_id'],x=7.5,y=7.5,mode='pick')
        np.testing.assert_allclose(hit['xyz_nm'],[110+offset,215,320])
    assert replay(session.dataset,session.output,tmp_path/'replayed')['partition_equal']


def test_stack_rejects_invalid_offsets_without_state_change(session):
    before=session.state()
    for offsets in ([],list(range(10)),[float('inf')]):
        if offsets and not np.isfinite(offsets).all():
            # Direct validation avoids writing a non-JSON infinity to the event log.
            import pytest
            with pytest.raises(ValueError):session.look_stack(offsets)
        else:
            assert not session.execute('look_stack',{'offsets_nm':offsets})['ok']
        assert session.state()==before
        assert not session.views


def test_look_accepts_stack_offsets_without_changing_camera(session):
    before=session.state()
    args={'center_nm':[110,215,320],'width':16,'height':16,'pixel_size_nm':1}
    stack=call(session,'look',offsets_nm=[-4,0,4],**args)
    assert len(stack['slices'])==3 and session.state()==before
    assert not session.execute('look',{'offsets_nm':[]})['ok']
