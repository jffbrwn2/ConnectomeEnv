import numpy as np
import pytest
from proofreading.multiscale_em import MultiscaleEM
from proofreading.volume import Grid
from proofreading.view import View


def pyramid():
    scales = [dict(resolution=[n,n,40], voxel_offset=[32//n,64//n,2], size=[128//n,128//n,4]) for n in [8,16,32]]
    calls = []
    def loader(mip, lo, hi):
        calls.append((mip,lo.copy(),hi.copy()))
        resolution = np.array(scales[mip]['resolution'])
        coords = np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'),-1)
        return ((coords+.5)*resolution).sum(-1)
    original = Grid(loader(2,np.array([1,2,2]),np.array([5,6,6])), [32,32,40], [32,64,80])
    em = MultiscaleEM(original, {'source':'fake','info':{'scales':scales}}, loader)
    calls.clear()
    return em,calls


def test_explicit_resolution_not_zoom():
    em,calls = pyramid()
    for zoom in [1,8,16,128]:
        view = View([96,128,160], pixel_size_nm=zoom)
        assert em.for_view(view) is em.original
        assert em.describe(view)['resolution_nm'] == [32,32,40]
    assert not calls
    assert em.describe(View([96,128,160],em_resolution_nm=8))['resolution_nm'] == [8,8,40]
    with pytest.raises(ValueError):
        em.for_view(View([96,128,160],em_resolution_nm=4))


def test_physical_alignment_oblique_offset_and_cache():
    em,calls=pyramid()
    q=2**-.5
    view=View([96,128,160],right=[q,q,0],down=[0,0,1],width=16,height=16,pixel_size_nm=2,em_resolution_nm=8)
    yy,xx=np.mgrid[:16,:16]
    xyz=view.xyz(xx,yy)
    grid=em.for_view(view)
    actual,valid=grid.linear(xyz)
    assert valid.all()
    np.testing.assert_allclose(actual, xyz.sum(-1),atol=1e-10)
    n=len(calls)
    grid.linear(xyz)
    assert len(calls)==n
    assert em.cache_bytes<=em.max_cache_bytes


def test_bounded_cutouts_crop_and_source_edges():
    em,calls=pyramid()
    em.max_cutout_voxels=32
    grid=em.for_view(View([96,128,160],em_resolution_nm=8))
    xyz=np.array([[32,64,80],[159,191,239],[31,64,80],[160,128,160],[96,128,160]],float)
    values,valid=grid.linear(xyz)
    assert valid.tolist()==[True,True,False,False,True]
    assert values[0]==sum([36,68,100])
    assert values[1]==sum([156,188,220])
    assert values[-1]==384
    assert all(np.prod(hi-lo)<=32 for _,lo,hi in calls)


def test_action_resolution_undo_and_failed_fetch(tmp_path, monkeypatch):
    from proofreading.session import Session
    from proofreading.volume import Dataset
    em,_=pyramid()
    atoms=Grid(np.ones(em.original.data.shape,dtype=np.uint64),[32,32,40],[32,64,80])
    dataset=Dataset(atoms,em.original,{1:1})
    dataset.em=em
    dataset.segmentation_levels=[atoms, Grid(np.ones((16,16,4),dtype=np.uint64),[8,8,40],[32,64,80])]
    session=Session(dataset,tmp_path/'episode')
    session.view.width=session.view.height=16
    monkeypatch.setattr(session,'_paired_scene',lambda path: {'image':path})
    original_view=session.view
    response=session.execute('set_em_resolution',{'xy_nm':8})
    assert response['ok']
    assert response['result']['em_sampling']['resolution_nm']==[8,8,40]
    assert session.view.pixel_size_nm==original_view.pixel_size_nm
    assert session.execute('undo_action')['ok']
    assert session.view==original_view
    assert not session.execute('set_em_resolution',{'xy_nm':4})['ok']
    assert session.view==original_view
    def fail(*args):raise ValueError('EM download failed')
    em.loader=fail
    em.cache.clear()
    em.cache_bytes=0
    assert not session.execute('set_em_resolution',{'xy_nm':8})['ok']
    assert session.view==original_view


def test_cropped_dataset_pins_source_without_changing_default_pixels(tmp_path):
    import json
    from proofreading.demo import create
    from proofreading.volume import Dataset
    from proofreading.view import render
    path=create(tmp_path/'case')
    original=Dataset.load(path)
    view=View([2920,3920,5480],width=16,height=16,pixel_size_nm=40)
    before,_=render(original,original.initial,set(),set(),view)
    config=json.loads(path.read_text())
    config['em_pyramid']={'source':'precomputed://https://example.test/em','info':{'scales':[
        {'resolution':[20,20,80],'voxel_offset':[0,0,0],'size':[1000,1000,1000]},
        {'resolution':[40,40,80],'voxel_offset':[0,0,0],'size':[500,500,1000]}]}}
    path.write_text(json.dumps(config))
    finer=Dataset.load(path)
    after,_=render(finer,finer.initial,set(),set(),view)
    assert np.array_equal(before,after)
    assert finer.fingerprint!=original.fingerprint
    assert finer.geometry_fingerprint==original.fingerprint
    assert finer.em.describe(view)['available_xy_nm']==[20,40]
    config['em_pyramid']['source']='precomputed://https://example.test/other'
    path.write_text(json.dumps(config))
    assert Dataset.load(path).fingerprint!=finer.fingerprint


def test_streaming_sampling_matches_legacy_and_reuses_pan_chunks():
    em,calls=pyramid()
    config={'source':em.source,'info':{**em.info,'data_type':'float64'},'streaming':True}
    streamed=MultiscaleEM(em.original,config,em.loader)
    q=2**-.5
    view=View([96,128,160],right=[q,q,0],down=[0,0,1],width=16,height=16,pixel_size_nm=2,em_resolution_nm=8)
    yy,xx=np.mgrid[:16,:16];xyz=view.xyz(xx,yy)
    expected,ok=em.for_view(view).linear(xyz)
    actual,valid=streamed.for_view(view).linear(xyz)
    np.testing.assert_array_equal(valid,ok)
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    n=len(calls)
    streamed.for_view(view).linear(xyz+[1,0,0])
    assert len(calls)==n
