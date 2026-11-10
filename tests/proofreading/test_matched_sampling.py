import numpy as np
import pytest
from proofreading.volume import Grid, Dataset
from proofreading.view import View, visible_segments
from proofreading.multiscale_em import MultiscaleEM
from proofreading.sampling import grids, metadata
from proofreading.session import Session


def dataset():
    fine=np.ones((8,8,4),np.uint64)
    fine[1::2]=2  # details that disappear in the coarse grid
    atoms=Grid(fine,[8,8,40],[32,64,80])
    coarse=Grid(np.ones((2,2,4),np.uint64),[32,32,40],[32,64,80])
    em=Grid(np.zeros((2,2,4),np.uint8),[32,32,40],[32,64,80])
    d=Dataset(atoms,em,{1:101,2:202})
    d.segmentation_levels=[atoms,coarse]
    config={'source':'fake','info':{'scales':[dict(resolution=[8,8,40],voxel_offset=[4,8,2],size=[8,8,4]),dict(resolution=[32,32,40],voxel_offset=[1,2,2],size=[2,2,4])]}}
    d.em=MultiscaleEM(em,config,lambda mip,lo,hi:np.zeros(hi-lo,np.uint8))
    return d


def test_both_layers_follow_resolution_and_edits():
    d=dataset();v=View([64,96,160],width=16,height=16,pixel_size_nm=4)
    assert grids(d,v)[1].resolution.tolist()==[32,32,40]
    assert set(visible_segments(d,d.initial,set(),set(),v)[2])=={101}
    v.em_resolution_nm=8
    em,seg=metadata(d,v)
    assert em['resolution_nm']==seg['resolution_nm']==[8,8,40]
    assert set(visible_segments(d,d.initial,set(),set(),v)[2])=={101,202}
    assert set(visible_segments(d,{1:999,2:999},set(),set(),v)[2])=={999}


def test_click_uses_saved_view_resolution_and_undo(tmp_path,monkeypatch):
    d=dataset();s=Session(d,tmp_path/'s');s.view=View([64,96,160],width=16,height=16,pixel_size_nm=4)
    monkeypatch.setattr(s,'_paired_scene',lambda p:{'image':p})
    fine=s.execute('set_em_resolution',{'xy_nm':8})
    assert fine['ok']
    view_id=fine['result']['view_id']
    assert s.execute('undo_action')['ok']
    assert s.view.em_resolution_nm is None
    # x=3 samples x=46nm: fine atom2, coarse atom1. The saved fine image governs clicks.
    assert s.click(view_id,3,8,'pick')['segment']=='202'
    assert s._at([46,98,160])[1]==101


def test_unmatched_resolution_is_explicit_and_transactional(tmp_path,monkeypatch):
    d=dataset();d.segmentation_levels=d.segmentation_levels[1:]
    s=Session(d,tmp_path/'s');s.view.width=s.view.height=16
    monkeypatch.setattr(s,'_paired_scene',lambda p:{'image':p})
    assert s.state()['em_sampling']['available_xy_nm']==[32.0]
    old=s.view
    result=s.execute('set_em_resolution',{'xy_nm':8})
    assert not result['ok'] and 'matching segmentation' in str(result)
    assert s.view==old
    with pytest.raises(ValueError,match='matching segmentation'):
        s.look(em_resolution_nm=8)
    assert s.view==old


def test_matching_resolution_requires_aligned_grid():
    d=dataset();d.segmentation_levels[-1].origin+=1
    with pytest.raises(ValueError,match='not aligned'):
        grids(d,View([64,96,160]))


def test_native_downsampling_has_exact_centers_and_uint64_ids():
    from scripts.proofreading.prepare_matched_cb2 import coarser_labels
    native=np.arange(8*8*2,dtype=np.uint64).reshape(8,8,2)+2**60
    for ratio in (2,4):
        coarse=coarser_labels(native,ratio)
        assert coarse.dtype==np.uint64
        assert coarse.shape==(8//ratio,8//ratio,2)
        for x in range(coarse.shape[0]):
            for y in range(coarse.shape[1]):
                np.testing.assert_array_equal(coarse[x,y],native[x*ratio+ratio//2,y*ratio+ratio//2])
    with pytest.raises(ValueError):coarser_labels(native[:7],4)


def test_manifest_levels_are_frozen_and_reject_unknown_atoms(tmp_path):
    import json
    d=dataset()
    np.save(tmp_path/'atoms.npy',d.atoms.data)
    np.save(tmp_path/'coarse.npy',d.segmentation_levels[-1].data)
    np.save(tmp_path/'em.npy',d.em.original.data)
    np.savez(tmp_path/'partition.npz',atoms=np.array([1,2],np.uint64),segments=np.array([101,202],np.uint64))
    config={'atoms':{'path':'atoms.npy','resolution_nm':[8,8,40],'origin_nm':[32,64,80]},
            'em':{'path':'em.npy','resolution_nm':[32,32,40],'origin_nm':[32,64,80]},
            'partition':'partition.npz','segmentation_pyramid':[{'path':'coarse.npy','resolution_nm':[32,32,40],'origin_nm':[32,64,80]}]}
    path=tmp_path/'dataset.json';path.write_text(json.dumps(config))
    first=Dataset.load(path)
    assert metadata(first,View([64,96,160]))[1]['matched_to_em']
    np.save(tmp_path/'coarse.npy',np.full((2,2,4),2,np.uint64))
    assert Dataset.load(path).fingerprint!=first.fingerprint
    np.save(tmp_path/'coarse.npy',np.full((2,2,4),3,np.uint64))
    with pytest.raises(ValueError,match='missing from the frozen partition'):Dataset.load(path)
