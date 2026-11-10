import json
import numpy as np
import pytest
from proofreading import Dataset, Session
from proofreading.volume import Grid
from proofreading.classifier_reconstruction import letterbox, reconstruct_observation


def test_letterbox_preserves_shape_and_masks():
    x=np.zeros((7,70,90),np.float16)
    x[0,20:50,30:60]=1;x[5]=x[0]
    out,pad=letterbox(x,90)
    assert pad==dict(x=0,y=10,width=90,height=70,resolution=90)
    np.testing.assert_array_equal(out[:,10:80],x)
    assert not out[:,:10].any() and not out[:,80:].any()
    with pytest.raises(ValueError):letterbox(x,0)


def make(tmp_path):
    a=np.zeros((24,20,20),np.uint64)
    a[3:8,5:15,5:15]=1;a[13:18,5:15,5:15]=2
    d=Dataset(Grid(a,[8,8,8],[0,0,0]),Grid(np.full(a.shape,100,np.uint8),[8,8,8],[0,0,0]),{1:10,2:20})
    s=Session(d,tmp_path/'episode')
    result=s.execute('select',{'segments':['10','20']})
    assert result['ok']
    image=result['result']['observation']['scene_observation']['image']
    preview=s.execute('preview_merge',{'segments':['10','20']})
    assert preview['ok']
    assert s.execute('commit',{'preview_id':preview['result']['preview_id']})['ok']
    return d,s,image,preview['result']['scene_observation']['image']


def test_replay_captures_historical_and_preview_meshes(tmp_path,monkeypatch):
    d,s,image,preview=make(tmp_path)
    captured=[]
    def writer(meshes,camera,output,**kwargs):
        captured.append([sid for sid,_,_ in meshes]);return kwargs['provenance']
    monkeypatch.setattr('proofreading.classifier_reconstruction.write_reconstruction',writer)
    old=reconstruct_observation(d,s.output,image,tmp_path/'audit')
    assert old['rgb_pixels_equal'] and old['revision']==0
    assert captured[-1]==[10,20]
    reconstruct_observation(d,s.output,preview,tmp_path/'preview-audit')
    assert len(captured[-1])==1 and captured[-1][0] not in (10,20)
    assert json.loads((tmp_path/'audit/snapshot.json').read_text())['segment_ids']==['10','20']


def test_replay_refuses_changed_image(tmp_path):
    from PIL import Image
    d,s,image,_=make(tmp_path)
    path=s.output/image
    im=Image.open(path).convert('RGB');im.putpixel((0,0),(255,0,0));im.save(path)
    with pytest.raises(ValueError,match='RGB pixels differ'):
        reconstruct_observation(d,s.output,image,tmp_path/'bad')


@pytest.mark.parametrize('azimuth,elevation',[(0,0),(35,25),(125,55)])
def test_channel_camera_matches_workbench(azimuth,elevation):
    pytest.importorskip('pygfx')
    pytest.importorskip('octarine')
    pytest.importorskip('cloudvolume')
    import trimesh
    from proofreading.classifier_reconstruction import render_channels
    from proofreading.raster import Renderer
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=100)
    mesh.vertices*=np.array([1.,1.7,.6])
    mesh.vertices+=np.array([90.,-40.,50.])
    meshes=[(10,mesh.vertices,mesh.faces)]
    camera=dict(target_nm=[0,0,0],azimuth=azimuth,elevation=elevation,view_height_nm=600)
    native,model,_=render_channels(meshes,camera)
    renderer=Renderer()
    rgb=np.asarray(renderer.render(meshes,camera,[[-1000]*3,[1000]*3],
        [[1e6,1e6,1e6]]*2,None,show_bounds=False)['image'])
    foreground=np.any(rgb!=np.array([9,21,29]),axis=2)
    reconstructed=native[0]>0
    iou=np.sum(foreground&reconstructed)/np.sum(foreground|reconstructed)
    assert iou>.98, iou
    np.testing.assert_array_equal(native[0],native[5])
    assert not native[6].any()
    assert model.shape==(1,7,224,224)
    assert np.isfinite(model).all()
    assert not native[:,~reconstructed].any()


def test_full_reconstruction_exports_pair_channels(tmp_path):
    pytest.importorskip('pygfx')
    pytest.importorskip('octarine')
    pytest.importorskip('cloudvolume')
    d,s,image,_=make(tmp_path)
    result=reconstruct_observation(d,s.output,image,tmp_path/'full')
    assert result['provenance']['rgb_pixels_equal']
    assert [m['segment_id'] for m in result['meshes']]==['10','20']
    with np.load(tmp_path/'full/channels/geometry.npz') as data:
        a=data['native']
        assert a.shape==(7,700,900)
        assert data['views'].shape==(1,7,224,224)
        assert a[5].any() and a[6].any()
        np.testing.assert_array_equal(a[0],np.maximum(a[5],a[6]))
    assert (tmp_path/'full/mesh-snapshot.npz').is_file()
    from proofreading.classifier_reconstruction import reconstruct_snapshot
    reconstruct_snapshot(tmp_path/'full',tmp_path/'rerendered')
    with np.load(tmp_path/'full/channels/geometry.npz') as a, np.load(tmp_path/'rerendered/geometry.npz') as b:
        np.testing.assert_array_equal(a['views'],b['views'])
