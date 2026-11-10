"""Geometric correctness and reuse, independent of renderer implementation."""
import numpy as np
from PIL import Image
from proofreading.raster import render
from proofreading.view import color
from proofreading.scene import geometry


def test_depth_camera_and_buffer_reuse(tmp_path):
    # Camera looks along +Y; the nearer triangle must win regardless of draw order.
    vertices=np.array([[-1000,1000,-1000],[1000,1000,-1000],[0,1000,1000]],float)
    faces=np.array([[0,1,2]],np.uint32)
    near=(11,vertices,faces);far=(12,vertices-[0,2000,0],faces)
    c={'target_nm':[0,0,0],'azimuth':90,'elevation':0,'view_height_nm':4000}
    bounds=[[-2000]*3,[2000]*3];plane=np.array([[10000]*3]*5)
    meshes=[near,far];path=tmp_path/'image.png'
    first=render(meshes,c,bounds,plane,path)
    assert np.allclose(np.asarray(Image.open(path))[350,450],color(11),atol=1)
    second=render(meshes,c,bounds,plane,path)
    assert first['uploaded'] and not second['uploaded']
    render([far,near],c,bounds,plane,path)
    assert np.allclose(np.asarray(Image.open(path))[350,450],color(11),atol=1)
    c['target_nm']=[100000,0,0]
    render(meshes,c,bounds,plane,path)
    assert tuple(np.asarray(Image.open(path))[350,450])==(9,21,29)


def test_mesh_cache_selection_edit_and_undo(tmp_path):
    from proofreading import Dataset, Session
    from proofreading.demo import create
    create(tmp_path/'data')
    session=Session(Dataset.load(tmp_path/'data/dataset.json'),tmp_path/'episode')
    session.execute('select',{'segments':['11'],'mode':'replace'})
    original=geometry(session)
    session.execute('select',{'segments':['12'],'mode':'replace'})
    other=geometry(session);assert other is not original
    session.execute('select',{'segments':['11']})
    combined=geometry(session)
    assert combined[0][1] is original[0][1] and combined[1][1] is other[0][1]
    session.execute('select',{'segments':['11'],'mode':'replace'})
    assert geometry(session) is original
    preview=session.execute('preview_merge',{'segments':['11','12']})['result']
    assert session.execute('commit',{'preview_id':preview['preview_id']})['ok']
    merged=geometry(session);assert merged is not original
    assert session.execute('undo_edit')['ok']
    restored=geometry(session)
    assert restored[0][0]==original[0][0]
    np.testing.assert_array_equal(restored[0][1],original[0][1])
    np.testing.assert_array_equal(restored[0][2],original[0][2])


def test_disk_mesh_reuse_and_membership_identity(tmp_path,monkeypatch):
    from proofreading import Dataset,Session
    from proofreading.demo import create
    from proofreading.mesh_cache import path_for
    create(tmp_path/'data');d=Dataset.load(tmp_path/'data/dataset.json')
    monkeypatch.setenv('CONNECTOME_MESH_CACHE',str(tmp_path/'cache'))
    a=Session(d,tmp_path/'a');a.select(['11'])
    first=geometry(a);assert a._scene_geometry_sources==['extracted']
    b=Session(d,tmp_path/'b');b.select(['11'])
    second=geometry(b);assert b._scene_geometry_sources==['disk']
    np.testing.assert_array_equal(first[0][1],second[0][1])
    np.testing.assert_array_equal(first[0][2],second[0][2])
    atoms=a._segment('11')[1]
    assert path_for(d,atoms,1)!=path_for(d,atoms,2)
    assert path_for(d,atoms,1)!=path_for(d,atoms|a._segment('12')[1],1)
    preview=b.execute('preview_merge',{'segments':['11','12']})['result']
    assert b.execute('commit',{'preview_id':preview['preview_id']})['ok']
    merged=geometry(b);assert merged[0][1].shape != second[0][1].shape
    assert b._scene_geometry_sources==['memory']


def test_unrelated_edit_reuses_unchanged_mesh(tmp_path):
    from proofreading import Dataset,Session
    from proofreading.demo import create
    create(tmp_path/'data')
    s=Session(Dataset.load(tmp_path/'data/dataset.json'),tmp_path/'episode')
    original=geometry(s,[11])
    # Change another segment's membership and advance the edit revision.
    atom=next(a for a,v in s.partition.items() if v==12)
    s.partition[atom]=99;s.revision+=1
    unchanged=geometry(s,[11])
    assert s._scene_geometry_sources==['memory']
    assert unchanged[0][1] is original[0][1]
    assert unchanged[0][2] is original[0][2]


def test_camera_only_changes_reuse_em_plane_but_selection_invalidates(tmp_path,monkeypatch):
    import proofreading.session as module
    from proofreading import Dataset,Session
    from proofreading.demo import create
    create(tmp_path/'data')
    s=Session(Dataset.load(tmp_path/'data/dataset.json'),tmp_path/'episode')
    s.view.width=s.view.height=32
    calls=[];original=module.render
    def counted(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(module,'render',counted)
    s.look();s.look()
    assert len(calls)==1
    s.set_scene(azimuth=80)
    s.look()
    assert len(calls)==1
    s.select(['11']);s.look()
    assert len(calls)==2


def test_gpu_reuses_unchanged_parts_but_reuploads_when_origin_changes(tmp_path):
    from proofreading.raster import Renderer
    renderer=Renderer()
    v=np.array([[0,0,0],[100,0,0],[0,100,0]],float);f=np.array([[0,1,2]],np.uint32)
    w=v+1000
    a=[(1,v,f)];both=[(1,v,f),(2,w,f)]
    assert renderer.upload(a,np.zeros(3))
    first=renderer.buffers[0][1]
    assert renderer.upload(both,np.zeros(3))
    assert renderer.buffers[0][1] is first
    assert not renderer.upload([(99,v,f)],np.zeros(3))  # Label/color change only.
    assert renderer.upload(a,np.ones(3))  # Physical coordinate origin changed.
    assert renderer.buffers[0][1] is not first
