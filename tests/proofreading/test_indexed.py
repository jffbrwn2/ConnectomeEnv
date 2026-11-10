import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from proofreading import Dataset, Session
from proofreading.replay import replay

@pytest.fixture
def indexed(tmp_path):
    w=tmp_path/'world'; w.mkdir(); g=w/'geom_index';g.mkdir();c=w/'root_atoms_csr';c.mkdir()
    aa=np.array([0,2**56+1,2**56+2,2**56+3],np.int64)
    roots=np.array([864691135838138933,864691135838138934],np.int64)
    dense=np.zeros((132,8,8),np.uint32)
    dense[0:64,2:6,2:6]=2;dense[64:129,2:6,2:6]=1;dense[129:,2:6,2:6]=3
    inverse=aa[[0,2,1,3]]
    np.save(w/'seg.npy',inverse[dense].astype(np.uint64)); np.save(w/'em.npy',np.full(dense.shape,128,np.uint8))
    m={'name':'test80','resolution_xyz_nm':[32,32,40],'bounds_min_xyz':[10,20,30],
       'extra':{'n_initial_roots':2}}
    (w/'manifest.json').write_text(json.dumps(m))
    arrays={'native':aa,'dense':np.array([0,2,1,3]),'dense_cube':dense,
            'atom_cnt':np.bincount(dense.ravel(),minlength=4).astype(float),
            'atom_edges_dense':np.array([[1,2],[1,3]],np.int64),
            'atom_inc_eids':np.array([0,1,0,1]),'atom_inc_indptr':np.array([0,0,2,3,4])}
    positions=[np.argwhere(dense==i) for i in range(1,4)]
    arrays['label_lo']=np.array([p.min(axis=0) for p in positions])
    arrays['label_hi']=np.array([p.max(axis=0)+1 for p in positions])
    arrays['atom_cent_vox']=np.vstack([np.zeros(3),[p.mean(axis=0) for p in positions]])
    for name,arr in arrays.items():np.save(g/(name+'.npy'),arr)
    for name,arr in [('root_ids',roots),('indptr',[0,2,3]),('atoms',aa[1:])]:np.save(c/(name+'.npy'),np.asarray(arr,np.int64))
    out=tmp_path/'prepared'
    script=Path(__file__).resolve().parents[2]/'scripts/proofreading/prepare_indexed_world.py'
    subprocess.run([sys.executable,str(script),str(w),str(out)],check=True)
    return Dataset.load(out/'dataset.json'),aa,roots

def test_ownership_edits_replay_and_bounded_geometry(indexed,tmp_path):
    d,aa,roots=indexed;s=Session(d,tmp_path/'session')
    assert [s.partition[int(a)] for a in aa[1:]]==[roots[0],roots[0],roots[1]]
    point=(d.atoms.origin+np.array([1.5,3.5,3.5])*d.atoms.resolution).tolist()
    view=s.execute('set_view',{'center_nm':point})['result']
    picked=s.execute('click',{'view_id':view['view_id'],'x':255.5,'y':255.5,'mode':'pick'})
    assert picked['ok'] and picked['result']['atom']==str(aa[1])
    assert picked['result']['segment']==str(roots[0])
    info=s.execute('inspect_segment',{'segment':str(roots[0])},replay_legacy=True)['result']
    assert info['n_atoms']==2 and info['voxels']==129*16
    assert info['bounds_nm']==[[320,704,1280],[4448,832,1440]]
    assert info['neighbors']==[str(roots[1])]
    assert 'unit geometric' in s.state()['split_capacity']
    args={'segment':str(roots[0]),'groups':[[str(aa[1])],[str(aa[2])]]}
    preview=s.execute('preview_split',args)['result']
    children=s.execute('commit',{'preview_id':preview['preview_id']})['result']['children']
    assert len(children)==2
    assert len(s.partition.changes)==2
    assert not s.execute('inspect_segment',{'segment':str(roots[0])})['ok']
    s.execute('undo_edit'); assert s.partition[int(aa[1])]==roots[0]
    s.execute('redo_edit');assert s.partition[int(aa[1])]!=roots[0]
    p=s.execute('preview_merge',{'segments':children+[str(roots[1])]})['result']
    merged=s.execute('commit',{'preview_id':p['preview_id']})['result']['children'][0]
    assert d.list_segments(s.partition,0,100)['segments']==[{'id':merged,'voxels':132*16}]
    with np.load(s.output/'partition.npz') as z:
        assert str(z['format'])=='sparse-overlay-v1' and len(z['atoms'])==3
    assert replay(d,s.output,tmp_path/'replay')['partition_equal']

def test_tiled_surface_has_no_caps_or_missing_boundary_faces(indexed):
    from skimage.measure import marching_cubes,mesh_surface_area
    d,aa,_=indexed
    parts=list(d.mesh_parts(set(map(int,aa[1:])),1))
    mask=d.atoms.data!=0
    v,f,_,_=marching_cubes(np.pad(mask,1).astype(np.float32),0.5)
    v=d.atoms.origin+(v-1)*d.atoms.resolution+d.atoms.resolution/2
    actual=sum(mesh_surface_area(vv,ff) for vv,ff in parts)
    assert actual==pytest.approx(mesh_surface_area(v,f),rel=1e-6)
    allv=np.concatenate([p[0] for p in parts])
    assert np.allclose(allv.min(axis=0),v.min(axis=0))
    assert np.allclose(allv.max(axis=0),v.max(axis=0))

def test_modified_source_is_rejected(indexed):
    d,_,_=indexed
    (d.world/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='source changed'):Dataset.load(d.manifest)

def test_unknown_historical_root_is_not_one_neuron(indexed,tmp_path):
    d,aa,roots=indexed
    r=roots.copy();r[0]=0
    np.save(d.world/'root_atoms_csr/root_ids.npy',r)
    out=tmp_path/'unknown-prepared'
    script=Path(__file__).resolve().parents[2]/'scripts/proofreading/prepare_indexed_world.py'
    subprocess.run([sys.executable,str(script),str(d.world),str(out)],check=True)
    d=Dataset.load(out/'dataset.json');s=Session(d,tmp_path/'unknown-session')
    assert s.partition[int(aa[1])]==int(aa[1])
    assert s._segment(str(aa[1]))[1]=={int(aa[1])}
    assert not s.execute('inspect_segment',{'segment':'0'})['ok']
    assert d.list_segments(s.partition,0,100)['total']==3
    p=s.preview_merge([str(aa[1]),str(roots[1])])
    s.commit(p['preview_id'])
    assert d.list_segments(s.partition,0,100)['total']==2
    s.undo();assert d.list_segments(s.partition,0,100)['total']==3


def test_world_relocation_preserves_identity_and_checks(indexed, tmp_path, monkeypatch):
    import shutil
    d, aa, roots = indexed
    moved = tmp_path / 'cloud-world'
    shutil.move(str(d.world), moved)
    monkeypatch.setenv('CONNECTOME_WORLD_ROOT', str(moved))
    relocated = Dataset.load(d.manifest)
    assert relocated.fingerprint == d.fingerprint
    assert relocated.geometry_fingerprint == d.geometry_fingerprint
    assert relocated.initial[int(aa[1])] == int(roots[0])
    (moved / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        Dataset.load(d.manifest)
