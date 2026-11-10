import json
import numpy as np
import networkx as nx
from proofreading.streamed_dataset import StreamedDataset
from proofreading.session import Session


class Source:
    def __init__(self):self.known={};self.mesh_calls=0;self.member_calls=0
    def ensure(self,atoms):
        result={int(a):101 if int(a)<=2 else 102 for a in atoms};self.known.update(result);return result
    def owner(self,a):return self.known[a]
    def observed(self,r):return [a for a,s in self.known.items() if s==r]
    def members(self,r):
        self.member_calls+=1
        atoms=[1,2] if r==101 else [3,4];self.ensure(atoms);return atoms
    def download(self,channel,mip,lo,hi):
        res=8*2**mip
        x=(np.arange(lo[0],hi[0])+.5)*res
        labels=(x//1024+1).astype(np.uint64)
        return np.broadcast_to(labels[:,None,None],hi-lo).astype(np.uint8 if channel=='em' else np.uint64).copy()
    def mesh(self,root,bounds):
        self.mesh_calls+=1
        x=512 if root==101 else 3072
        return np.array([[x,900,900],[x+50,1100,900],[x,1100,1100]],float),np.array([[0,1,2]],np.uint32)
    def graph(self,roots):
        g=nx.Graph()
        for r in roots:g.add_edge(*( (1,2) if r==101 else (3,4)),capacity=1.)
        return g


def make(tmp_path):
    scales=[{'resolution':[n,n,40],'voxel_offset':[0,0,0],'size':[4096//n,4096//n,64]} for n in (8,16,32)]
    config={k:{'source':k,'info':{'data_type':'uint8' if k=='em' else 'uint64','scales':scales}} for k in ('em','segmentation')}
    config.update(backend='streamed-world-v1',cache_directory=str(tmp_path/'cache'),
                  initial_view={'center_nm':[512,1024,1000],'width':32,'height':32,'pixel_size_nm':16})
    path=tmp_path/'dataset.json';path.write_text(json.dumps(config))
    source=Source();dataset=StreamedDataset(path,source)
    return Session(dataset,tmp_path/'episode'),source


def test_session_exploration_click_global_merge_and_undo(tmp_path):
    s,source=make(tmp_path)
    first=s.execute('look');assert first['ok']
    assert s.execute('click',{'view_id':first['result']['view_id'],'x':16,'y':16})['ok']
    calls=source.mesh_calls
    assert s.execute('set_scene',{'azimuth':80})['ok']
    assert source.mesh_calls==calls
    distant=s.execute('set_view',{'center_nm':[3584,1024,1000]});assert distant['ok']
    picked=s.execute('click',{'view_id':distant['result']['view_id'],'x':16,'y':16})
    assert picked['ok'] and picked['result']['segment']=='102'
    members_before=source.member_calls
    preview=s.execute('preview_merge',{'segments':['101','102']});assert preview['ok']
    committed=s.execute('commit',{'preview_id':preview['result']['preview_id']});assert committed['ok']
    child=int(committed['result']['children'][0])
    # Material in a previously unseen region inherits the identity-level merge.
    s.execute('set_view',{'center_nm':[1536,1024,1000]})
    assert s.partition[2]==child
    assert source.member_calls==members_before
    assert s.execute('undo_edit')['ok'] and s.partition[2]==101
    assert s.execute('redo_edit')['ok'] and s.partition[2]==child
    with np.load(s.output/'partition.npz') as z:
        assert z['format'].item()=='streaming-rules-v1'
        assert len(json.loads(z['rules'].item()))==1


def test_session_split_loads_full_membership_before_preview(tmp_path):
    s,source=make(tmp_path)
    s.execute('look')
    bad=s.execute('preview_split',{'segment':'101','groups':[['1'],['99']]})
    assert not bad['ok'] and source.member_calls>0
    good=s.execute('preview_split',{'segment':'101','groups':[['1'],['2']]})
    assert good['ok'],good
    assert s.execute('commit',{'preview_id':good['result']['preview_id']})['ok']
    assert s.partition[1]!=s.partition[2]


def test_cached_split_mesh_retains_display_resolution(tmp_path):
    s,_=make(tmp_path)
    s.execute('look')
    p=s.execute('preview_split',{'segment':'101','groups':[['1'],['2']]})
    assert p['ok']
    c=s.execute('commit',{'preview_id':p['result']['preview_id']})
    assert c['ok']
    assert s.execute('select',{'segments':c['result']['children'],'mode':'replace'})['ok']
    first=list(s._scene_geometry_sources)
    assert any(x.startswith('edited-label-native-') for x in first)
    assert s.execute('set_scene',{'azimuth':70})['ok']
    assert all(x in s._scene_geometry_sources for x in first)


def test_source_plane_reused_across_selection_edit_and_contrast(tmp_path,monkeypatch):
    s,_=make(tmp_path)
    assert s.execute('look')['ok']
    em,seg=s.dataset.sampling_grids(s.view)
    def unexpected(*args):raise AssertionError('unchanged plane was sampled again')
    monkeypatch.setattr(em,'linear',unexpected);monkeypatch.setattr(seg,'nearest',unexpected)
    assert s.execute('select',{'segments':['101'],'mode':'replace'})['ok']
    assert s.execute('set_view',{'contrast':[0,180]})['ok']
    p=s.execute('preview_merge',{'segments':['101','102']});assert p['ok']
    c=s.execute('commit',{'preview_id':p['result']['preview_id']});assert c['ok']
    assert s.execute('look')['ok']
    assert s.execute('undo_edit')['ok']


def test_paired_observations_overlap_and_settle_before_failed_action(tmp_path,monkeypatch):
    import threading
    import proofreading.session as module
    s,_=make(tmp_path)
    barrier=threading.Barrier(2,timeout=5)
    finished=threading.Event()
    original=module.render
    def plane(*args,**kwargs):
        barrier.wait()
        try:return original(*args,**kwargs)
        finally:finished.set()
    def failed_scene(path):
        barrier.wait()
        raise ValueError('failed scene')
    monkeypatch.setattr(module,'render',plane)
    monkeypatch.setattr(s,'_paired_scene',failed_scene)
    result=s.execute('look')
    assert not result['ok'] and 'failed scene' in str(result)
    assert finished.is_set()
    assert not s.views and not hasattr(s,'_em_plane_cache')


def test_select_unobserved_root_validates_without_enumerating_members(tmp_path):
    s,source=make(tmp_path)
    checked=[]
    def validate(root):
        checked.append(root)
        if root not in (101,102):raise ValueError('not live at pinned time')
    source.validate_root=validate
    assert s.execute('select',{'segments':['101'],'mode':'replace'})['ok']
    assert checked==[101] and source.member_calls==0
    assert s.selected=={101}
    result=s.execute('select',{'segments':['999'],'mode':'replace'})
    assert not result['ok'] and s.selected=={101}
    # A complete edit still obtains membership rather than treating observation
    # samples as the entire neuron.
    result=s.execute('preview_split',{'segment':'101','groups':[['1'],['2']]})
    assert result['ok'] and source.member_calls>0


def test_native_mesh_keeps_thin_process_across_tile_boundaries(tmp_path):
    from skimage.measure import marching_cubes,mesh_surface_area
    s,source=make(tmp_path)
    # One native voxel thick: a coarse nearest-neighbor mesh can lose it entirely.
    def download(channel,mip,lo,hi):
        indices=np.stack(np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)),indexing='ij'),-1)
        return ((indices[...,1]==7)&(indices[...,2]==8)&(indices[...,0]>=10)&(indices[...,0]<270)).astype(np.uint64)
    source.download=download
    s.dataset.sources.loader=download
    for grid in s.dataset.sources.grids.values():
        grid.loader=lambda lo,hi:download('segmentation',0,lo,hi)
    bounds=np.array([[0,0,0],[280*8,24*8,16*40]])
    v,f=s.dataset._edited_mesh({1},bounds)
    mask=download('segmentation',0,np.zeros(3,int),np.array([280,24,16])).astype(bool)
    expected,faces,_,_=marching_cubes(np.pad(mask,1).astype(np.float32),.5)
    expected=(expected-1)*[8,8,40]+[4,4,20]
    assert np.allclose(v.min(0),expected.min(0)) and np.allclose(v.max(0),expected.max(0))
    assert np.isclose(mesh_surface_area(v,f),mesh_surface_area(expected,faces))
    # Rotations/previews with the same membership reuse persisted geometry.
    s.dataset.atoms.read_box=lambda *args:(_ for _ in ()).throw(AssertionError('remeshed cached edit'))
    v2,f2=s.dataset._edited_mesh({1},bounds)
    assert np.array_equal(v,v2) and np.array_equal(f,f2)


def test_edit_reuses_intact_fragments_and_only_remeshes_mixed_ones(tmp_path,monkeypatch):
    s,source=make(tmp_path)
    s.partition.apply_groups("split",[101],[{1},{2}],[201,202])
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    faces=np.array([[0,1,2]],np.uint32)
    source.fragment_members=lambda root:{11:{1},12:{1,2}}
    source.mesh_fragments=lambda root,bounds:[(11,vertices,faces),(12,vertices+100,faces)]
    remeshed=[]
    def mesh(atoms,bounds):remeshed.append(set(atoms));return vertices+200,faces
    monkeypatch.setattr(s.dataset,'_edited_mesh',mesh)
    pieces=s.dataset.scene_geometry(s,[201,202],1)
    assert any(v is vertices for _,v,f in pieces)
    assert remeshed==[{1},{2}]


def test_protected_source_connection_rejects_split_without_mutation(tmp_path):
    s,source=make(tmp_path)
    def protected(roots):
        g=nx.Graph();g.add_edge(1,2,capacity=float('inf'));return g
    source.graph=protected
    result=s.execute('preview_split',{'segment':'101','source_nm':[[512,1024,1000]],'sink_nm':[[1536,1024,1000]]})
    assert not result['ok'] and 'protected' in result['error']
    assert not s.partition.rules and not s.previews


def test_unavailable_historical_pixels_do_not_block_images_or_become_editable(tmp_path):
    from proofreading.ownership_cache import OwnershipCache,MissingOwnership
    original,source=make(tmp_path)
    d=original.dataset
    owners=OwnershipCache('missing-test',lambda a:np.where(a==1,0,101).astype(np.uint64),tmp_path/'owners')
    source.observe=owners.observe;source.ensure=owners.ensure;source.owner=owners.owner
    d.sources.level('segmentation',d.sources.native).on_labels=source.observe
    s=Session(d,tmp_path/'missing-episode')
    s.view.center_nm=[512,1024,1000];s.view.width=16;s.view.height=16;s.view.pixel_size_nm=8
    result=s.execute('look')
    assert result['ok'],result
    obs=result['result']
    assert obs['segmentation_coverage']['unavailable_pixels']==256
    assert obs['visible_segments']==[]
    before=s.state();hit=s.execute('click',{'view_id':obs['view_id'],'x':8,'y':8,'mode':'replace'})
    assert hit['ok'] and not hit['result']['hit']
    assert hit['result']['reason']=='unavailable_historical_ownership' and s.state()==before
    import pytest
    with pytest.raises(MissingOwnership):s._at([512,1024,1000])
    from PIL import Image
    a=np.asarray(Image.open(s.output/obs['image']))
    assert np.any(a[:,:,0]!=a[:,:,1])  # hatched, not silently ordinary background
    slab=s.execute('look',{'slab_nm':40,'projection':'mean'})
    assert slab['ok'] and slab['result']['segmentation_coverage']['unavailable_pixels']==256
    owners.close()


def test_episode_close_drains_workers_and_releases_memory(tmp_path):
    s, _ = make(tmp_path)
    assert s.execute('look')['ok']
    dataset=s.dataset
    dataset.close()
    dataset.close()
    assert dataset.sources.cache.resident_bytes==0
    import pytest
    with pytest.raises(RuntimeError):dataset.observation_pool.submit(lambda:None)
    with pytest.raises(RuntimeError):dataset.sources.fetch_pool.submit(lambda:None)
