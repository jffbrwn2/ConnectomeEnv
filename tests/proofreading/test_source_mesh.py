"""Render-only assembly preserves manifest scope and the generic fallback."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from proofreading.streamed_dataset import CaveSource


@pytest.mark.parametrize('fragment_api',[True,False])
def test_mesh_manifest_scope_fallback_and_disk_reuse(tmp_path,monkeypatch,fragment_api):
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    faces=np.array([[0,1,2]],np.uint32)
    mesh=SimpleNamespace(vertices=vertices,faces=faces)
    calls=[]
    class Box:
        def __init__(self,lo,hi):self.minpt=lo;self.maxpt=hi
    class Mesh:
        @staticmethod
        def concatenate(*fragments):
            assert fragments==(mesh,)
            return mesh
    monkeypatch.setitem(sys.modules,'cloudvolume',SimpleNamespace(Bbox=Box,Mesh=Mesh))
    def manifest(root,box,use_byte_offsets):
        calls.append((root,box));assert use_byte_offsets
        return [mesh]
    def get(root,bounding_box,deduplicate_chunk_boundaries):
        calls.append((root,bounding_box));assert not deduplicate_chunk_boundaries
        return {root:mesh}
    api=SimpleNamespace(get=get)
    if fragment_api:api.get_meshes_via_manifest=manifest
    source=CaveSource.__new__(CaveSource)
    source.config={'segmentation':{'info':{'scales':[{'resolution':[8,8,40]}]}}}
    source.mesh_directory=tmp_path
    source.volumes={'segmentation':SimpleNamespace(mesh=api)}
    bounds=np.array([[8,16,40],[80,160,400]])
    for _ in range(2):
        v,f=source.mesh(2**59+1,bounds)
        np.testing.assert_array_equal(v,vertices);np.testing.assert_array_equal(f,faces)
    assert len(calls)==1 and calls[0][0]==2**59+1
    np.testing.assert_array_equal(calls[0][1].minpt,[1,2,1])
    np.testing.assert_array_equal(calls[0][1].maxpt,[10,20,10])


def test_root_validation_is_pinned_and_rejects_invalid_ids():
    from collections import OrderedDict
    from datetime import datetime,timezone
    source=CaveSource.__new__(CaveSource)
    source._valid_roots=OrderedDict()
    source.timestamp=datetime(2025,5,2,tzinfo=timezone.utc)
    calls=[]
    def valid(roots,timestamp):
        calls.append((roots,timestamp))
        return np.array([roots[0]==101],bool)
    source.client=SimpleNamespace(chunkedgraph=SimpleNamespace(is_latest_roots=valid))
    source.validate_root(101);source.validate_root(101)
    assert calls==[([101],source.timestamp)]
    with pytest.raises(ValueError,match='pinned timestamp'):source.validate_root(999)
    assert 999 not in source._valid_roots
    n = len(calls)
    for root in (-1, 0, 2**64, 2**64+1):
        with pytest.raises(ValueError, match='uint64'):
            source.validate_root(root)
    assert len(calls) == n


def test_source_graph_preserves_affinities_and_protected_edges(monkeypatch):
    import networkx as nx
    from types import SimpleNamespace
    source=CaveSource.__new__(CaveSource)
    edges=np.array([[1,2],[2,3],[3,4]],np.uint64)
    source.client=SimpleNamespace(chunkedgraph=SimpleNamespace(
        get_subgraph=lambda root,bounds:(edges,np.array([9.,.25,np.inf]),np.ones(3))))
    source.members=lambda root:[1,2,3,4]
    from proofreading.source_graph import validate
    monkeypatch.setattr("proofreading.source_graph.complete",lambda s,r:iter([validate(*s.client.chunkedgraph.get_subgraph(r,None)[:2])]))
    graph=source.graph([101])
    assert graph[1][2]['capacity']==9.
    assert graph[2][3]['capacity']==.25
    assert np.isinf(graph[3][4]['capacity'])
    cost,(left,right)=nx.minimum_cut(graph,1,4,capacity='capacity')
    assert cost==.25 and left=={1,2} and right=={3,4}
    source.client.chunkedgraph.get_subgraph=lambda root,bounds:(edges,[1.,np.nan,1.],None)
    with pytest.raises(ValueError,match='invalid affinity'):source.graph([101])


def test_multiscale_download_uses_requested_coordinate_resolution(monkeypatch):
    calls=[]
    class Box:
        def __init__(self,lo,hi):self.lo=lo;self.hi=hi
    monkeypatch.setitem(sys.modules,'cloudvolume',SimpleNamespace(Bbox=Box))
    def download(box,mip,agglomerate,coord_resolution):
        assert mip==2 and not agglomerate
        assert coord_resolution==[32,32,40]
        calls.append((box.lo,box.hi))
        return np.zeros((2,3,4,1),np.uint8)
    source=CaveSource.__new__(CaveSource)
    source.volumes={'em':SimpleNamespace(meta=SimpleNamespace(resolution=lambda mip:[32,32,40]),download=download)}
    a=source.download('em',2,np.array([1,2,3]),np.array([3,5,7]))
    assert a.shape==(2,3,4,1) and len(calls)==1
