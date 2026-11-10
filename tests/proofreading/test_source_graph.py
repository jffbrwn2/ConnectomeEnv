from types import SimpleNamespace
import numpy as np
import pytest
from proofreading.source_graph import boxes,complete,request
from proofreading.streamed_dataset import CaveSource


def test_boxes_cover_occupied_chunks_once_under_limit():
    chunks=np.unique(np.random.default_rng(13).integers(0,50,(1200,3)),axis=0)
    cover=np.zeros(len(chunks),int)
    for lo,hi in boxes(chunks):
        inside=np.all((chunks>=lo)&(chunks<hi),axis=1)
        assert 0<inside.sum()<=128
        cover+=inside
    assert np.all(cover==1)


def source(tmp_path):
    s=CaveSource.__new__(CaveSource);s.directory=tmp_path
    s.members=lambda root:np.arange(1,261,dtype=np.uint64)
    meta=SimpleNamespace(decode_chunk_position=lambda a:[a,0,0],graph_chunk_size=[64,64,64],
                         chunks_start_at_voxel_offset=True,watershed_mip=0,voxel_offset=lambda mip:[10,20,30])
    s.volumes={'segmentation':SimpleNamespace(meta=meta)}
    calls=[]
    def get(url,params,timeout):
        assert params['only_internal_edges']=='false' and params['int64_as_str']=='true'
        bounds=np.array([[int(v) for v in axis.split('-')] for axis in params['bounds'].split('_')])
        lo,hi=(bounds[0]-10)//64
        calls.append((lo,hi)); assert hi-lo<=128
        # Chain plus a nonlocal, protected manual edge. Both endpoints must
        # survive even though they never occupy the same query box.
        edges=[[a,a+1] for a in range(1,260) if lo<=a<hi or lo<=a+1<hi]
        weights=[1.]*len(edges)
        if lo<=1<hi or lo<=260<hi:edges.append([1,260]);weights.append(np.inf)
        edges.append([int(lo),999]);weights.append(1.)  # another root
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'nodes':edges,'affinities':weights})
    s.client=SimpleNamespace(chunkedgraph=SimpleNamespace(default_url_mapping={},
        _endpoints={'get_subgraph':'https://example.test/{root_id}'},session=SimpleNamespace(get=get)))
    return s,calls


def test_complete_graph_preserves_seams_long_range_links_capacities_and_cache(tmp_path):
    s,calls=source(tmp_path)
    g=s.graph([1000]);assert len(g)==260 and g.number_of_edges()==260
    assert np.isinf(g[1][260]['capacity'])
    assert g[128][129]['capacity']==1 and 999 not in g
    count=len(calls);assert count>1
    assert s.graph([1000]).number_of_edges()==260 and len(calls)==count


def test_partial_graph_failure_is_not_published_and_retries_reuse_good_tiles(tmp_path):
    s,calls=source(tmp_path);original=s.client.chunkedgraph.session.get
    def fail(url,params,timeout):
        if params['bounds'].startswith('8330-'):raise TimeoutError()
        return original(url,params,timeout)
    s.client.chunkedgraph.session.get=fail
    # Fail explicitly after one good cached batch regardless of tree packing.
    import proofreading.source_graph as mod
    original_request=mod.request
    from unittest.mock import patch
    def failing(client,root,bounds):
        if bounds[0,0]>12000:raise ValueError('source offline')
        return original_request(client,root,bounds)
    with patch.object(mod,'request',failing):
        with pytest.raises(ValueError):s.graph([1000])
    s.client.chunkedgraph.session.get=original
    assert s.graph([1000]).number_of_edges()==260


def test_http_failure_keeps_status_without_secrets():
    class HTTPError(Exception):
        response=SimpleNamespace(status_code=403)
    def get(*a,**kw):raise HTTPError('secret-url-token')
    c=SimpleNamespace(default_url_mapping={},_endpoints={'get_subgraph':'https://x/{root_id}'},session=SimpleNamespace(get=get))
    with pytest.raises(ValueError,match='HTTP 403') as exc:request(c,1,np.array([[0,1]]*3))
    assert 'secret-url-token' not in str(exc.value)


def test_missing_cross_box_connections_cannot_be_used_for_a_split(tmp_path,monkeypatch):
    s,_=source(tmp_path)
    import proofreading.source_graph as mod
    original=mod.request
    def clipped(client,root,bounds):
        e,w=original(client,root,bounds)
        keep=~(((e[:,0]==130)&(e[:,1]==131))|((e[:,0]==1)&(e[:,1]==260)))
        return e[keep],w[keep]
    monkeypatch.setattr(mod,'request',clipped)
    with pytest.raises(ValueError,match='disconnected historical root'):
        s.graph([1000])
