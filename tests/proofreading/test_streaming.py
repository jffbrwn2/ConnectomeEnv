from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from proofreading.streaming import ChunkCache,StreamingGrid
from proofreading.volume import Grid


def setup(cache=None,identity='pinned-source'):
    data=np.arange(19*17*9,dtype=np.uint64).reshape(19,17,9)+2**54
    calls=[];offset=np.array([11,13,7]);resolution=[8,8,40]
    def load(lo,hi):
        calls.append((lo.copy(),hi.copy()))
        return data[tuple(slice(a,b) for a,b in zip(lo-offset,hi-offset))].copy()
    stream=StreamingGrid(identity,resolution,offset,data.shape,data.dtype,load,cache,chunk_shape=(4,4,4))
    dense=Grid(data,resolution,offset*resolution)
    return stream,dense,calls


def test_sample_far_away_then_return_only_downloads_intersecting_chunks():
    cache=ChunkCache(max_bytes=4096);s,d,calls=setup(cache)
    p=d.origin+[4,4,20];q=d.upper-[4,4,20]
    assert s.nearest(p)[0]==d.nearest(p)[0]
    assert len(calls)==1
    assert s.nearest(q)[0]==d.nearest(q)[0]
    assert len(calls)==2
    s.nearest(p)
    assert len(calls)==2
    assert cache.resident_bytes<=cache.max_bytes


def test_oblique_interpolation_and_native_uint64_labels():
    s,d,calls=setup()
    rng=np.random.default_rng(4)
    xyz=d.origin+rng.random((12,15,3))*(d.upper-d.origin)
    actual,valid=s.nearest(xyz);expected,ok=d.nearest(xyz)
    assert np.array_equal(actual,expected) and np.array_equal(valid,ok)
    actual,valid=s.linear(xyz);expected,ok=d.linear(xyz)
    assert np.allclose(actual,expected,rtol=1e-15)
    outside=d.upper+[1,1,1]
    assert not s.nearest(outside)[1] and not s.linear(outside)[1]
    assert all(np.prod(hi-lo)*8<=32*1024**2 for lo,hi in calls)


def test_eviction_disk_reuse_and_source_isolation(tmp_path):
    cache=ChunkCache(max_bytes=512,directory=tmp_path,disk_bytes=4096)
    s,d,calls=setup(cache)
    p=d.origin+[4,4,20];q=d.upper-[4,4,20]
    s.nearest(p);s.nearest(q);s.nearest(p)
    assert len(calls)==2 and cache.disk_hits==1
    assert cache.resident_bytes<=512
    other,_,other_calls=setup(cache,'different-snapshot')
    other.nearest(p)
    assert len(other_calls)==1
    assert sum(p.stat().st_size for p in tmp_path.glob('*.npy'))<=4096


def test_concurrent_requests_fetch_once_and_errors_never_become_background():
    cache=ChunkCache(max_bytes=512);s,d,calls=setup(cache)
    p=d.origin+[4,4,20]
    with ThreadPoolExecutor(4) as executor:results=list(executor.map(s.nearest,[p]*8))
    assert len(calls)==1 and all(r[0]==results[0][0] for r in results)
    bad=StreamingGrid('broken',[1,1,1],[0,0,0],[4,4,4],'uint64',lambda a,b:np.zeros((1,1,1),np.uint64),chunk_shape=(4,4,4))
    with pytest.raises(ValueError,match='shape/dtype'):bad.nearest([1,1,1])


def test_independent_cache_misses_overlap_and_warm_reads_do_not_wait():
    import threading
    cache=ChunkCache(max_bytes=128)
    gate=threading.Barrier(2)
    def load():
        gate.wait(timeout=3)
        return np.ones(4,np.uint64)
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(cache.get,'a',(4,),np.uint64,load)
        b=pool.submit(cache.get,'b',(4,),np.uint64,load)
        assert np.array_equal(a.result(timeout=5),b.result(timeout=5))
    assert cache.misses==2 and cache.resident_bytes==64
    parent=np.ones(16,np.uint64)
    held=cache.get('slice',(4,),np.uint64,lambda:parent[:4])
    assert not np.shares_memory(parent,held)


def test_duplicate_failed_cache_fetches_share_error_then_retry():
    import threading
    cache=ChunkCache(max_bytes=128);entered=threading.Event();release=threading.Event();calls=[]
    def fail():
        calls.append(1);entered.set();release.wait(3);raise ValueError('unavailable')
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(cache.get,'a',(4,),np.uint64,fail)
        assert entered.wait(2)
        second=pool.submit(cache.get,'a',(4,),np.uint64,fail)
        release.set()
        for f in (first,second):
            with pytest.raises(ValueError,match='unavailable'):f.result(timeout=5)
    assert not cache.inflight and cache.resident_bytes==0
    assert np.array_equal(cache.get('a',(4,),np.uint64,lambda:np.ones(4,np.uint64)),np.ones(4,np.uint64))


def test_cache_identity_includes_chunk_geometry():
    cache=ChunkCache(max_bytes=4096)
    a,_,_=setup(cache)
    b=StreamingGrid('pinned-source',[8,8,40],[11,13,7],[19,17,9],np.uint64,
                    a.loader,cache,chunk_shape=(4,4,2))
    assert a.identity!=b.identity
    p=a.origin+np.array([4,4,260])
    assert a.nearest(p)[0]==b.nearest(p)[0]


@pytest.mark.parametrize('method,max_calls', [('nearest', 16), ('linear', 100)])
def test_wide_view_batches_fit_cache_and_preserve_samples(method, max_calls):
    data=np.arange(64*8*64,dtype=np.uint64).reshape(64,8,64)
    cache=ChunkCache(max_bytes=16384)
    def load(lo,hi):
        assert np.prod(hi-lo)*data.dtype.itemsize<=cache.max_bytes
        return data[tuple(slice(a,b) for a,b in zip(lo,hi))].copy()
    stream=StreamingGrid('wide',[1,1,1],[0,0,0],data.shape,data.dtype,load,
                         cache,chunk_shape=(4,4,4))
    dense=Grid(data,[1,1,1],[0,0,0])
    x,z=np.meshgrid(np.arange(64)+.3,np.arange(64)+.6,indexing='ij')
    xyz=np.stack([x,np.full_like(x,3.4),z],-1)
    actual,valid=getattr(stream,method)(xyz)
    expected,ok=getattr(dense,method)(xyz)
    assert np.array_equal(valid,ok)
    assert np.allclose(actual,expected,rtol=0,atol=1e-10)
    assert stream.source_calls<=max_calls  # Previously one request per chunk: 256.
    assert cache.resident_bytes<=cache.max_bytes


@pytest.mark.parametrize('method', ['nearest', 'linear'])
def test_small_cache_oblique_samples_keep_original_order(method):
    cache=ChunkCache(max_bytes=2048)
    stream,dense,_=setup(cache)
    rng=np.random.default_rng(7)
    xyz=dense.origin+rng.random((23,29,3))*(dense.upper-dense.origin)
    actual,valid=getattr(stream,method)(xyz)
    expected,ok=getattr(dense,method)(xyz)
    assert np.array_equal(valid,ok)
    assert np.allclose(actual,expected,rtol=1e-15)
    assert cache.resident_bytes<=cache.max_bytes
