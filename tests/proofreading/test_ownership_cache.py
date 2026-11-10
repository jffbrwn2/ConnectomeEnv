import numpy as np
import pytest
from proofreading.ownership_cache import OwnershipCache


def test_batched_persistent_bounded_ownership_and_snapshot_isolation(tmp_path):
    calls=[]
    def resolve(atoms):calls.append(atoms.copy());return atoms+100
    c=OwnershipCache({'timestamp':'t1'},resolve,tmp_path,memory_entries=2,disk_entries=4)
    ids=np.array([2**63+1,2**63+2,2**63+3],np.uint64)
    assert c.ensure(ids)==dict(zip(map(int,ids),map(int,ids+100)))
    assert len(calls)==1 and len(c.memory)==2
    assert c.owner(int(ids[0]))==int(ids[0])+100 and len(calls)==1
    c.close()
    c=OwnershipCache({'timestamp':'t1'},resolve,tmp_path,memory_entries=2,disk_entries=4)
    c.ensure(ids);assert len(calls)==1
    c.ensure([4,5,6,7]);assert c.stats()['disk_entries']<=4
    c.close()
    other=OwnershipCache({'timestamp':'t2'},resolve,tmp_path)
    other.ensure(ids);assert len(calls)==3
    other.close()


def test_missing_ownership_is_not_cached(tmp_path):
    c=OwnershipCache('pinned',lambda a:np.zeros(len(a),np.uint64),tmp_path)
    with pytest.raises(ValueError,match='no guessed'):c.ensure([1,2])
    assert not c.memory and c.stats()['disk_entries']==0
    c.close()


def test_parallel_batches_settle_and_failure_is_atomic(tmp_path):
    import threading
    barrier=threading.Barrier(2,timeout=5)
    finished=[]
    fail=True
    def resolve(atoms):
        barrier.wait()
        finished.append(int(atoms[0]))
        if fail and atoms[0]==1:raise ValueError('failed batch')
        return atoms+100
    c=OwnershipCache('pinned',resolve,tmp_path,resolver_workers=2)
    ids=np.arange(1,20001,dtype=np.uint64)
    with pytest.raises(ValueError,match='failed batch'):c.ensure(ids)
    assert len(finished)==2
    assert not c.memory and c.stats()['disk_entries']==0
    fail=False
    result=c.ensure(ids)
    assert result==dict(zip(map(int,ids),map(int,ids+100)))
    assert c.stats()['resolved_atoms']==20000
    c.close()


def test_display_can_record_missing_owner_but_edits_remain_strict(tmp_path):
    from proofreading.ownership_cache import MissingOwnership
    c=OwnershipCache('pinned',lambda a:np.where(a==2,0,101).astype(np.uint64),tmp_path)
    assert c.observe([1,2])=={1:101,2:0}
    assert c.owner(1)==101
    with pytest.raises(MissingOwnership):c.owner(2)
    with pytest.raises(MissingOwnership):c.ensure([1,2])
    assert c.stats()['resolver_requests']==1
    c.close()


def test_local_ownership_isolated_from_shared_database(tmp_path,monkeypatch):
    shared=tmp_path/'shared';local=tmp_path/'local'
    first=OwnershipCache('t1',lambda a:a+100,shared)
    assert first.owner(1)==101
    first.db.execute('BEGIN EXCLUSIVE')
    monkeypatch.setenv('CONNECTOME_LOCAL_OWNERSHIP',str(local))
    second=OwnershipCache('t1',lambda a:a+100,shared)
    assert second.owner(1)==101
    assert list(local.rglob('*.sqlite'))
    second.close();first.db.rollback();first.close()
    third=OwnershipCache('t1',lambda a:pytest.fail('local cache should persist'),shared)
    assert third.owner(1)==101
    third.close()
