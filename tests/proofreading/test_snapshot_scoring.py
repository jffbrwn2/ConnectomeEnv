import json
from types import SimpleNamespace
import numpy as np
import pytest
from proofreading.compact_scoring import pack, read, score
from proofreading.evaluate import file_hash
from proofreading.snapshot_scoring import remap, prepare, evaluate_snapshot, timestamp


def fixture(tmp_path):
    atoms = np.array([1, 1, 2, 3, 4, 5], dtype=np.uint64)
    initial = {i: 10 for i in range(1, 6)}
    ref = np.array([20, 20, 20, 30, 30, 0], dtype=np.uint64)
    scope = np.array([True, True, True, True, True, False])
    path = tmp_path/'old.npz'
    pack(atoms, initial, ref, scope, path)
    spec = tmp_path/'old.json'
    spec.write_text(json.dumps({'schema':'streamed-frozen-v2','counts':path.name,
        'counts_sha256':file_hash(path),'dataset_sha256':'test','scope_note':'fixed'}))
    return read(path), initial, spec


def test_multiple_splits_and_false_split(tmp_path):
    data, initial, _ = fixture(tmp_path)
    # Reference now has THREE identities; original human edit had only two.
    new = remap(data, np.array([20, 20, 30, 40], np.uint64))
    perfect = {1:100, 2:100, 3:101, 4:102, 5:10}
    assert score(new, initial, perfect)['final']['vi_total_bits'] == 0
    single_split = {1:100, 2:100, 3:101, 4:101, 5:10}
    assert score(new, initial, single_split)['final']['vi_merge_bits'] > 0
    oversplit = {**perfect, 2:103}
    assert score(new, initial, oversplit)['final']['vi_split_bits'] > 0
    # Outside-scope material does not change reference agreement.
    assert score(new, initial, {**perfect, 5:100})['final']['vi_total_bits'] == 0


def test_unknown_and_validation(tmp_path):
    data, _, _ = fixture(tmp_path)
    mapped = remap(data, np.array([20, 20, 0, 40], np.uint64))
    assert int(mapped['unknown_voxels']) == 1
    assert sum(mapped['count']) == 4
    for bad in ([1,2], [1.,2.,3.,4.], [1,2,-1,4], [0,0,0,0]):
        with pytest.raises(ValueError): remap(data, np.array(bad))
    with pytest.raises(ValueError): timestamp('2026-09-28')


def test_pinned_lookup_rescore_and_tampering(tmp_path):
    data, initial, source = fixture(tmp_path)
    calls = []
    def get_roots(atoms, timestamp):
        calls.append(timestamp)
        return np.array([{1:20,2:20,3:30,4:40}[int(a)] for a in atoms], np.uint64)
    client = SimpleNamespace(datastack_name='test', chunkedgraph=SimpleNamespace(get_roots=get_roots))
    spec_path = tmp_path/'snapshot.json'
    prepare(source, spec_path, client, '2026-09-28T00:00:00+00:00', batch_size=2)
    assert len(calls)==2 and calls[0]==calls[1]
    ep=tmp_path/'ep';ep.mkdir()
    (ep/'session.json').write_text(json.dumps({'dataset_sha256':'test','created_at_unix':1}))
    rules=[{'parents':['10'],'children':['100','101','102'],'assignments':
        {'1':'100','2':'100','3':'101','4':'102','5':'102'}}]
    np.savez(ep/'partition.npz',format='streaming-rules-v1',dataset_sha256='test',rules=json.dumps(rules))
    result=evaluate_snapshot(ep,spec_path)
    assert result['final']['vi_total_bits']==0
    assert result['prepared_before_episode'] is False
    with pytest.raises(FileExistsError):prepare(source,spec_path,client,'2026-09-28T00:00:00+00:00')
    with (tmp_path/'snapshot.counts.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError, match='changed'):evaluate_snapshot(ep,spec_path)
