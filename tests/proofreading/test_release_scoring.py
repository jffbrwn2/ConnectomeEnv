import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_snapshot_scoring import fixture
from proofreading.snapshot_scoring import prepare
from proofreading.release_scoring import error_volume, evaluate_volume
from proofreading.evaluate import file_hash


def test_assignment_error_handles_renaming_and_wrong_joins(tmp_path):
    data, initial, _ = fixture(tmp_path)
    assert error_volume(data, initial) == (2, 5)
    assert error_volume(data, {1:90, 2:90, 3:91, 4:91, 5:99}) == (0, 5)
    assert error_volume(data, {1:90, 2:92, 3:91, 4:91, 5:99}) == (1, 5)


def test_volume_human_and_model_use_same_reference(tmp_path):
    _, _, source = fixture(tmp_path)
    client = SimpleNamespace(datastack_name='test', chunkedgraph=SimpleNamespace(
        get_roots=lambda atoms,timestamp: np.array([20,20,30,30],np.uint64)))
    ref=tmp_path/'snapshot.json'
    prepare(source,ref,client,'2026-09-28T00:00:00+00:00')
    human=tmp_path/'human.npz'
    np.savez(human,atoms=np.arange(1,6,dtype=np.uint64),segments=np.array([90,90,91,91,99],np.uint64))
    spec=json.loads(ref.read_text());spec.update(voxel_size_um3=.002,human_partition=human.name,human_partition_sha256=file_hash(human));ref.write_text(json.dumps(spec))
    ep=tmp_path/'ep';ep.mkdir()
    (ep/'session.json').write_text(json.dumps({'dataset_sha256':'test','created_at_unix':1}))
    rules=[{'parents':['10'],'children':['90','91'],'assignments':{'1':'90','2':'90','3':'91','4':'91','5':'91'}}]
    np.savez(ep/'partition.npz',format='streaming-rules-v1',dataset_sha256='test',rules=json.dumps(rules))
    v=evaluate_volume(ep,ref)['volume']
    assert v['initial_error_um3']==.004
    assert v['final_error_um3']==0 and v['human_error_um3']==0
    assert v['model_vs_human_improvement_percent']==100
    human.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Human partition changed'):evaluate_volume(ep,ref)


def test_full_cube_penalizes_wrong_addition_and_discloses_unknown(tmp_path):
    """Correct target merge plus a wrong neighbor cannot look like perfection."""
    from proofreading.compact_scoring import pack, read
    atoms = np.array([1]*4 + [2]*4 + [3]*6, np.uint64)
    initial = {1:10, 2:20, 3:30}
    counts = tmp_path/'cube.npz'
    pack(atoms, initial, np.where(atoms == 3, 200, 100).astype(np.uint64), atoms != 3, counts)
    source = tmp_path/'source.json'
    source.write_text(json.dumps({'schema':'streamed-frozen-v2', 'counts':counts.name,
        'counts_sha256':file_hash(counts), 'dataset_sha256':'test', 'scope_note':'target only'}))
    def client(unknown=False):
        return SimpleNamespace(datastack_name='test', chunkedgraph=SimpleNamespace(
            get_roots=lambda ids,timestamp:np.array([{1:100,2:100,3:0 if unknown else 200}[int(a)] for a in ids],np.uint64)))
    ref = tmp_path/'full.json'
    prepare(source, ref, client(), '2026-09-28T00:00:00+00:00', scope='full-cube')
    ep=tmp_path/'ep';ep.mkdir()
    (ep/'session.json').write_text(json.dumps({'dataset_sha256':'test','created_at_unix':1}))
    def score(parents):
        rules=[{'parents':parents,'children':['90'],'assignments':None}]
        np.savez(ep/'partition.npz',format='streaming-rules-v1',dataset_sha256='test',rules=json.dumps(rules))
        return evaluate_volume(ep,ref)['volume']
    good=score(['10','20'])
    assert good['initial_error_voxels']==4 and good['final_error_voxels']==0
    assert good['outcome']=='exact' and good['accounts_for_in_cube_additions']
    bad=score(['10','20','30'])
    assert bad['final_error_voxels']==6 and bad['outcome']=='worsened'
    assert bad['error_reduction_percent']==-50
    unknown=tmp_path/'unknown.json'
    prepare(source,unknown,client(True),'2026-09-28T00:00:00+00:00',scope='full-cube')
    v=evaluate_volume(ep,unknown)['volume']
    assert v['unknown_reference_voxels']==6 and v['outcome']=='unverified'
    assert v['known_scope_outcome']=='exact'
