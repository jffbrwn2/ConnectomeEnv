import json,time
import numpy as np
import pytest
from proofreading.streamed_evaluate import project_rules,freeze_streamed,evaluate_streamed
from proofreading.evaluate import freeze
from proofreading.volume import Dataset
from test_streamed_dataset import make


def test_project_global_merge_and_complete_split_on_fixed_atoms():
    initial={1:10,3:20}
    merge={'parents':['10','20'],'children':['100'],'assignments':None}
    split={'parents':['100'],'children':['101','102'],'assignments':{'1':'101','2':'101','3':'102','4':'102'}}
    assert project_rules(initial,[merge])=={1:100,3:100}
    assert project_rules(initial,[merge,split])=={1:101,3:102}
    split['assignments'].pop('3')
    with pytest.raises(ValueError,match='omitted'):project_rules(initial,[merge,split])


def test_streamed_reference_keeps_denominator_fixed_and_detects_merge(tmp_path,monkeypatch):
    session,source=make(tmp_path)
    core=tmp_path/'core';core.mkdir()
    atoms=np.array([1,2,3,4],np.uint64).reshape(4,1,1)
    np.save(core/'atoms.npy',atoms);np.save(core/'em.npy',np.zeros_like(atoms,dtype=np.uint8))
    np.savez(core/'partition.npz',atoms=np.array([1,2,3,4],np.uint64),segments=np.array([101,101,102,102],np.uint64))
    config={'partition':'partition.npz',**{k:{'path':k+'.npy','resolution_nm':[8,8,40],'origin_nm':[0,0,0]} for k in ('atoms','em')}}
    manifest=core/'dataset.json';manifest.write_text(json.dumps(config));dense=Dataset.load(manifest)
    np.save(core/'reference.npy',np.array([101,101,102,102],np.uint64).reshape(4,1,1));np.save(core/'scope.npy',np.ones_like(atoms,dtype=bool))
    freeze(dense,core/'reference.npy',core/'scope.npy',core/'frozen.json')
    spec=tmp_path/'streamed-frozen.json'
    freeze_streamed(session.dataset,manifest,core/'frozen.json',spec)
    # Real execution constructs a session after freezing, as done here.
    from proofreading.session import Session
    episode=Session(session.dataset,tmp_path/'evaluated')
    before=evaluate_streamed(session.dataset,episode.output,spec)
    assert before['initial']['vi_total_bits']==0 and before['final']['scored_voxels']==4
    # The packaged evaluator must remain usable after every source artifact is gone.
    import shutil
    from proofreading.streamed_evaluate import evaluate_package
    assert json.loads(spec.read_text())['schema']=='streamed-frozen-v2'
    shutil.rmtree(core)
    def unavailable(*args,**kwargs):raise AssertionError('Scoring loaded the environment')
    monkeypatch.setattr(Dataset,'load',unavailable)
    assert evaluate_package(episode.output,spec)==before
    source.ensure([1,3]);episode.partition.apply_groups('merge',[101,102],[{1,3}],[2**63+1]);episode._save_partition()
    after=evaluate_streamed(session.dataset,episode.output,spec)
    assert after['final']['vi_total_bits']>0 and after['final']['scored_voxels']==4
    assert after['outside_reference_material']=='unscored'

    # Package identity, freeze timing and content integrity remain enforced.
    saved=json.loads(spec.read_text())
    saved['frozen_at_unix']=time.time()+100;spec.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='frozen'):evaluate_package(episode.output,spec)
    saved['frozen_at_unix']=0;spec.write_text(json.dumps(saved))
    counts=spec.parent/saved['counts'];counts.write_bytes(counts.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='counts changed'):evaluate_package(episode.output,spec)
