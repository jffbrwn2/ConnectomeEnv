"""Portable contracts with synthetic data and fake providers; no paid requests."""
import copy
import json
import threading
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import numpy as np
import pytest
from proofreading import Dataset, Session
from proofreading.demo import create
from proofreading.evaluate import freeze
from proofreading.harness import run
from proofreading.experiment import execute, load_config
from proofreading.results_server import make_server
from proofreading.snapshots import snapshot


@pytest.fixture
def prepared(tmp_path):
    data = create(tmp_path / 'data')
    ds = Dataset.load(data)
    spec = tmp_path / 'spec.json'
    freeze(ds, data.parent / 'evaluation/reference.npy', data.parent / 'evaluation/scope.npy', spec)
    cfg = {'schema': 1, 'mode': 'open-world', 'dataset': str(data), 'seed_segment': '11',
           'view': {'width': 32, 'height': 32},
           'scoring': {'kind': 'frozen', 'spec': str(spec)},
           'model': {'provider': 'chat', 'name': 'synthetic-test', 'endpoint': 'http://localhost:9999/v1/chat/completions'},
           'budget': {'actions': 3, 'turns': 3, 'output_tokens': 100}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    return data, cfg, path


def finish_chat():
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': 'Done.',
            'tool_calls': [{'id': 'finish-1', 'type': 'function', 'function': {'name': 'finish', 'arguments': '{"summary":"Synthetic check"}'}}]}}], 'usage': {'prompt_tokens': 1}}


@pytest.mark.parametrize('provider', ['chat', 'openrouter'])
def test_chat_history_images_and_reasoning(prepared, tmp_path, provider):
    data, _, _ = prepared
    session = Session(Dataset.load(data), tmp_path / provider)
    session.execute('set_view', {'width': 32, 'height': 32})
    seen = []
    def transport(endpoint, payload, headers):
        seen.append(copy.deepcopy(payload))
        if len(seen) == 1:
            return {'choices': [{'message': {'role': 'assistant', 'content': 'Inspecting.',
                     'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'private-test'}],
                     'tool_calls': [{'id': 'look-1', 'type': 'function', 'function': {'name': 'look', 'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}]}
        return finish_chat()
    result = run(session, 'test-model', provider=provider, endpoint='http://localhost:9999/v1/chat/completions', api_key='test-secret', transport=transport)
    assert result['actions'] == 2 and result['status'] == 'finished'
    if provider == 'openrouter':
        assert 'parallel_tool_calls' not in seen[0]
    else:
        assert seen[0]['parallel_tool_calls'] is False
    assert seen[0]['messages'][1]['content'][1]['type'] == 'image_url'
    messages = seen[1]['messages']
    assert any(m.get('reasoning_details') for m in messages)
    assert messages[-2]['role'] == 'tool' and messages[-2]['tool_call_id'] == 'look-1'
    assert messages[-1]['role'] == 'user' and messages[-1]['content'][1]['type'] == 'image_url'
    snap, _ = snapshot(session.output)
    assert snap['frames'][1]['text'] == 'Inspecting.'
    assert 'private-test' not in json.dumps(snap)
    assert 'test-secret' not in (session.output / 'harness.json').read_text()


def test_openrouter_multiple_calls_before_image_observations(prepared, tmp_path):
    data, _, _ = prepared
    session = Session(Dataset.load(data), tmp_path / 'multiple-calls')
    session.execute('set_view', {'width': 32, 'height': 32})
    seen = []
    def transport(endpoint, payload, headers):
        seen.append(copy.deepcopy(payload))
        if len(seen) == 1:
            return {'choices': [{'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': 'Inspect two views.',
                'tool_calls': [{'id': f'look-{i}', 'type': 'function',
                    'function': {'name': 'look', 'arguments': '{}'}} for i in range(2)]}}]}
        return finish_chat()
    result = run(session, 'test-model', provider='openrouter', api_key='test-secret',
                 transport=transport, max_actions=3)
    assert result['status'] == 'finished' and result['actions'] == 3
    following = seen[1]['messages'][3:]
    assert [m['role'] for m in following] == ['tool', 'tool', 'user', 'user']
    assert [m['tool_call_id'] for m in following[:2]] == ['look-0', 'look-1']
    assert all(m['content'][1]['type'] == 'image_url' for m in following[2:])


def test_open_world_run_and_local_viewer(prepared, tmp_path, monkeypatch):
    _, _, cfg = prepared
    monkeypatch.setenv('MODEL_API_KEY', 'offline-test')
    result = execute(cfg, tmp_path / 'run', transport=lambda *args: finish_chat())
    assert result[0]['status'] == 'finished' and 'frozen' in result[0]['scores']
    with pytest.raises(FileExistsError):
        execute(cfg, tmp_path / 'run')
    server = make_server(tmp_path / 'run', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        rows = json.load(urlopen(base + '/api/episodes'))
        assert len(rows) == 1
        snap = json.load(urlopen(base + '/episodes/0/snapshot'))
        assert len(snap['frames']) == 2
        img = snap['frames'][0]['em']
        assert urlopen(base + '/episodes/0/' + img).read().startswith(b'\x89PNG')
        for url in ['/episodes/0/frames/../../config.json', '/episodes/0/model-responses.jsonl']:
            with pytest.raises(HTTPError):urlopen(base + url)
        with pytest.raises(HTTPError) as e:
            urlopen(Request(base + '/', headers={'Host': 'attacker.example'}))
        assert e.value.code == 403
    finally:
        server.shutdown();server.server_close();thread.join()


def test_preflight_never_calls_provider(prepared, tmp_path):
    _, _, cfg = prepared
    def forbidden(*args):
        raise AssertionError('Paid call during preflight')
    result = execute(cfg, tmp_path / 'preflight', preflight=True, transport=forbidden)
    assert result[0]['model_calls'] == 0


def test_corrupt_reference_rejected_before_inference(prepared, tmp_path):
    data, _, cfg = prepared
    np.save(data.parent / 'evaluation/reference.npy', np.zeros((3, 3, 3), np.uint64))
    with pytest.raises(ValueError):
        execute(cfg, tmp_path / 'bad', transport=lambda *a: pytest.fail('Inference before validation'))
    assert (tmp_path / 'bad/open-world/failure.json').exists()


def test_cb2_contract_and_split(prepared, tmp_path, monkeypatch):
    data, cfg, path = prepared
    import shutil
    benchmark = tmp_path / 'benchmark';case = benchmark / 'synthetic-before'
    shutil.copytree(data.parent, case / 'public')
    ds = Dataset.load(case / 'public/dataset.json')
    (case / 'evaluator').mkdir()
    for metric in ('human','certified'):
        freeze(ds, case / 'public/evaluation/reference.npy', case / 'public/evaluation/scope.npy', case / 'evaluator' / (metric+'-frozen.json'))
    (case / 'public/task.json').write_text(json.dumps({'seed_segment':'11','center_nm':[2920,3920,5480],'extent_nm':4000}))
    (benchmark / 'index.json').write_text(json.dumps([{'case_id':'synthetic-before','split':'train','dataset_sha256':ds.fingerprint}]))
    cfg = {k:cfg[k] for k in ['schema','model','budget']}
    cfg.update(mode='cb2',benchmark=str(benchmark),phase='train',cases=['synthetic-before'])
    path.write_text(json.dumps(cfg));monkeypatch.setenv('MODEL_API_KEY','offline')
    result = execute(path,tmp_path/'cb2-run',transport=lambda *a: finish_chat())
    assert set(result[0]['scores']) == {'human','certified'}
    cfg['phase']='test';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='split'):
        execute(path,tmp_path/'wrong-split')
    assert not (tmp_path/'wrong-split').exists()


def test_config_rejects_secrets_and_missing_scoring(prepared):
    _, cfg, path = prepared
    cfg['model']['api_key']='nope';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='credentials'):load_config(path)
    del cfg['model']['api_key'];del cfg['scoring'];path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='scoring'):load_config(path)


def test_relocation_keeps_frozen_hash_and_rejects_tampering(prepared, tmp_path, monkeypatch):
    import shutil
    from proofreading.evaluate import evaluate_frozen, file_hash
    data, cfg, path = prepared
    spec = Path(cfg['scoring']['spec'])
    original_hash = file_hash(spec)
    moved = tmp_path / 'relocated'
    shutil.move(str(data.parent), moved)
    monkeypatch.setenv('CONNECTOME_PATH_MAP', json.dumps({str(data.parent): str(moved)}))
    ds = Dataset.load(moved / 'dataset.json')
    session = Session(ds, tmp_path / 'relocation-session')
    result = evaluate_frozen(ds, session.output, spec)
    assert result['frozen_spec_sha256'] == original_hash
    np.save(moved / 'evaluation/reference.npy', np.zeros((3,3,3),np.uint64))
    with pytest.raises(ValueError, match='changed'):
        evaluate_frozen(ds, session.output, spec)

from test_indexed import indexed


def test_indexed_open_world_uses_frozen_anchor(indexed, tmp_path, monkeypatch):
    import time
    from proofreading.evaluate import file_hash
    from proofreading.indexed_evaluate import CertifiedScorer
    from proofreading.transfer_evaluate import freeze_anchor
    d, atoms, roots = indexed
    reference = tmp_path/'certified';reference.mkdir()
    spec={'geometry_sha256':d.geometry_fingerprint,'arrays':{}}
    for name,data in [('atoms',atoms[1:].astype(np.uint64)),('reference',np.array([11,11,12],np.uint64)),
                      ('voxel_counts',d.atom_cnt[d.to_dense(atoms[1:])].astype(np.uint64))]:
        target=reference/(name+'.npy');np.save(target,data)
        spec['arrays'][name]={'file':target.name,'sha256':file_hash(target)}
    (reference/'reference-spec.json').write_text(json.dumps(spec))
    scorer=CertifiedScorer(d,reference);scorer.build_baseline(reference/'baseline.npy')
    scope=tmp_path/'scope';scope.mkdir()
    for name in ['seed','reachable']:np.save(scope/(name+'-atoms.npy'),atoms[1:3])
    contract={'dataset_sha256':d.fingerprint,'reference_spec_sha256':scorer.spec_hash,
              'seed_atoms_sha256':file_hash(scope/'seed-atoms.npy'),'reachable_atoms_sha256':file_hash(scope/'reachable-atoms.npy'),
              'seed_segment':str(roots[0]),'evaluator_target_neuron':'11','frozen_at_unix':time.time(),'scope':'synthetic reachable scope'}
    (scope/'frozen.json').write_text(json.dumps(contract));freeze_anchor(d,scorer,scope)
    cfg={'schema':1,'mode':'open-world','dataset':str(d.manifest),'seed_segment':str(roots[0]),
         'scoring':{'kind':'transfer','reference':str(reference),'scope':str(scope)},
         'model':{'provider':'chat','name':'fake','endpoint':'http://localhost:1/v1/chat/completions'},
         'view':{'width':32,'height':32},'budget':{'actions':1,'turns':1,'output_tokens':100}}
    path=tmp_path/'transfer.json';path.write_text(json.dumps(cfg));monkeypatch.setenv('MODEL_API_KEY','offline')
    result=execute(path,tmp_path/'transfer-run',transport=lambda *a:finish_chat())
    score=result[0]['scores']['transfer']
    assert score['frozen_before_episode'] and not score['full_neuron_recall']
    assert score['evidence']['recall']['final']==1


def test_cb2_setup_applies_requested_resolution_and_preserves_task_defaults(tmp_path):
    from proofreading.experiment import setup
    public=tmp_path/'public';public.mkdir()
    (public/'task.json').write_text(json.dumps({'seed_segment':'11','center_nm':[1,2,3],'extent_nm':6000}))
    calls=[]
    class Recorder:
        def execute(self,name,args):
            calls.append((name,args));return {'ok':True}
    setup({'mode':'cb2','view':{'em_resolution_nm':8,'pixel_size_nm':8,'width':768,'height':768}}, {'directory':str(tmp_path)}, Recorder())
    view=dict(calls)['set_view']
    assert view=={'center_nm':[1,2,3],'pixel_size_nm':8,'width':768,'height':768}
    assert dict(calls)['set_em_resolution']=={'xy_nm':8}
    from proofreading.tools import schemas
    from jsonschema import validate
    by_name={s['name']:s['parameters'] for s in schemas()}
    for name,args in calls:validate(args,by_name[name])
    assert dict(calls)['select']['segments']==['11']


def test_early_model_stop_is_scored_and_flagged(prepared,tmp_path,monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "offline-test")
    _,_,config=prepared
    response={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':''}}]}
    result=execute(config,tmp_path/'early',transport=lambda *args:response)[0]
    assert result['status']=='model_stopped_without_finish'
    assert result['quality_flags']==['early_model_stop'] and result['scores']


def test_storage_failure_preserves_original_exception(prepared,tmp_path,monkeypatch,capsys):
    import errno
    monkeypatch.setenv("MODEL_API_KEY", "offline-test")
    import proofreading.experiment as experiment
    original=experiment.write
    def write(path,value):
        if path.name in ('failure.json','status.json','partial-score-failure.json') and value.get('status')=='paused':
            raise OSError(errno.EDQUOT,'cannot record failure')
        return original(path,value)
    monkeypatch.setattr(experiment,'write',write)
    def transport(*args):
        raise OSError(errno.EDQUOT,'quota exceeded')
    with pytest.raises(OSError) as caught:
        execute(prepared[2],tmp_path/'quota',transport=transport)
    assert caught.value.errno==errno.EDQUOT
    assert 'storage_exhausted' in capsys.readouterr().err
