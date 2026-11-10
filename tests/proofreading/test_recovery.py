import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from proofreading.experiment import execute
from proofreading.recovery import JournalTransport
from proofreading.resilient import supervise
from test_portable import prepared, finish_chat


def call(name, args):
    r = finish_chat()
    c = r['choices'][0]['message']['tool_calls'][0]
    c['id'] = name
    c['function'] = {'name': name, 'arguments': json.dumps(args)}
    return r


def test_resume_preserves_committed_edit_and_images(prepared, tmp_path, monkeypatch):
    # The first episode extracts meshes; the resumed one reuses the disk cache.
    monkeypatch.setenv('CONNECTOME_MESH_CACHE', str(tmp_path/'mesh-cache'))
    _, cfg, path = prepared
    cfg['budget']['actions'] = cfg['budget']['turns'] = 6
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv('MODEL_API_KEY', 'not-recorded-secret')
    calls = []
    def interrupted(endpoint, payload, headers):
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            return call('preview_merge', {'segments': ['11', '12']})
        if len(calls) == 2:
            text = next(x['content'] for x in reversed(payload['messages']) if x['role']=='tool')
            preview = json.loads(text)['result']['preview_id']
            return call('commit', {'preview_id': preview})
        raise TimeoutError('interruption')
    first = tmp_path/'first'
    with pytest.raises(TimeoutError):
        execute(path, first, transport=interrupted)
    assert (first/'open-world/partial-result.json').exists()
    seen = []
    def resumed(endpoint, payload, headers):
        seen.append(copy.deepcopy(payload))
        assert payload == calls[-1]
        return finish_chat()
    result = execute(path, tmp_path/'second', transport=resumed, resume_from=first)
    assert result[0]['status'] == 'finished'
    assert len(seen) == 1  # No paid replay of the first two model calls.
    ep = tmp_path/'second/open-world/episode'
    events = [json.loads(x) for x in (ep/'events.jsonl').read_text().splitlines()]
    assert sum(x['tool']=='commit' for x in events) == 1
    assert all('not-recorded-secret' not in p.read_text() for p in (ep/'exchanges').glob('*.json'))


def test_journal_recovers_response_before_tool_crash(tmp_path):
    source=tmp_path/'source';source.mkdir()
    j=JournalTransport(source, lambda *args: {'output': ['saved']})
    assert j('endpoint', {'input': ['image']}, {}) == {'output':['saved']}
    dest=tmp_path/'dest';dest.mkdir()
    recovered=JournalTransport(dest, lambda *a: pytest.fail('must not call provider'), source)
    assert recovered('endpoint', {'input':['image']}, {}) == {'output':['saved']}


def test_changed_observation_refuses_recovery(tmp_path):
    src=tmp_path/'src';src.mkdir();dst=tmp_path/'dst';dst.mkdir()
    JournalTransport(src, lambda *a: {'ok':True})('url', {'input':'original image'}, {})
    j=JournalTransport(dst, lambda *a:pytest.fail('must not call provider'), src)
    with pytest.raises(ValueError, match='mismatch'):
        j('url', {'input':'different image'}, {})


def test_automatic_transient_recovery_and_no_repeat_after_completion(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY','test')
    _,_,path=prepared
    responses=iter([TimeoutError(), finish_chat()])
    def send(*args):
        r=next(responses)
        if isinstance(r,Exception):raise r
        return r
    sleeps=[]
    out=tmp_path/'supervised'
    result=supervise(path,out,transport=send,sleep=sleeps.append)
    assert result[0]['status']=='finished' and sleeps==[30]
    with pytest.raises(ValueError,match='Existing attempts'):
        supervise(path,out,transport=send)
    execute(path,tmp_path/'completed-copy',resume_from=out/'attempt-002',
            transport=lambda *a:pytest.fail('completed case must not rebill'))


def test_hard_limit_pauses_without_repeated_attempts(prepared,tmp_path,monkeypatch):
    from urllib.error import HTTPError
    monkeypatch.setenv('MODEL_API_KEY','test')
    error=HTTPError('url',400,'Rejected',{},None)
    error.provider_detail={'reason':'image_count_limit','max_images':50,'retryable':False}
    def send(*a):raise error
    out=tmp_path/'supervised'
    with pytest.raises(HTTPError):
        supervise(prepared[2],out,transport=send,sleep=lambda *a:pytest.fail('no retry'))
    assert json.loads((out/'status.json').read_text())['reason']=='image_count_limit'
    assert len([p for p in out.glob('attempt-*') if p.is_dir()])==1


def test_crash_after_edit_recovers_saved_response_without_duplicate_commit(prepared,tmp_path,monkeypatch):
    from proofreading import Session
    _,cfg,path=prepared
    cfg['budget']['actions']=cfg['budget']['turns']=6
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv('MODEL_API_KEY','test')
    original=Session.execute
    def crash_after_commit(self,name,arguments=None,**kwargs):
        r=original(self,name,arguments,**kwargs)
        if name=='commit':raise RuntimeError('simulated crash after edit')
        return r
    def send(endpoint,payload,headers):
        calls=[m for m in payload['messages'] if m.get('role')=='tool']
        if not calls:return call('preview_merge',{'segments':['11','12']})
        preview=json.loads(calls[-1]['content'])['result']['preview_id']
        return call('commit',{'preview_id':preview})
    first=tmp_path/'crashed'
    with patch.object(Session,'execute',crash_after_commit),pytest.raises(RuntimeError):
        execute(path,first,transport=send)
    live=[]
    def finish(*a):live.append(1);return finish_chat()
    result=execute(path,tmp_path/'recovered',transport=finish,resume_from=first)
    assert len(live)==1
    assert result[0]['harness']['replayed_turns']==2
    assert len(result[0]['harness']['new_request_usage'])==1
    ep=tmp_path/'recovered/open-world/episode'
    events=[json.loads(x) for x in (ep/'events.jsonl').read_text().splitlines()]
    assert sum(e['tool']=='commit' for e in events)==1


def test_legacy_responses_import_reconstructs_exact_request_hashes(prepared,tmp_path,monkeypatch):
    import shutil
    from proofreading.recovery import import_responses_episode
    from proofreading.experiment import load_config
    _,cfg,path=prepared
    cfg['model']['provider']='responses'
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv('OPENAI_API_KEY','test')
    calls=[]
    def send(*a):
        calls.append(1)
        if len(calls)==1:
            return {'id':'r1','status':'completed','output':[{'type':'function_call','call_id':'c1','name':'look','arguments':'{}'}]}
        raise TimeoutError()
    out=tmp_path/'legacy'
    with pytest.raises(TimeoutError):execute(path,out,transport=send)
    ep=out/'open-world/episode'
    before=[json.loads(p.read_text())['request_sha256'] for p in sorted((ep/'exchanges').glob('*.json'))]
    shutil.rmtree(ep/'exchanges')
    assert import_responses_episode(ep,load_config(path))['saved_responses']==1
    after=[json.loads(p.read_text())['request_sha256'] for p in sorted((ep/'exchanges').glob('*.json'))]
    assert before==after


def test_existing_attempt_can_resume_after_scheduler_restart(prepared,tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY','test')
    out=tmp_path/'scheduler'
    def broken(*a):raise TimeoutError()
    with pytest.raises(TimeoutError):
        supervise(prepared[2],out,attempts=1,transport=broken)
    result=supervise(prepared[2],out,resume_latest=True,transport=lambda *a:finish_chat())
    assert result[0]['status']=='finished'
    assert (out/'attempt-002').is_dir()


def test_recovery_refuses_changed_configuration_before_inference(prepared,tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY','test')
    _,cfg,path=prepared
    first=tmp_path/'first'
    execute(path,first,transport=lambda *a:finish_chat())
    cfg['model']['name']='other-model'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='identical configuration'):
        execute(path,tmp_path/'changed',resume_from=first,transport=lambda *a:pytest.fail('must not call'))


def test_run_lock_prevents_two_writers(prepared,tmp_path):
    import fcntl
    output=tmp_path/'locked'
    with (tmp_path/'locked.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='already active'):
            execute(prepared[2],output,transport=lambda *a:pytest.fail('must not call'))


@pytest.mark.parametrize('name',['frames/test-scene.png','frames/scene-000025.png'])
@pytest.mark.parametrize('channel_delta,accepted',[(6,True),(100,False)])
def test_scene_roundoff_restores_original_bytes_but_rejects_real_changes(tmp_path,channel_delta,accepted,name):
    from proofreading.recovery import restore_recorded_observations
    from PIL import Image
    import numpy as np
    src=tmp_path/'src';dst=tmp_path/'dst'
    for p in [src,dst]:(p/'frames').mkdir(parents=True)
    a=np.zeros((400,400,3),dtype=np.uint8);b=a.copy();b[0,0,0]=channel_delta
    Image.fromarray(a).save(src/name);Image.fromarray(b).save(dst/name)
    response={'ok':True,'result':{'image':name}}
    event={'step':1,'tool':'look','arguments':{},'response':response}
    (src/'events.jsonl').write_text(json.dumps(event)+'\n')
    class Fake:
        serial=0;events=[];output=dst
        def execute(self,*args,**kwargs):
            self.serial+=1;self.events.append(copy.deepcopy(event));return copy.deepcopy(response)
    session=Fake();restore_recorded_observations(session,src)
    if accepted:
        session.execute('look',{})
        assert (src/name).read_bytes()==(dst/name).read_bytes()
    else:
        with pytest.raises(RuntimeError,match='image mismatch'):session.execute('look',{})

@pytest.mark.parametrize('change_state',[False,True])
@pytest.mark.parametrize('direct',[False,True])
def test_renderer_device_metadata_is_restored_but_state_is_checked(tmp_path,change_state,direct):
    from proofreading.recovery import restore_recorded_observations
    src=tmp_path/'src';dst=tmp_path/'dst';src.mkdir();dst.mkdir()
    original={'result':{'selected':['11'],'observation':{'scene_observation':{'rendering':{'device':'LLVM 17.0.2'}}}}}
    current=copy.deepcopy(original)
    current['result']['observation']['scene_observation']['rendering']['device']='LLVM 17.0.6'
    if direct:
        for response in (original,current):
            response['result']['rendering']=response['result'].pop('observation')['scene_observation']['rendering']
    if change_state:current['result']['selected']=['12']
    event={'step':1,'tool':'select','arguments':{},'response':original}
    (src/'events.jsonl').write_text(json.dumps(event)+'\n')
    class Fake:
        serial=0;output=dst
        def __init__(self):self.events=[]
        def execute(self,*a,**kw):
            self.serial+=1;self.events.append({**event,'response':current});return current
    session=Fake();restore_recorded_observations(session,src)
    if change_state:
        with pytest.raises(RuntimeError,match='observation mismatch'):session.execute('select',{})
    else:assert session.execute('select',{})==original

@pytest.mark.parametrize('changed_image',[False,True])
def test_legacy_mesh_timestamp_recovery_checks_image_bytes(tmp_path,changed_image):
    from proofreading.recovery import restore_recorded_observations
    src=tmp_path/'src';dst=tmp_path/'dst'
    for p in (src,dst):(p/'frames').mkdir(parents=True)
    old='frames/mesh-000001-123.png';new='frames/mesh-000001-456.png'
    from PIL import Image
    Image.new('RGB',(2,2),'red').save(src/old)
    Image.new('RGB',(2,2),'blue' if changed_image else 'red').save(dst/new)
    event={'step':1,'tool':'mesh','arguments':{},'response':{'image':old}}
    (src/'events.jsonl').write_text(json.dumps(event)+'\n')
    class Fake:
        serial=0;output=dst
        def __init__(self):self.events=[]
        def execute(self,*a,**kw):
            self.serial+=1;self.events.append({**event,'response':{'image':new}});return {'image':new}
    session=Fake();restore_recorded_observations(session,src)
    if changed_image:
        with pytest.raises(RuntimeError,match='image mismatch'):session.execute('mesh',{})
    else:
        assert session.execute('mesh',{})=={'image':old}
        assert (dst/old).read_bytes()==(src/old).read_bytes()


def test_storage_failures_pause_without_retry():
    import errno
    from proofreading.recovery import failure_state
    for code in (errno.ENOSPC, errno.EDQUOT):
        result=failure_state(OSError(code,'storage exhausted'))
        assert result['reason']=='storage_exhausted' and not result['retryable']
