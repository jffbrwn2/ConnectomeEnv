import importlib.util
import json
from pathlib import Path


def load_batch(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    base=tmp_path/'orcd/pool/connectome-validation/agent-workbench-2026-10-08'
    base.mkdir(parents=True)
    config=tmp_path/'.config/connectome/site.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'origin':'https://example.invalid'}))
    path=Path(__file__).resolve().parents[2]/'scripts/proofreading/comparison_batch.py'
    spec=importlib.util.spec_from_file_location('isolated_batch',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_billing_failure_stops_before_submission(tmp_path, monkeypatch):
    m=load_batch(tmp_path, monkeypatch)
    episode=m.BASE/'portal-sessions/example/episode'
    episode.mkdir(parents=True)
    (episode/'harness.json').write_text(json.dumps({'provider_error':{'code':'credit_balance_exhausted'}}))
    m.save({'status':'running','runs':[{'index':1,'model':'gpt-6-astra','id':'example','status':'error'},{'index':2,'model':'claude-opus-5-5'}]})
    calls=[]
    def api(path, data=None):
        calls.append((path,data))
        assert data is None, 'No submission allowed after a billing failure'
        return {'status':'error'}
    monkeypatch.setattr(m,'api',api)
    m.main()
    state=json.loads(m.CHECKPOINT.read_text())
    assert state['status']=='paused_provider_billing'
    assert state['blocked_run']==1
    assert 'id' not in state['runs'][1]


def test_existing_pause_does_not_call_provider_or_portal(tmp_path, monkeypatch):
    m=load_batch(tmp_path, monkeypatch)
    m.save({'status':'paused_provider_billing','runs':[]})
    def api(*args,**kwargs):
        raise AssertionError('Paused batch must not make requests')
    monkeypatch.setattr(m,'api',api)
    m.main()
    assert json.loads(m.CHECKPOINT.read_text())['status']=='paused_provider_billing'
