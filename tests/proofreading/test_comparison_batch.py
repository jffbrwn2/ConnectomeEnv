import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
import pytest


@pytest.fixture
def batch(tmp_path,monkeypatch):
    (tmp_path/'.config/connectome').mkdir(parents=True)
    (tmp_path/'.config/connectome/site.json').write_text(json.dumps({'origin':'https://example.test','token':'test'}))
    (tmp_path/'orcd/pool/connectome-validation/agent-workbench-2026-10-08').mkdir(parents=True)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    path=Path(__file__).resolve().parents[2]/'scripts/proofreading/comparison_batch.py'
    spec=importlib.util.spec_from_file_location('comparison_batch_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_pending_worker_does_not_submit_duplicates(batch,monkeypatch):
    calls=[]
    def run(args,**kwargs):
        calls.append(args);return SimpleNamespace(stdout='PENDING\n')
    monkeypatch.setattr(batch.subprocess,'run',run)
    state={'worker_job':'new'}
    overview={'worker':{'connected':True,'job':'old'},'sessions':[]}
    assert not batch.ensure_worker(state,overview)
    assert len(calls)==1 and calls[0][0]=='squeue'


def test_no_rotation_while_session_active(batch,monkeypatch):
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**k:pytest.fail('must not rotate'))
    assert not batch.ensure_worker({}, {'worker':{'connected':True,'job':'old'},'sessions':[{'status':'running'}]})


def test_ready_worker_and_atomic_report(batch,monkeypatch):
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**k:pytest.fail('must not submit'))
    state={'worker_job':'new','status':'running','runs':[{'index':1,'model':'gpt-6-astra','id':'example','status':'complete','summary':{'actions':3}}]}
    assert batch.ensure_worker(state,{'worker':{'connected':True,'job':'new','expires_at':(time.time()+10000)*1000},'sessions':[]})
    batch.save(state);batch.summary(state)
    assert json.loads(batch.CHECKPOINT.read_text())==state
    assert json.loads((batch.ROOT/'report.json').read_text())['scored'] is False
