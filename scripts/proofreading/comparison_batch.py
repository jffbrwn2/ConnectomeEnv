"""Durable sequential Claude/Astra comparison; no model keys in its checkpoint.

Run on a Slurm CPU allocation. The launcher sets CONNECTOME_CODE_DIR and the
locked interpreter/dependency paths. Checkpoints live on the shared filesystem.
A lost/ambiguous submission pauses the batch instead of risking duplicate billing.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = Path.home()/'orcd/pool/connectome-validation/agent-workbench-2026-10-08'
BATCH_ID = 'claude-astra-20-2026-10-08'
ROOT = BASE/BATCH_ID
ROOT.mkdir(exist_ok=True)
CONFIG = json.loads((Path.home()/'.config/connectome/site.json').read_text())
CHECKPOINT = ROOT/'checkpoint.json'


def api(path, data=None):
    req = Request(CONFIG['origin'].rstrip('/')+path,
                  data=json.dumps(data).encode() if data is not None else None,
                  headers={'OAI-Sites-Authorization':'Bearer '+CONFIG['token'],
                           'Content-Type':'application/json'})
    for attempt in range(6):
        try:
            with urlopen(req, timeout=60) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError):
            if data is not None or attempt==5:
                raise
            time.sleep(min(30,2**attempt))


def save(state):
    tmp = CHECKPOINT.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(CHECKPOINT)


def summary(state):
    rows=[]
    for row in state['runs']:
        if row.get('id'):
            rows.append({**row,'url':CONFIG['origin'].rstrip('/')+'/#'+row['id']})
    (ROOT/'report.json').write_text(json.dumps({'batch_id':BATCH_ID,'status':state['status'],
        'scored':False,'note':'Repeated free-choice axon task with identical initial conditions; not biological accuracy scores.',
        'runs':rows},indent=2)+'\n')
    lines=['# Claude / Astra comparison', '', '200-action budget each; agents may finish early. Unscored.', '',
           '| Run | Model | Status | Actions | Turns | Commits |', '|---|---|---|---:|---:|---:|']
    for row in rows:
        s=row.get('summary') or {}
        lines.append(f"| [{row['index']}]({row['url']}) | {row['model']} | {row.get('status','queued')} | {s.get('actions','')} | {s.get('turns','')} | {s.get('commits','')} |")
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')


def ensure_worker(state, overview):
    w=overview['worker']
    own=state.get('worker_job')
    if own and (not w.get('connected') or w.get('job')!=own):
        status=subprocess.run(['squeue','-h','-j',own,'-o','%T'],capture_output=True,text=True,check=True).stdout.strip()
        if status:
            return False
    if w.get('connected') and w.get('job')==own and w.get('expires_at',0)/1000-time.time()>7200:
        return True
    # Rotation occurs only between runs and only when no session is active.
    if any(x['status'] in ('queued','running') for x in overview['sessions']):
        return False
    if w.get('connected'):
        subprocess.run(['scancel',w['job']],check=True)
    env=dict(os.environ,PORTAL_LIFETIME_SECONDS='21500')
    script=Path(os.environ['CONNECTOME_CODE_DIR'])/'scripts/proofreading/portal_gpu.sbatch'
    job=subprocess.run(['sbatch','--parsable','--time=06:00:00','--export=ALL',
        '--output='+str(BASE/'logs/portal-%j.log'),str(script)],env=env,check=True,capture_output=True,text=True).stdout.strip()
    state['worker_job']=job
    save(state)
    (BASE/'gpu-render-worker-job.json').write_text(json.dumps({'job':job,'code':os.environ['CONNECTOME_CODE_DIR']}))
    print('WORKER',job,flush=True)
    return False


def provider_blocker(row):
    if row.get('status') != 'error' or not row.get('id'):
        return None
    path = BASE/'portal-sessions'/row['id']/'episode/harness.json'
    if not path.exists():
        return None
    detail = json.loads(path.read_text()).get('provider_error', {})
    if detail.get('billing_action_required') or detail.get('code') in {
        'credit_balance_exhausted', 'insufficient_quota',
        'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
        'organization_usage_limit_exceeded',
    }:
        return detail
    return None


def main():
    with (ROOT/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if CHECKPOINT.exists():
            state=json.loads(CHECKPOINT.read_text())
        else:
            state={'batch_id':BATCH_ID,'status':'running','runs':[
                {'index':i+1,'model':'gpt-6-astra' if i%2==0 else 'claude-opus-5-5'} for i in range(20)]}
        if state.get('status') == 'paused_provider_billing':
            print('BATCH PAUSED: restore provider credits and explicitly clear the pause before resuming', flush=True)
            return
        state['status']='running'
        started=time.time()
        last_reconcile=0
        while True:
            # Publishing can be repaired without repeating a paid model episode.
            if time.time()-last_reconcile>60:
                for previous in state['runs']:
                    if previous.get('status')=='error' and previous.get('id'):
                        repaired=api('/api/sessions/'+previous['id'])
                        if repaired['status']=='complete':
                            previous.update(status='complete',summary=repaired.get('summary') or {})
                            snap=api('/api/sessions/'+previous['id']+'/snapshot')
                            (ROOT/f"run-{previous['index']:02d}-snapshot.json").write_text(json.dumps(snap))
                            previous['summary']['resolution_actions']=sum(e['tool']=='set_em_resolution' for e in snap.get('events',[]))
                            print('RECOVERED',previous['index'],flush=True)
                last_reconcile=time.time()
            for previous in state['runs']:
                blocker=provider_blocker(previous)
                if blocker:
                    state.update(status='paused_provider_billing', blocked_run=previous['index'], provider_error=blocker)
                    save(state);summary(state)
                    print('BATCH PAUSED: provider billing failure', previous['index'], flush=True)
                    return
            save(state);summary(state)
            # CPU partition permits 12h. Hand off checkpoint before its limit.
            if time.time()-started>10*3600:
                result=subprocess.run(['sbatch','--parsable','--dependency=afterany:'+os.environ['SLURM_JOB_ID'],
                    '--partition='+os.environ.get('SLURM_JOB_PARTITION','mit_normal'),'--export=ALL',str(Path(os.environ.get('CONNECTOME_COORDINATOR_DIR',os.environ['CONNECTOME_CODE_DIR']))/'scripts/proofreading/comparison_batch.sbatch')],
                    check=True,capture_output=True,text=True)
                state['successor_job']=result.stdout.strip();save(state);return
            row=next((r for r in state['runs'] if r.get('status') not in ('complete','error')),None)
            if row is None:
                state['status']='complete_with_errors' if any(r.get('status')=='error' for r in state['runs']) else 'complete';save(state);summary(state);print('BATCH TERMINAL',state['status'],flush=True);return
            if row.get('id'):
                worker=api('/api/overview')['worker']
                if not worker.get('connected') and state.get('worker_job'):
                    live=subprocess.run(['squeue','-h','-j',state['worker_job'],'-o','%T'],capture_output=True,text=True,check=True).stdout.strip()
                    if not live:
                        api('/api/worker/recover',{})
                result=api('/api/sessions/'+row['id'])
                row.update(status=result['status'],summary=result.get('summary') or {})
                if row['status'] in ('complete','error'):
                    try:
                        snap=api('/api/sessions/'+row['id']+'/snapshot')
                    except HTTPError as exc:
                        if exc.code!=404:raise
                        snap={'events':[],'missing_snapshot':True}
                    (ROOT/f"run-{row['index']:02d}-snapshot.json").write_text(json.dumps(snap))
                    row['summary']['resolution_actions']=sum(e['tool']=='set_em_resolution' for e in snap.get('events',[]))
                    print('DONE',row['index'],row['status'],flush=True)
                time.sleep(10);continue
            overview=api('/api/overview')
            if row.get('submission_started'):
                matches=[s for s in overview['sessions'] if s['config'].get('batch_id')==BATCH_ID and s['config'].get('batch_index')==row['index']]
                if len(matches)==1:
                    row['id']=matches[0]['id'];save(state);continue
                raise RuntimeError('Ambiguous submission; inspect batch record before resuming')
            if not ensure_worker(state,overview):
                time.sleep(10);continue
            row['submission_started']=time.time();save(state)
            result=api('/api/sessions',{'kind':'agent','dataset':'median-mouse-80um','model':row['model'],
                'max_actions':200,'batch_id':BATCH_ID,'batch_index':row['index']})
            row.update(id=result['id'],status='queued');save(state)
            print('RUN',row['index'],row['model'],row['id'],flush=True)
            time.sleep(10)


if __name__=='__main__':
    main()
