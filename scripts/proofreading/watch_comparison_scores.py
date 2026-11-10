"""Offline observer: score terminal batch episodes without invoking a model.

Runs independently of the coordinator/GPU worker. No reference information is
sent to agent tools. Failed scoring is recorded once and requires inspection.
"""
import argparse,fcntl,json,re,subprocess,sys,time
from pathlib import Path
from proofreading.evaluate import file_hash


def eligible(row):
    return row.get('status') in ('complete','error') and bool(re.fullmatch(r'[A-Za-z0-9_-]+',row.get('id','')))


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--watch-seconds',type=int,default=0)
    args=p.parse_args();b=args.base;out=args.output;out.mkdir(parents=True,exist_ok=True)
    checkpoint=b/'claude-astra-20-2026-10-08/checkpoint.json'
    reference=b/'evaluator-certified-reference-v1';started=time.monotonic()
    with (out/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=json.loads(checkpoint.read_text())
            for row in state['runs']:
                if not eligible(row):continue
                sid=row['id'];result_path=out/(sid+'.json');failure_path=out/(sid+'.failure.json')
                if result_path.exists() or failure_path.exists():continue
                episode=b/'portal-sessions'/sid/'episode'
                try:
                    harness=json.loads((episode/'harness.json').read_text())
                    fingerprint=file_hash(episode/'partition.npz')
                    raw=out/(sid+'.score.json')
                    command=[sys.executable,str(Path(__file__).with_name('score_indexed_sessions.py')),
                             str(b/'public-80um-explicit-em/dataset.json'),str(reference),str(reference/'baseline.npy'),
                             '--session',str(episode),'--output',str(raw)]
                    with (out/(sid+'.log')).open('w') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                    if fingerprint!=file_hash(episode/'partition.npz'):raise ValueError('Partition changed during scoring')
                    report=json.loads(raw.read_text())['reports'][0]
                    atomic_json(result_path,{'index':row['index'],'session_id':sid,'model':row['model'],
                        'portal_status_at_score':row['status'],'episode_status':harness['status'],
                        'actions':harness['actions'],'model_options':harness.get('model_options',{}),
                        'partition_sha256':fingerprint,'score':report})
                    print('SCORED',row['index'],sid,flush=True)
                except Exception as exc:
                    atomic_json(failure_path,{'session_id':sid,'error_type':type(exc).__name__,'message':str(exc)})
                    print('SCORING FAILED',sid,type(exc).__name__,flush=True)
            rows=[json.loads((out/(r['id']+'.json')).read_text()) for r in state['runs'] if r.get('id') and (out/(r['id']+'.json')).exists()]
            failures=[json.loads(x.read_text()) for x in out.glob('*.failure.json')]
            atomic_json(out/'report.json',{'batch_status':state['status'],'updated_at_unix':time.time(),
                'expected_runs':len(state['runs']),'scored_runs':len(rows),'failures':failures,'runs':rows,
                'interpretation':'Retrospective certified-identity diagnostics. Unknown material is not certified correct; whole-crop split VI is not crop-reachable recall.'})
            lines=['# Overnight certified-reference diagnostics','',
                   'These are retrospective checks, not a ranking of biological accuracy. Unknown material remains unresolved. Negative Δ wrong pairs means fewer errors.','',
                   '| Run | Model | Episode status | Actions | Δ wrong voxel pairs | Changed unknown voxels |',
                   '|---|---|---|---:|---:|---:|']
            for r in rows:
                s=r['score'];lines.append(f"| {r['index']} | {r['model']} | {r['episode_status']} | {r['actions']} | {-s['false_merge_pairs_removed']} | {s['changed_unknown_voxels']} |")
            lines+=['',f"Scored {len(rows)}/{len(state['runs'])}; scoring failures: {len(failures)}."]
            tmp=out/'report.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(out/'report.md')
            if state['status'] in ('complete','complete_with_errors','paused_provider_billing') or time.monotonic()-started>=args.watch_seconds:return
            time.sleep(30)


if __name__=='__main__':main()
