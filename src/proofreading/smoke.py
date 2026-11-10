"""Offline end-to-end smoke: synthetic data and scripted tool calls, never a model."""
import argparse
import json
from pathlib import Path
from .demo import create
from .evaluate import freeze
from .volume import Dataset
from .experiment import execute


def create_and_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data = create(root / 'benchmark/synthetic-before/public')
    ds = Dataset.load(data)
    evaluator = data.parent.parent / 'evaluator';evaluator.mkdir()
    for metric in ('human', 'certified'):
        freeze(ds, data.parent / 'evaluation/reference.npy', data.parent / 'evaluation/scope.npy', evaluator / (metric + '-frozen.json'))
    (data.parent / 'task.json').write_text(json.dumps({'seed_segment': '11', 'center_nm': [2920,3920,5480], 'extent_nm': 4000}))
    benchmark = data.parent.parent.parent
    (benchmark / 'index.json').write_text(json.dumps([{'case_id': 'synthetic-before', 'split': 'train', 'dataset_sha256': ds.fingerprint}]))
    common = {'schema':1, 'model': {'provider':'chat', 'name':'SCRIPTED-SYNTHETIC-NOT-A-MODEL',
              'endpoint':'http://localhost:1/v1/chat/completions','api_key_env':'CONNECTOME_SMOKE_KEY'},
              'budget': {'actions':3,'turns':3,'output_tokens':100}}
    configs = {
        'cb2': {**common, 'mode':'cb2','benchmark':str(benchmark.resolve()), 'phase':'train','cases':['synthetic-before']},
        'open-world': {**common, 'mode':'open-world','dataset':str(data.resolve()), 'seed_segment':'11',
                       'view':{'width':64,'height':64},
                       'scoring':{'kind':'frozen','spec':str((evaluator/'human-frozen.json').resolve())}}}
    import os
    # This is a dummy credential consumed by an in-process fake transport only.
    previous = os.environ.get('CONNECTOME_SMOKE_KEY')
    os.environ['CONNECTOME_SMOKE_KEY'] = 'offline-only'
    results = {}
    try:
        for mode, cfg in configs.items():
            config = root / (mode + '.json');config.write_text(json.dumps(cfg,indent=2))
            count = 0
            def scripted(endpoint, payload, headers):
                nonlocal count
                count += 1
                if count == 1:
                    name, args = 'preview_merge', {'segments':['11','12']}
                elif count == 2:
                    tool = next(x for x in reversed(payload['messages']) if x['role']=='tool')
                    response = json.loads(tool['content'])
                    name, args = 'commit', {'preview_id':response['result']['preview_id']}
                else:
                    name, args = 'finish', {'summary':'Scripted synthetic merge; not model performance.'}
                return {'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant',
                    'content':'Synthetic smoke: '+name, 'tool_calls':[{'id':str(count),'type':'function',
                    'function':{'name':name,'arguments':json.dumps(args)}}]}}]}
            result = execute(config,root / (mode+'-run'),transport=scripted)
            scores = result[0]['scores']
            if result[0]['status']!='finished' or not all(s['vi_improvement_bits']>0 for s in scores.values()):
                raise RuntimeError('Synthetic edit did not finish and improve the frozen score')
            results[mode] = {'status':'passed','actions':3,'scores':scores}
    finally:
        if previous is None:os.environ.pop('CONNECTOME_SMOKE_KEY',None)
        else:os.environ['CONNECTOME_SMOKE_KEY']=previous
    (root/'smoke.json').write_text(json.dumps(results,indent=2))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    create_and_run(args.out)
    print(json.dumps({'synthetic_smoke':'passed','modes':['cb2','open-world'],'paid_model_calls':0}))


if __name__=='__main__':main()
