"""Run matched prompt trials on pre-frozen CB2 training cases; never retry inference."""
import argparse,fcntl,json,time
from pathlib import Path
from proofreading import Dataset,Session
from proofreading.harness import run
from proofreading.evaluate import evaluate_frozen,file_hash
from proofreading.objectives import LOCAL_PROOFREADING,LOCAL_PROOFREADING_JUNCTION_EVIDENCE


def model_configuration(model,effort=None,prompt_cache="auto"):
    if prompt_cache not in ("auto","off","5m","1h"):
        raise ValueError("Unknown prompt cache TTL")
    if prompt_cache not in ("auto", "off") and model != "claude-opus-5-5":
        raise ValueError("Explicit prompt caching is Anthropic-only")
    if model=='gpt-6-astra':
        if effort=='xhigh':raise ValueError('Use a verified Astra effort: low, medium, high, or max')
        return 'responses',{'reasoning':{'effort':effort or 'max'}}
    if model=='claude-opus-5-5':
        options={'output_config':{'effort':effort}} if effort else {}
        options['cache_control']=None if prompt_cache == 'off' else {'type':'ephemeral','ttl':'5m' if prompt_cache == 'auto' else prompt_cache}
        return 'anthropic',options
    raise ValueError('Unsupported practice model')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--operation',action='append',required=True)
    p.add_argument('--model',choices=['claude-opus-5-5','gpt-6-astra'],default='claude-opus-5-5')
    p.add_argument('--max-actions',type=int,default=40)
    p.add_argument('--phase',choices=['practice','validation'],default='practice')
    p.add_argument('--effort',choices=['low','medium','high','xhigh','max'])
    p.add_argument('--max-output-tokens',type=int,default=8192)
    p.add_argument('--prompt-cache',choices=['auto','off','5m','1h'],default='auto')
    p.add_argument('--prompt',choices=['baseline','junction-evidence'],action='append')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    prompts={'baseline':LOCAL_PROOFREADING,'junction-evidence':LOCAL_PROOFREADING_JUNCTION_EVIDENCE}
    if args.prompt:prompts={k:v for k,v in prompts.items() if k in args.prompt}
    provider,options=model_configuration(args.model,args.effort,args.prompt_cache)
    records=json.loads((args.benchmark/'index.json').read_text())
    selected=[r for r in records if r['operation_id'] in args.operation]
    if {r['operation_id'] for r in selected}!=set(args.operation):raise ValueError('Unknown requested operation')
    expected_split='train' if args.phase=='practice' else 'test'
    if any(r['split']!=expected_split for r in selected):raise ValueError('Benchmark split does not match requested phase')
    with (args.output/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for record in selected:
            case=args.benchmark/record['case_id']
            dataset=Dataset.load(case/'public/dataset.json')
            if dataset.fingerprint!=record['dataset_sha256']:raise ValueError('Frozen dataset changed')
            task=json.loads((case/'public/task.json').read_text())
            for label,objective in prompts.items():
                dest=args.output/(record['case_id']+'--'+label)
                if dest.exists():
                    previous=json.loads((dest/'intent.json').read_text())
                    if previous['model']!=args.model or previous.get('model_options',{})!=options or previous['max_actions']!=args.max_actions:
                        raise ValueError('Existing episode uses a different configuration; choose a separate output directory')
                    print('SKIP existing episode (never automatically repeat)',dest.name,flush=True);continue
                dest.mkdir()
                # Persist intent before any API call. Interruption requires inspection.
                intent={'case_id':record['case_id'],'phase':args.phase,'prompt':label,'model':args.model,'provider':provider,
                        'model_options':options,'max_output_tokens':args.max_output_tokens,
                        'max_actions':args.max_actions,'created_at_unix':time.time(),
                        'benchmark_index_sha256':file_hash(args.benchmark/'index.json')}
                (dest/'intent.json').write_text(json.dumps(intent,indent=2)+'\n')
                try:
                    session=Session(dataset,dest/'episode')
                    # Verify reference hashes and pre-episode freeze before spending tokens.
                    for metric in ('human','certified'):
                        evaluate_frozen(dataset,session.output,case/'evaluator'/(metric+'-frozen.json'))
                    for name,arguments in [
                        ('set_view',{'center_nm':task['center_nm'],'pixel_size_nm':16,'width':384,'height':384}),
                        ('select',{'segments':[task['seed_segment']],'mode':'replace'}),
                        ('set_scene',{'target_nm':task['center_nm'],'view_height_nm':task['extent_nm']*1.1})]:
                        result=session.execute(name,arguments)
                        if not result['ok']:raise RuntimeError('Practice setup failed: '+str(result))
                    objective+='\n\nReview the starting process near the initial view center. Focus edits on this local junction; record uncertainty if a continuation leaves the crop.'
                    summary=run(session,args.model,provider=provider,max_turns=args.max_actions,max_actions=args.max_actions,max_output_tokens=args.max_output_tokens,model_options=options,objective=objective)
                    result={'intent':intent,'harness':{k:v for k,v in summary.items() if k!='usage'},
                            'scores':{metric:evaluate_frozen(dataset,session.output,case/'evaluator'/(metric+'-frozen.json')) for metric in ('human','certified')}}
                    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                    print(json.dumps({'run':dest.name,'status':summary['status'],'actions':summary['actions'],
                                      'human_vi_improvement':result['scores']['human']['vi_improvement_bits'],
                                      'certified_vi_improvement':result['scores']['certified']['vi_improvement_bits']}),flush=True)
                except Exception as exc:
                    (dest/'failure.json').write_text(json.dumps({'error_type':type(exc).__name__,'message':str(exc),'automatic_retry':False},indent=2)+'\n')
                    print('FAILED',dest.name,type(exc).__name__,str(exc),flush=True)
                    raise


if __name__=='__main__':main()
