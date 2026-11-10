"""Create a small Docker benchmark configuration from an unpacked release bundle."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path)
    p.add_argument('--out',type=Path,default=Path('config.json'))
    p.add_argument('--provider',choices=['responses','anthropic','gemini','openrouter','chat'],default='responses')
    p.add_argument('--model',default='preflight-only')
    p.add_argument('--endpoint')
    p.add_argument('--case',action='append',dest='cases',help='Explicit case ID; repeat to select several')
    p.add_argument('--container-bundle',default='/data/cb2-1000')
    a=p.parse_args()
    if a.out.exists():p.error('Output already exists; choose a new filename')
    if a.provider=='chat' and not a.endpoint:p.error('chat requires --endpoint')
    records=json.loads((a.bundle/'index.json').read_text())
    if a.cases:
        if len(set(a.cases))!=len(a.cases) or set(a.cases)-{r['case_id'] for r in records}:p.error('Unknown or duplicate case ID')
        cases=a.cases
    else:
        cases=[next(r['case_id'] for r in records if r['task']==kind) for kind in ['split_error','merge_error']]
    config=json.loads((a.bundle/'example.json').read_text())
    config.update(benchmark=a.container_bundle,cases=cases,model={'provider':a.provider,'name':a.model})
    if a.endpoint:config['model']['endpoint']=a.endpoint
    a.out.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'config':str(a.out),'cases':cases,'model':a.model,'preflight_only_placeholder':a.model=='preflight-only'}))


if __name__=='__main__':main()
