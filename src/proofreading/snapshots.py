"""Saved model-visible observations; excludes provider private reasoning."""
import json,re
from pathlib import Path

def rows(path):
 return [json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []

def visual_updates(value):
 if isinstance(value,list):
  for x in value:yield from visual_updates(x)
 elif isinstance(value,dict):
  img=value.get('image');scene=value.get('scene_observation',{})
  if isinstance(img,str):
   if not re.fullmatch(r'frames/[A-Za-z0-9_.-]+\.png',img):raise ValueError('Unsafe image path')
   segment='scene' in value or Path(img).name.startswith('mesh-') or img.endswith('-scene.png')
   yield {'segment' if segment else 'em':img,**({'segment':scene['image']} if scene.get('image') else {})}
  for k,x in value.items():
   if k=='scene_observation' and img and scene.get('image'):continue
   if isinstance(x,(list,dict)):yield from visual_updates(x)

def snapshot(directory):
 events={r['step']:r for r in rows(directory/'events.jsonl')};actions=rows(directory/'model-actions.jsonl');text_by_call={}
 for row in rows(directory/'model-responses.jsonl'):
  response=row['response']
  if response.get('choices'):
   message=response['choices'][0]['message']
   blocks=[{'type':'text','text':message.get('content') or ''}]+[{'type':'tool_use','id':c['id']} for c in message.get('tool_calls',[])]
  else:blocks=response.get('content',response.get('output',[]))
  text=[];ids=[]
  for block in blocks:
   if block.get('type')=='text':text.append(block.get('text',''))
   elif block.get('type')=='message':text.extend(x.get('text','') for x in block.get('content',[]) if x.get('type') in ('text','output_text'))
   elif block.get('type') in ('function_call','tool_use'):ids.append(block.get('call_id',block.get('id')))
  for call_id in ids:text_by_call[call_id]='\n\n'.join(text)
 initial=json.loads((directory/'initial-observations.json').read_text());last={'em':None,'segment':None,'em_step':None,'segment_step':None};frames=[]
 def add(event,number,explanation,arguments,tool):
  nonlocal last
  views=[]
  for update in visual_updates(event.get('response',{})):
   for key,path in update.items():last[key]=path;last[key+'_step']=number
   views.append(dict(last))
  frames.append({**last,'views':views,'tool':tool,'text':explanation,'arguments':arguments,'ok':event.get('response',{}).get('ok',False),'error':event.get('response',{}).get('error'),'event_step':event.get('step')})
 add(events[initial[-1]],0,'The initial EM and segment views supplied to the model. The configured starting segment is selected.',{},'initial')
 for n,action in enumerate(actions,1):
  event=events.get(action.get('step'),{'response':{'ok':False,'error':'No recorded tool event'}})
  explanation=text_by_call.get(action.get('call_id'),'')
  if action['name']=='finish':explanation='\n\n'.join(x for x in [explanation,action.get('arguments',{}).get('summary','')] if x)
  add(event,n,explanation,action.get('arguments',{}),action['name'])
 names=set()
 for f in frames:
  for v in [f]+f['views']:
   for key in ['em','segment']:
    if v.get(key):names.add(v[key])
 for name in names:
  if not re.fullmatch(r'frames/[A-Za-z0-9_.-]+\.png',name) or not (directory/name).resolve().is_relative_to((directory/'frames').resolve()):raise ValueError('Unsafe saved image path')
  if not (directory/name).is_file():raise ValueError('Missing saved image '+name)
 return {'frames':frames,'source':'saved model-visible images and outward text; no evaluator imagery or private reasoning'},names
