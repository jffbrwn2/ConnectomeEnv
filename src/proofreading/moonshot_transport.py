"""Byte-preserving native Moonshot image transport; no observation compaction."""
import base64,copy,fcntl,hashlib,json,os,time,uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from proofreading.provider_http import request_json
from .moonshot_spend_guard import guarded_post

class MoonshotTransport:
 def __init__(self, directory):
  self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
  self.api='https://api.moonshot.ai/v1'
 def upload(self, data, mime, headers):
  digest=hashlib.sha256(data).hexdigest();path=self.directory/(digest+'.json')
  with (self.directory/(digest+'.lock')).open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if path.exists():return json.loads(path.read_text())['id']
   boundary='connectome-'+uuid.uuid4().hex
   body=(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nimage\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{digest}.png"\r\nContent-Type: {mime}\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
   response=request_json(Request(self.api+'/files',body,{'Authorization':headers['Authorization'],'Content-Type':'multipart/form-data; boundary='+boundary}))
   if not response.get('id'):raise RuntimeError('Image upload returned no file ID')
   if response.get('bytes') is not None and response['bytes']!=len(data):raise RuntimeError('Uploaded file byte count mismatch')
   record={'id':response['id'],'sha256':digest,'bytes':len(data),'mime_type':mime,'response':response,'uploaded_at':time.time()}
   temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(path)
   return record['id']
 def prune(self, payload, headers):
  """Keep every image in this full history; evict only our other cached copies.

  Called under the shared request lock, including during inference, so another
  worker cannot delete images referenced by an in-flight request.
  """
  keep=set()
  for message in payload.get('messages',[]):
   blocks=message.get('content',[])
   if not isinstance(blocks,list):continue
   for block in blocks:
    if block.get('type')=='image_url':
     url=block['image_url']['url']
     if not url.startswith('data:image/'):raise ValueError('Expected original inline images')
     keep.add(hashlib.sha256(base64.b64decode(url.split(',',1)[1],validate=True)).hexdigest())
  if len(keep)>900:raise RuntimeError('History needs more than 900 unique image files; refusing to alter observations')
  cached={p.stem:p for p in self.directory.glob('*.json') if len(p.stem)==64 and all(c in '0123456789abcdef' for c in p.stem)}
  missing=len(keep-set(cached));removed=0
  if len(cached)+missing<=900:return
  for digest,path in sorted(cached.items(),key=lambda x:x[1].stat().st_mtime):
   if digest in keep:continue
   record=json.loads(path.read_text())
   try:
    with urlopen(Request(self.api+'/files/'+record['id'],headers={'Authorization':headers['Authorization']},method='DELETE'),timeout=120) as response:response.read()
   except HTTPError as exc:
    if exc.code!=404:raise
   # Preserve provenance, and invalidate cache only after confirmed removal.
   with (self.directory/'retired.jsonl').open('a') as history:
    history.write(json.dumps({'sha256':digest,'record':record,'retired_at':time.time()})+'\n')
    history.flush();os.fsync(history.fileno())
   path.unlink()
   (self.directory/(digest+'.lock')).unlink(missing_ok=True)
   removed+=1
   if len(cached)-removed+missing<=800:break
  print('MOONSHOT_FILE_CLEANUP '+json.dumps({'deleted_copies':removed,'protected_history_images':len(keep)}),flush=True)
 def convert(self,payload,headers):
  result=copy.deepcopy(payload);images=[]
  for m in result.get('messages',[]):
   if not isinstance(m.get('content'),list):continue
   for block in m['content']:
    if block.get('type')!='image_url':continue
    url=block['image_url']['url']
    if not url.startswith('data:image/'):raise ValueError('Expected original inline image for auditable upload')
    prefix,encoded=url.split(',',1)
    mime=prefix[5:].split(';')[0];data=base64.b64decode(encoded,validate=True)
    fid=self.upload(data,mime,headers)
    block['image_url']['url']='ms://'+fid
    images.append({'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'file_id':fid})
  return result,images
 def __call__(self,endpoint,payload,headers):
  # Cross-process lock spans cleanup, upload and inference. File references must
  # remain valid throughout the provider call. Original PNGs stay on disk.
  with (self.directory/'request-lifecycle.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   return self._call_locked(endpoint,payload,headers)
 def _call_locked(self,endpoint,payload,headers):
  if endpoint!=self.api+'/chat/completions':raise ValueError('Moonshot transport requires native endpoint')
  self.prune(payload,headers)
  mapped,images=self.convert(payload,headers)
  raw=json.dumps(mapped).encode()
  if len(raw)>=100_000_000:raise ValueError('Native request exceeds 100 MB; refusing to alter history')
  for attempt in range(6):
   try:
    response=guarded_post(self.directory,endpoint,mapped,headers)
    break
   except HTTPError as exc:
    detail=getattr(exc,'provider_detail',{})
    if not detail.get('retryable') or attempt==5:raise
    delay=min(60,5*2**attempt)
    print('MOONSHOT_RETRY '+json.dumps({'attempt':attempt+1,'http_status':exc.code,'delay_seconds':delay}),flush=True)
    time.sleep(delay)
  with (self.directory/'requests.jsonl').open('a') as f:
   f.write(json.dumps({'response_id':response.get('id'),'model':payload['model'],'request_bytes':len(raw),'images':images,'usage':response.get('usage'), 'at':time.time()})+'\n')
  return response
