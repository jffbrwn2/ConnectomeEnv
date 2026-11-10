"""Shared dollar ceiling, including reservations for concurrent/uncertain calls."""
import fcntl,json,time,uuid,os,math
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError

def cost(usage):
 inp=usage.get('prompt_tokens',0);out=usage.get('completion_tokens',0)
 details=usage.get('prompt_tokens_details') or {}
 cached=details.get('cached_tokens',details.get('cache_read_tokens',usage.get('cached_tokens',0)))
 return (3*(inp-cached)+.3*cached+15*out)/1e6

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def guarded_post(directory,endpoint,payload,headers):
 if payload['model']!='kimi-k3':raise ValueError('Budget rates are pinned to kimi-k3')
 path=Path(directory)/'spend-ledger.json';lockpath=path.with_suffix('.lock')
 # Full context at the uncached rate, plus maximum possible output. This is
 # conservative even when the provider token estimator or cache misses differ.
 reserve=(1048576*3+payload['max_tokens']*15)/1e6
 ident=uuid.uuid4().hex
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():ledger=json.loads(path.read_text())
  else:
   prior=Path(directory).parent/'two-image-probe.json'
   initial=cost(json.loads(prior.read_text())['usage']) if prior.exists() else 0
   limit=float(os.environ.get('CONNECTOME_MOONSHOT_BUDGET_USD', '0'))
   if not math.isfinite(limit) or limit<=0:
    raise ValueError('Set an explicit positive CONNECTOME_MOONSHOT_BUDGET_USD before inference')
   ledger={'limit_usd':limit,'prior_probe_usd':initial,'calls':{}}
  committed=ledger['prior_probe_usd']+sum(c.get('actual_usd',c['reserved_usd']) for c in ledger['calls'].values())
  if committed+reserve>ledger['limit_usd']:
   write(Path(directory).parent/'PAUSED_BUDGET.json',{'committed_or_reserved_usd':committed,'next_reservation_usd':reserve,'limit_usd':ledger['limit_usd'],'at':time.time()})
   raise RuntimeError('Dollar ceiling reached; request not sent; history retained')
  ledger['calls'][ident]={'reserved_usd':reserve,'status':'inflight','at':time.time()};write(path,ledger)
 # One attempt only. Ambiguous failures retain their full reservation and
 # recovery cannot invisibly retry a paid request without another reservation.
 try:
  request=Request(endpoint,json.dumps(payload).encode(),{**headers,'Content-Type':'application/json'})
  with urlopen(request,timeout=1200) as response:result=json.load(response)
 except Exception as exc:
  detail={'type':type(exc).__name__}
  if isinstance(exc,HTTPError):
   try:
    err=json.loads(exc.read(65536)).get('error',{})
    detail.update(http_status=exc.code,code=err.get('code'),message=str(err.get('message',''))[:600])
   except Exception:pass
   message=str(detail.get('message','')).lower()
   billing=exc.code==402 or any(x in message for x in ['insufficient balance','insufficient credit','spending cap','billing','quota exhausted'])
   exc.provider_detail={'http_status':exc.code,'retryable':not billing and exc.code in [429,500,502,503,504,520,529],'billing_action_required':billing}
  with lockpath.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(path.read_text())
   ledger['calls'][ident].update(status='uncertain',error=detail)
   if isinstance(exc,HTTPError) and exc.code in [400,401,402,403,404,413,422,429]:
    ledger['calls'][ident].update(status='rejected',actual_usd=0.0)
   write(path,ledger)
  raise
 usage=result.get('usage')
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(path.read_text());entry=ledger['calls'][ident]
  entry.update(status='completed',response_id=result.get('id'),usage=usage)
  if usage is not None:entry['actual_usd']=cost(usage)
  write(path,ledger)
 return result
