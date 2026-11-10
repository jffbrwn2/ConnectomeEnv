import base64,copy,io,json,sys,tempfile,types,unittest
from urllib.error import HTTPError
from proofreading import moonshot_transport as m
from proofreading import moonshot_spend_guard as s
from unittest.mock import patch

class Tests(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict('os.environ',{'CONNECTOME_MOONSHOT_BUDGET_USD':'10'})
  self.env.start();self.addCleanup(self.env.stop)
 def test_retry_preserves_all_images(self):
  with tempfile.TemporaryDirectory() as d:
   t=m.MoonshotTransport(d);t.upload=lambda data,mime,h:'test-'+data.decode()
   blocks=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(str(i).encode()).decode()}} for i in range(439)]
   p={'model':'kimi-k3','messages':[{'role':'user','content':blocks}]};original=copy.deepcopy(p);calls=[]
   def fake(directory,endpoint,payload,headers):
    calls.append(copy.deepcopy(payload))
    if len(calls)==1:
     e=HTTPError(endpoint,429,'overloaded',{},None);e.provider_detail={'retryable':True};raise e
    return {'id':'ok','usage':{}}
   old=m.guarded_post;sl=m.time.sleep
   try:
    m.guarded_post=fake;m.time.sleep=lambda seconds:None
    self.assertEqual(t(t.api+'/chat/completions',p,{})['id'],'ok')
   finally:m.guarded_post=old;m.time.sleep=sl
   self.assertEqual(p,original);self.assertEqual(calls[0],calls[1]);self.assertEqual(len(calls[1]['messages'][0]['content']),439)
 def test_billing_not_retryable_and_rejection_not_charged(self):
  with tempfile.TemporaryDirectory() as d:
   old=s.urlopen
   def fake(*a,**k):raise HTTPError('https://api.moonshot.ai/v1/chat/completions',429,'No credit',{},io.BytesIO(json.dumps({'error':{'message':'Insufficient balance'}}).encode()))
   try:
    s.urlopen=fake
    with self.assertRaises(HTTPError) as c:s.guarded_post(d,'https://api.moonshot.ai/v1/chat/completions',{'model':'kimi-k3','max_tokens':16384},{})
    self.assertFalse(c.exception.provider_detail['retryable'])
    ledger=json.load(open(d+'/spend-ledger.json'));self.assertEqual(next(iter(ledger['calls'].values()))['actual_usd'],0)
   finally:s.urlopen=old
 def test_cleanup_protects_history_and_invalidates_deleted_cache(self):
  from pathlib import Path
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   t=m.MoonshotTransport(d);keep=hashlib.sha256(b'protected').hexdigest()
   for i in range(901):
    digest=keep if i==0 else hashlib.sha256(str(i).encode()).hexdigest()
    Path(d,digest+'.json').write_text(json.dumps({'id':digest}))
    Path(d,digest+'.lock').touch()
   p={'messages':[{'content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b'protected').decode()}}]}]}
   original=copy.deepcopy(p);deleted=[];old=m.urlopen
   class Reply:
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self):return b'{}'
   def fake(req,**kwargs):
    self.assertEqual(req.get_method(),'DELETE');deleted.append(req.full_url.rsplit('/',1)[-1]);return Reply()
   try:m.urlopen=fake;t.prune(p,{'Authorization':'test'})
   finally:m.urlopen=old
   self.assertNotIn(keep,deleted);self.assertEqual(len(deleted),101)
   self.assertTrue(Path(d,keep+'.json').exists());self.assertEqual(p,original)
   self.assertEqual(len(Path(d,'retired.jsonl').read_text().splitlines()),101)
   for digest in deleted:
    self.assertFalse(Path(d,digest+'.json').exists())
    self.assertFalse(Path(d,digest+'.lock').exists())
 def test_failed_delete_keeps_cache(self):
  from pathlib import Path
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   for i in range(901):Path(d,hashlib.sha256(str(i).encode()).hexdigest()+'.json').write_text(json.dumps({'id':str(i)}))
   old=m.urlopen
   def fake(*a,**kw):raise HTTPError('test',500,'failure',{},None)
   try:
    m.urlopen=fake
    with self.assertRaises(HTTPError):m.MoonshotTransport(d).prune({'messages':[]},{'Authorization':'test'})
   finally:m.urlopen=old
   self.assertEqual(len(list(Path(d).glob('*.json'))),901)
if __name__=='__main__':unittest.main()


def test_no_implicit_spending_allowance(tmp_path,monkeypatch):
 import pytest
 monkeypatch.delenv('CONNECTOME_MOONSHOT_BUDGET_USD',raising=False)
 monkeypatch.setattr(s,'urlopen',lambda *a,**kw:pytest.fail('must not bill without explicit limit'))
 with pytest.raises(ValueError,match='explicit positive'):
  s.guarded_post(tmp_path,'https://api.moonshot.ai/v1/chat/completions',{'model':'kimi-k3','max_tokens':100},{})


def test_existing_ledger_limit_is_not_reset(tmp_path,monkeypatch):
 import pytest
 ledger={'limit_usd':1,'prior_probe_usd':0,'calls':{}}
 (tmp_path/'spend-ledger.json').write_text(json.dumps(ledger))
 monkeypatch.setenv('CONNECTOME_MOONSHOT_BUDGET_USD','600')
 monkeypatch.setattr(s,'urlopen',lambda *a,**kw:pytest.fail('must not exceed existing cap'))
 with pytest.raises(RuntimeError,match='ceiling'):
  s.guarded_post(tmp_path,'https://api.moonshot.ai/v1/chat/completions',{'model':'kimi-k3','max_tokens':100},{})
 assert json.loads((tmp_path/'spend-ledger.json').read_text())['limit_usd']==1
