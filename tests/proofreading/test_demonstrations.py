import base64,copy,hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from proofreading.demonstrations import load_context


def manifest(tmp_path):
    d=tmp_path/'demos';(d/'frames').mkdir(parents=True)
    p=d/'frames/a.png';Image.new('RGB',(32,32),'red').save(p)
    s={'schema':'visual-demonstrations-v1','introduction':'Independent example.',
       'examples':[{'case_id':'demo','image':'frames/a.png','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'caption':'Before and after.'}]}
    f=d/'manifest.json';f.write_text(json.dumps(s));return f,p


@pytest.mark.parametrize('provider,kind',[('responses','input_image'),('anthropic','image'),('chat','image_url')])
def test_original_image_bytes(tmp_path,provider,kind):
    m,p=manifest(tmp_path);blocks=load_context(m,provider)[0]['content'];image=next(x for x in blocks if x['type']==kind)
    value=image['source']['data'] if provider=='anthropic' else (image['image_url'] if provider=='responses' else image['image_url']['url']).split(',',1)[1]
    assert base64.b64decode(value)==p.read_bytes()
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):load_context(m,provider)


def test_context_persists_and_live_images_remain(tmp_path):
    from proofreading import Dataset,Session
    from proofreading.demo import create
    from proofreading.harness import run
    m,p=manifest(tmp_path);session=Session(Dataset.load(create(tmp_path/'data')),tmp_path/'episode')
    session.execute('set_view',{'width':32,'height':32});seen=[]
    def transport(endpoint,payload,headers):
        seen.append(copy.deepcopy(payload));tool='look' if len(seen)==1 else 'finish'
        return {'output':[{'type':'function_call','call_id':str(len(seen)),'name':tool,'arguments':json.dumps({} if tool=='look' else {'summary':'Done'})}]}
    result=run(session,'test',api_key='test',transport=transport,demonstrations=m)
    assert result['status']=='finished'
    assert seen[0]['input'][0]==seen[1]['input'][0]
    assert sum(b['type']=='input_image' for b in seen[0]['input'][0]['content'])==1
    assert sum(b['type']=='input_image' for b in seen[0]['input'][1]['content'])==2
    assert json.loads((session.output/'demonstrations.json').read_text())['sha256']==hashlib.sha256(m.read_bytes()).hexdigest()


def test_answer_shown_context_is_not_called_independent(tmp_path):
    m,_=manifest(tmp_path);spec=json.loads(m.read_text());spec['closing']='The images show the target for THIS task.';m.write_text(json.dumps(spec))
    assert load_context(m,'responses')[0]['content'][-1]['text']==spec['closing']
