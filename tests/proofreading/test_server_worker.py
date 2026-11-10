import json,threading
from types import SimpleNamespace
from urllib.request import Request,urlopen
from proofreading.server import make_server


def test_http_actions_keep_one_worker_across_connections(tmp_path):
    threads=[]
    def execute(name,args):
        threads.append(threading.get_ident());return {'ok':True}
    session=SimpleNamespace(lock=threading.RLock(),execute=execute,output=tmp_path)
    server=make_server(session,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        for _ in range(3):
            request=Request(f'http://127.0.0.1:{server.server_port}/api/tool',
                data=json.dumps({'name':'look','arguments':{}}).encode(),
                headers={'Authorization':'Bearer '+server.agent_token,'Content-Type':'application/json','Connection':'close'})
            with urlopen(request) as response:assert json.load(response)['ok']
        assert len(set(threads))==1 and threads[0]!=threading.get_ident()
    finally:
        server.shutdown();server.server_close();thread.join()
