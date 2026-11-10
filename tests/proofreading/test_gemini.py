import json
from test_workbench import dataset, session
from proofreading.harness import run
from proofreading.usage import summarize_usage


def test_native_image_tool_roundtrip(session):
    requests=[]
    def transport(endpoint,payload,headers):
        requests.append(payload)
        assert 'x-goog-api-key' in headers
        assert payload['system_instruction']
        assert len(payload['tools']) >= 18
        if len(requests)==1:
            assert len([x for x in payload['input'] if x['type']=='image'])==2
            return {'id':'interaction1','status':'requires_action','steps':[
                {'type':'thought','signature':'opaque'},
                {'type':'function_call','id':'call1','name':'look','arguments':{}}]}
        assert payload['previous_interaction_id']=='interaction1'
        result=payload['input'][0]
        assert result['type']=='function_result' and result['call_id']=='call1'
        assert len([x for x in result['result'] if x['type']=='image'])==2
        return {'id':'interaction2','status':'completed','steps':[{'type':'model_output','content':[{'type':'text','text':'done'}]}]}
    result=run(session,model='gemini-4-argon',provider='gemini',api_key='test',transport=transport,max_turns=2,max_actions=2)
    assert len(requests)==2
    saved=[json.loads(line) for line in (session.output/'model-responses.jsonl').read_text().splitlines()]
    assert saved


def test_usage_counts_cache_and_thoughts():
    result=summarize_usage('gemini',[{'total_input_tokens':100,'total_cached_tokens':75,'total_output_tokens':10,'total_thought_tokens':20}])
    assert result['uncached_input_tokens']==25
    assert result['output_tokens']==30
    assert result['cache_read_tokens']==75
