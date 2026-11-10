"""Keep native Interactions responses while exposing outward text/tool calls."""
import json


def normalized_response(response):
    if not isinstance(response.get('id'), str) or not response['id']:
        raise ValueError('Gemini response lacks an interaction ID')
    content=[]
    for step in response.get('steps', []):
        if step.get('type') == 'model_output':
            content.extend({'type':'text','text':block['text']} for block in step.get('content', [])
                           if block.get('type') == 'text')
        elif step.get('type') == 'function_call':
            args=step.get('arguments', {})
            if isinstance(args,str):args=json.loads(args)
            content.append({'type':'tool_use','id':step['id'],'name':step['name'],'input':args})
    return {**response, 'content':content}
