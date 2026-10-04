"""Installed memory smoke with actual local Ollama; no accuracy benchmark claim."""
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import nsn

ROOT = Path(__file__).parent
assert 'site-packages' in Path(nsn.__file__).parts
calls = []
def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    with urlopen(Request('http://127.0.0.1:11434/api/'+path, data=body,
                         headers={'Content-Type':'application/json'}), timeout=180) as response:
        return json.load(response)

tags = request('tags')
model_name = 'qwen3:1.7b'
model_digest = next(m['digest'] for m in tags['models'] if m['name']==model_name)
def existing_model(prompt):
    response = request('chat', {'model':model_name, 'stream':False, 'think':False,
        'messages':[{'role':'system','content':'Answer with only the requested token. Use supplied evidence. If it is absent answer UNKNOWN.'},
                    {'role':'user','content':prompt}],
        'options':{'temperature':0,'seed':42,'num_ctx':2048,'num_predict':32}})
    calls.append({'prompt':prompt,'response':response})
    assert response['done']
    return response['message']['content']

with tempfile.TemporaryDirectory() as directory:
    nsn.init(directory)
    model = nsn.wrap(existing_model, namespace='alice')
    model('Remember: Kestrel-X7422 access token is violet-7812.')
    answer = model('What is the Kestrel-X7422 access token?')
    assert '<retrieved_evidence>' in calls[1]['prompt']
    assert 'violet-7812' in calls[1]['prompt']
    assert not nsn.get_default_runtime().retrieve_pack('Kestrel-X7422 access token',namespace='bob').items
    nsn.close()
    nsn.init(directory)
    model = nsn.wrap(existing_model, namespace='alice')
    restarted = model('What is the Kestrel-X7422 access token?')
    assert 'violet-7812' in calls[2]['prompt']
    nsn.close()
assert len(calls)==3
result = {'installed_module':nsn.__file__, 'model':model_name,'model_digest':model_digest,
    'memory_injection_passed':True,'restart_injection_passed':True,'namespace_isolation_passed':True,
    'observed_token_answers':[answer,restarted], 'token_answer_matches':['violet-7812' in a for a in (answer,restarted)],
    'scope':'Single synthetic nonce smoke; not an accuracy or superiority benchmark', 'calls':calls}
(ROOT/'local-model-smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='calls'},indent=2))
