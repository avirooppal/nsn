"""Read-only competitor readiness. Missing systems never receive invented scores."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import socket

def readiness(pins_path, qualification=None):
    pins=json.loads(Path(pins_path).read_text(encoding='utf-8'))['sources']
    def port_open(port):
        try:
            socket.create_connection(('127.0.0.1',port),timeout=1).close();return True
        except OSError:return False
    result={'track':'actual OSS competitor readiness, not ranked results','systems':{}}
    definitions={
        'mem0':('mem0', 'Full native-conversation ingestion/query adapter and matched/recommended runs outstanding'),
        'graphiti':('graphiti_core','No configured graph backend/provider, isolated namespace adapter or matched/recommended run'),
        'letta_current':(None,'Current Letta code/App Server not provisioned; historical Python server is archived'),
        'amem':('agentic_memory','Pinned A-MEM package, Chroma store and local-provider adapter not provisioned')}
    for name,(module,reason) in definitions.items():
        result['systems'][name]={'source':pins[name],'module_available_in_this_environment':bool(importlib.util.find_spec(module)) if module else bool(shutil.which('letta')),'benchmark_status':'unevaluated','reason':reason,'score':None}
    result['systems']['graphiti']['localhost_ports']={str(p):port_open(p) for p in (6379,7687)}
    result['systems']['letta_current']['localhost_8283_open']=port_open(8283)
    if qualification:
        q=json.loads(Path(qualification).read_text(encoding='utf-8'))
        result['systems']['mem0']['local_setup_qualification']={'artifact':str(qualification),'passed':q['failure'] is None and q.get('returned_stored_port',False),'failure':q['failure'],'revision':q['source_revision'],'version':q['version'],'management_calls':len(q['calls']),'ingestion_ms':q.get('ingestion_ms'),'retrieval_ms':q.get('retrieval_ms')}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pins',default=str(Path(__file__).with_name('pins.json')));parser.add_argument('--qualification');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=readiness(args.pins,args.qualification)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
