"""Actual pinned Mem0 incremental public-corpus qualification, with bounded runs.

Run in the isolated competitor environment. Partial ingestion never becomes a
quality score. Each management call and completed turn is retained for resume.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from benchmarks.public_eval.corpus import factory
from benchmarks.public_eval.datasets import digest
from benchmarks.public_eval.runner import Ollama,text,packed,SYSTEM


def source_revision(distribution):
    direct=json.loads(distribution.read_text('direct_url.json') or '{}')
    revision=direct.get('vcs_info',{}).get('commit_id')
    if revision!='abb81c88e1f738a8117d8293530fbc31a5ef8fd9':raise ValueError('Mem0 source pin mismatch')
    return revision


def run(answer_run,assets,out,embedding_path,max_new_events=1,resume=False,management_context=None,management_output=None):
    os.environ.update(MEM0_TELEMETRY='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    root=Path(answer_run);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest['manifest_sha256']:
        raise ValueError('Answering manifest checksum changed')
    distribution=importlib.metadata.distribution('mem0ai')
    revision=source_revision(distribution)
    if not manifest.get('embedding'):raise ValueError('Matched neural encoder manifest required')
    for name,checksum in manifest['embedding']['file_sha256'].items():
        if hashlib.sha256((Path(embedding_path)/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Matched encoder asset changed')
    for name,info in manifest['sources']['files'].items():
        if name.endswith('.json') and ('locomo10' in name or manifest.get('longmemeval_variant','oracle') in name):
            checksum=hashlib.sha256()
            with (Path(assets)/name).open('rb') as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b''):checksum.update(chunk)
            if checksum.hexdigest()!=info['sha256']:raise ValueError('Native source changed')
    # Fixed first eligible question, rather than selection after seeing outcomes.
    case_id=manifest['case_ids'][0]
    case=next(c for c in factory(assets,manifest.get('longmemeval_variant','oracle'))() if c['id']==case_id)
    client=Ollama(manifest['model']['name'],manifest['model']['digest'],context=manifest['context_tokens'],output=manifest['reserved_output_tokens'])
    management_context=management_context or client.context
    management_output=management_output or client.output
    destination=Path(out)
    config={'llm':{'provider':'ollama','config':{'model':client.model,'ollama_base_url':client.host,'temperature':0,'max_tokens':client.output}},'embedder':{'provider':'huggingface','config':{'model':embedding_path,'embedding_dims':384,'model_kwargs':{'device':'cpu','local_files_only':True}}},'vector_store':{'provider':'qdrant','config':{'collection_name':'native_public','path':str(destination.resolve()/'qdrant'),'embedding_model_dims':384}},'history_db_path':str(destination.resolve()/'history.db')}
    config['llm']['config']['max_tokens']=management_output
    expected={'answer_manifest_sha256':manifest['manifest_sha256'],'case_id':case_id,'source_revision':revision,'version':distribution.version,'config':config,'native_events':len(case['events']),'events_sha256':digest(case['events']),'management_context':management_context,'management_output':management_output,'track':'matched resource' if (management_context,management_output)==(client.context,client.output) else 'explicit expanded local management budget; not matched or vendor-recommended'}
    if resume:
        state=json.loads((destination/'progress.json').read_text(encoding='utf-8'))
        if state['experiment']!=expected:raise ValueError('Cannot resume changed Mem0 experiment')
        if state['status']!='incomplete':raise ValueError('Completed or failed attempts must remain immutable')
    else:
        destination.mkdir(parents=True,exist_ok=False)
        state={'experiment':expected,'next_event':0,'status':'incomplete','quality_score':None,'calls':[],'turns':[]}
    from mem0 import Memory
    memory=None
    def counted(*args,**kwargs):
        kwargs['think']=False;kwargs.setdefault('options',{}).update(num_ctx=management_context,seed=42,num_predict=management_output)
        entry={'request':kwargs};started=time.perf_counter()
        try:
            if len(json.dumps(kwargs.get('messages',[]),ensure_ascii=False).encode('utf-8'))+256+management_output>management_context:
                entry['dispatched']=False
                raise ValueError('Management prompt exceeds conservative matched window; no silent server truncation')
            entry['dispatched']=True
            response=real(*args,**kwargs)
            entry['response']=response.model_dump(mode='json') if hasattr(response,'model_dump') else response
            return response
        except Exception as exc:entry['failure']=repr(exc);raise
        finally:
            entry['elapsed_ms']=(time.perf_counter()-started)*1000;state['calls'].append(entry)
    try:
        memory=Memory.from_config(config)
        import httpx
        memory.llm.client._client.timeout=httpx.Timeout(120)
        real=memory.llm.client.chat
        memory.llm.client.chat=counted
        for event in case['events'][state['next_event']:state['next_event']+max_new_events]:
            started=time.perf_counter()
            result=memory.add([{'role':event['role'],'content':text(event)}],user_id='native_public',infer=True)
            state['turns'].append({'event_id':event['id'],'result':result,'elapsed_ms':(time.perf_counter()-started)*1000})
            state['next_event']+=1
            (destination/'progress.json').write_text(json.dumps(state,indent=2,default=str),encoding='utf-8')
        if state['next_event']==len(case['events']):
            started=time.perf_counter();result=memory.search(case['question'],filters={'user_id':'native_public'},limit=10)
            state['retrieval_ms']=(time.perf_counter()-started)*1000
            memories=[{'id':r['id'],'session':'derived','date':'unspecified','speaker':'Mem0','role':'memory','content':r['memory']} for r in result.get('results',[])]
            context,ids,omitted=packed(memories,manifest['memory_budget_utf8_bytes'])
            prompt=manifest['answer_prompt_template'].format(context=context,date=case['question_date'],question=case['question'])
            answer,response=client.generate(prompt,system=manifest['system_prompt'])
            state.update(status='single_question_generation_complete_ungraded',answer=answer,answer_response=response,context=context,retrieved_derived_ids=ids,omitted_derived_ids=omitted)
            # Derived IDs are not native evidence coverage without claim validation.
            state['native_evidence_coverage']=None
    except Exception as exc:
        state.update(status='failed',failure=repr(exc),quality_score=None)
    finally:
        if memory is not None:
            memory.db.close();memory.vector_store.client.close()
        (destination/'progress.json').write_text(json.dumps(state,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:v for k,v in state.items() if k not in ('calls','turns','answer_response','experiment')},indent=2))
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--answer-run',required=True);parser.add_argument('--assets',default='benchmarks/assets/phase09');parser.add_argument('--output',required=True);parser.add_argument('--embedding-path',required=True);parser.add_argument('--max-new-events',type=int,default=1);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--management-context',type=int);parser.add_argument('--management-output',type=int)
    args=parser.parse_args()
    if args.max_new_events<1:parser.error('--max-new-events must be positive')
    run(args.answer_run,args.assets,args.output,args.embedding_path,args.max_new_events,args.resume,args.management_context,args.management_output)
