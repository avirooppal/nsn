"""Actual pinned Mem0 OSS local setup qualification, not head-to-head quality."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

def run(embedding_path,output,resume=None):
    os.environ['MEM0_TELEMETRY']='false'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    from mem0 import Memory
    distribution=importlib.metadata.distribution('mem0ai')
    direct=json.loads(distribution.read_text('direct_url.json') or '{}')
    if direct.get('vcs_info',{}).get('commit_id')!='abb81c88e1f738a8117d8293530fbc31a5ef8fd9':
        raise RuntimeError('Mem0 source pin mismatch')
    report={'system':'Mem0 OSS','source_revision':direct['vcs_info']['commit_id'],'version':distribution.version,'track':'setup qualification ONLY; not a public benchmark comparison','failure':None,'calls':[]}
    previous=json.loads(Path(resume).read_text(encoding='utf-8')) if resume else None
    directory=previous['retained_workspace'] if previous else tempfile.mkdtemp(prefix='nsn_mem0_qualification_')
    report['retained_workspace']=directory
    memory=None
    config={'llm':{'provider':'ollama','config':{'model':'qwen3:8b','ollama_base_url':'http://127.0.0.1:11434','temperature':0,'max_tokens':128}},'embedder':{'provider':'huggingface','config':{'model':embedding_path,'embedding_dims':384,'model_kwargs':{'device':'cpu','local_files_only':True}}},'vector_store':{'provider':'qdrant','config':{'collection_name':'nsn_setup_qualification','path':str(Path(directory)/'qdrant'),'embedding_model_dims':384}},'history_db_path':str(Path(directory)/'history.db')}
    report['config']=config
    if previous:
        if previous['source_revision']!=report['source_revision'] or not Path(directory).is_dir():
            raise ValueError('Resume qualification source/workspace mismatch')
        report.update(calls=previous['calls'],add_result=previous['add_result'],ingestion_ms=previous['ingestion_ms'],resume_artifact=resume)
    try:
        memory=Memory.from_config(config)
        import httpx
        memory.llm.client._client.timeout=httpx.Timeout(120)
        real=memory.llm.client.chat
        def counted(*args,**kwargs):
            kwargs['think']=False
            kwargs.setdefault('options',{}).update(num_ctx=2048,seed=42)
            start=time.perf_counter();entry={'request':kwargs}
            try:
                response=real(*args,**kwargs)
                entry['response']=response.model_dump(mode='json') if hasattr(response,'model_dump') else response
                return response
            except Exception as exc:
                entry['failure']=repr(exc);raise
            finally:
                entry['elapsed_ms']=(time.perf_counter()-start)*1000;report['calls'].append(entry)
        memory.llm.client.chat=counted
        if not previous:
            start=time.perf_counter()
            report['add_result']=memory.add([{'role':'user','content':'My production gateway port is 8080.'}],user_id='nsn_setup',infer=True)
            report['ingestion_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();report['search_result']=memory.search('What is my production gateway port?',filters={'user_id':'nsn_setup'},limit=3)
        report['retrieval_ms']=(time.perf_counter()-start)*1000
        report['returned_stored_port']=any('8080' in str(r) for r in report['search_result'].get('results',[]))
        report['disk_bytes']=sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
        memory.vector_store.client.close()
    except Exception as exc: report['failure']=repr(exc)
    finally:
        if memory is not None:
            memory.db.close()
            memory.vector_store.client.close()
    Path(output).write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('calls','config')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--embedding-path',required=True);parser.add_argument('--output',required=True);parser.add_argument('--resume-from')
    args=parser.parse_args();run(args.embedding_path,args.output,args.resume_from)
