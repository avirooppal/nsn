import asyncio
import copy
import json
import sqlite3
import subprocess
import sys
import pytest
from nsn import Runtime
from neurosleepnet.integrations.tool import MemoryTool


def test_procedures_observed_claims_failures_scope_and_invalidation(tmp_path):
    rt = Runtime(str(tmp_path))
    pid = rt.record_procedure(title='parse JSON file',steps=['Read UTF-8','Decode JSON'],environment={'format':'json'},tools=['read'])
    claim = rt.append_event(role='assistant',content='I successfully parsed the file')
    rt.record_outcome(procedure_id=pid,event_id=claim,outcome='success')
    args = dict(environment={'format':'json'},tools=['read'])
    assert rt.retrieve_procedures('parse JSON',**args)==[]
    with pytest.raises(ValueError):
        rt.record_outcome(procedure_id=pid,event_id=claim,outcome='success',evidence='observed')
    trace = rt.append_event(role='tool',content='JSON decode returned expected object')
    rt.record_outcome(procedure_id=pid,event_id=trace,outcome='success',evidence='observed')
    assert rt.retrieve_procedures('parse JSON',**args)[0]['status']=='confirmed'
    assert rt.retrieve_procedures('parse JSON',environment={'format':'csv'},tools=['read'])==[]
    assert rt.retrieve_procedures('parse JSON',namespace='other',**args)==[]
    with pytest.raises(ValueError):
        rt.record_outcome(procedure_id=pid,event_id=trace,outcome='success',namespace='other',evidence='observed')
    failed = rt.append_event(role='tool',content='JSONDecodeError: invalid input',turn_status='failed')
    rt.record_outcome(procedure_id=pid,event_id=failed,outcome='failure',evidence='observed')
    warning = rt.retrieve_procedures('parse JSON',**args)[0]
    assert warning['warning'] and warning['status']=='failed' and len(warning['traces'])==3
    rt.storage.delete_event(failed,'default')
    assert rt.retrieve_procedures('parse JSON',**args)[0]['status']=='confirmed'
    conn = sqlite3.connect(rt.db_path)
    with conn:
        conn.execute('UPDATE events SET content=? WHERE id=?',('unrelated content',trace))
    conn.close()
    assert rt.retrieve_procedures('parse JSON',**args)==[]


def populated(rt):
    event = rt.append_event(namespace='a',role='user',content='Alice works for Acme',event_time='2025-01-01T00:00:00Z',session_id='s')
    rt.record_fact(namespace='a',subject='Alice',predicate='employer',value='Acme',supporting_event_ids=[event],valid_from='2025-01-01T00:00:00Z')
    rt.extract_relations(event,namespace='a')
    rt.sleep(namespace='a')
    pid = rt.record_procedure(namespace='a',title='Read company JSON',steps=['Read file','Parse JSON'])
    outcome = rt.append_event(namespace='a',role='tool',content='Read succeeded')
    rt.record_outcome(namespace='a',procedure_id=pid,event_id=outcome,outcome='success',evidence='observed')


def test_backup_roundtrip_revisions_lineage_atomic_conflicts_and_deletion(tmp_path):
    rt = Runtime(str(tmp_path/'source'),graph=True)
    populated(rt)
    other = rt.append_event(namespace='b',content='private B')
    snapshot = rt.export_memory('a')
    target = Runtime(str(tmp_path/'target'),graph=True)
    target.append_event(namespace='b',content='retained B')
    target.import_memory(json.loads(json.dumps(snapshot)),namespace='a')
    assert target.export_memory('a')==snapshot
    assert target.retrieve_pack('Alice Acme',namespace='a').items
    with pytest.raises(ValueError):
        target.import_memory(snapshot,namespace='a')
    assert target.export_memory('a')==snapshot
    bad = copy.deepcopy(snapshot)
    bad['schema_version']=999
    with pytest.raises(ValueError):
        Runtime(str(tmp_path/'bad')).import_memory(bad,namespace='a')
    bad = copy.deepcopy(snapshot)
    bad['tables']['procedure_traces'][0]['event_id']=other
    empty = Runtime(str(tmp_path/'empty'))
    with pytest.raises(ValueError):
        empty.import_memory(bad,namespace='a')
    assert empty.stats('a')['counts']['events']==0
    target.delete_namespace('a')
    assert all(v==0 for v in target.stats('a')['counts'].values())
    assert len(target.timeline(namespace='b'))==1
    assert target.retrieve_pack('Alice',namespace='a').items==[]


def test_tool_and_runnable_contracts(tmp_path):
    from neurosleepnet.integrations.langchain import wrap_runnable
    rt = Runtime(str(tmp_path))
    tool = MemoryTool(namespace='a',runtime=rt)
    tool.invoke({'action':'remember','text':'Payment gateway port 8080'})
    assert asyncio.run(tool.ainvoke({'action':'recall','query':'gateway port'}))
    assert MemoryTool(namespace='b',runtime=rt).recall('gateway port')==[]
    with pytest.raises(ValueError):
        tool.invoke({'action':'record_outcome'})
    class Runnable:
        def invoke(self,text):
            assert '8080' in text
            if text.endswith('novel-unique'):
                assert 'novel-unique' not in text.split('</retrieved_evidence>')[0]
            return 'ok'
        async def ainvoke(self,text):
            return self.invoke(text)
    wrapped = wrap_runnable(Runnable(),rt,namespace='a')
    assert wrapped.invoke('gateway port novel-unique')=='ok'
    assert asyncio.run(wrapped.ainvoke('gateway port'))=='ok'
    class Broken:
        def invoke(self,text):
            raise RuntimeError('failed task')
        async def ainvoke(self,text):
            raise RuntimeError('failed async task')
    broken = wrap_runnable(Broken(),rt,namespace='a')
    with pytest.raises(RuntimeError):
        broken.invoke('gateway')
    with pytest.raises(RuntimeError):
        asyncio.run(broken.ainvoke('gateway'))
    assert len([r for r in rt.timeline(namespace='a') if r['turn_status']=='failed'])==2
    assert rt.timeline(namespace='b')==[]
    class Cancelled:
        async def ainvoke(self,text):
            raise asyncio.CancelledError()
    cancelled = wrap_runnable(Cancelled(),rt,namespace='a')
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled.ainvoke('gateway'))
    assert any(r['turn_status']=='cancelled' for r in rt.timeline(namespace='a'))


def test_optional_integrations_absent_minimal_operates(tmp_path):
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('fastapi','langchain_core','numpy','torch','faiss','spacy','sentence_transformers'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0,Block())
import nsn
from neurosleepnet.integrations.tool import MemoryTool
from neurosleepnet.integrations.api import create_app
from neurosleepnet.integrations.langchain import NeurosleepNetHistory
rt=nsn.Runtime(sys.argv[1])
tool=MemoryTool(runtime=rt)
tool.remember('gateway port 8080')
assert tool.recall('gateway port')
assert rt.sleep()['processed']
rt.close()
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True)


def test_rest_authentication_fixed_scope_and_limits(tmp_path):
    from fastapi.testclient import TestClient
    from neurosleepnet.integrations.api import create_app
    rt = Runtime(str(tmp_path))
    with pytest.raises(ValueError):
        create_app(runtime=rt)
    client = TestClient(create_app(runtime=rt,tokens={'a'*32:'a','b'*32:'b'}))
    a = {'Authorization':'Bearer '+'a'*32}
    b = {'Authorization':'Bearer '+'b'*32}
    assert client.post('/observe',json={'content':'secret a'}).status_code==401
    assert client.post('/observe',headers=a,json={'content':'secret a'}).status_code==200
    assert client.post('/observe',headers=a,json={'content':'secret','namespace':'b'}).status_code==422
    assert client.post('/observe',headers=a,json={'content':'x'*66000}).status_code==413
    assert len(client.get('/timeline',headers=a).json())==1
    assert client.get('/timeline',headers=b).json()==[]
    assert client.get('/search?q=secret&namespace=a',headers=b).status_code==422


def test_namespace_deletion_invalidates_other_runtime_vector_cache(tmp_path):
    from neurosleepnet.embeddings.pooled import FastHashEncoder
    encoder = FastHashEncoder(dimension=16)
    rt = Runtime(str(tmp_path),encoder=encoder)
    rt.append_event(namespace='a',content='private alpha')
    rt.append_event(namespace='b',content='private beta')
    reader = Runtime(str(tmp_path),encoder=encoder)
    assert reader.vector_store.search(encoder.encode('alpha'),namespace='a')
    rt.delete_namespace('a')
    assert reader.vector_store.search(encoder.encode('alpha'),namespace='a')==[]
    assert reader.vector_store.search(encoder.encode('beta'),namespace='b')


def test_backup_rejects_global_id_collision_without_partial_import(tmp_path):
    source = Runtime(str(tmp_path/'source'))
    eid = source.append_event(namespace='a',content='alpha')
    snapshot = source.export_memory('a')
    target = Runtime(str(tmp_path/'target'))
    target.storage.append_event(namespace='b',content='beta',id=str(eid))
    with pytest.raises(ValueError):
        target.import_memory(snapshot,namespace='a')
    assert target.timeline(namespace='a')==[]
    assert target.timeline(namespace='b')[0]['content']=='beta'


def test_procedure_constraints_and_failed_source_cannot_confirm_success(tmp_path):
    rt=Runtime(str(tmp_path))
    for bad in ({'tools':'read'}, {'prerequisites':{}}, {'environment':[]}):
        with pytest.raises(ValueError):
            rt.record_procedure(title='Read',steps=['Read'],**bad)
    pid=rt.record_procedure(title='Read',steps=['Read'])
    failed=rt.append_event(role='tool',content='Failed',turn_status='failed')
    with pytest.raises(ValueError):
        rt.record_outcome(procedure_id=pid,event_id=failed,outcome='success',evidence='observed')
    assert rt.retrieve_procedures('Read')==[]
