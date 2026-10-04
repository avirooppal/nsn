"""End-to-end memory contracts, without relying on an LLM's reasoning."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from nsn import Runtime
from neurosleepnet.embeddings.pooled import FastHashEncoder


def test_wrapped_model_receives_prior_memory_once_and_survives_restart(tmp_path):
    calls=[]
    def model(prompt):
        calls.append(prompt)
        return 'acknowledged'
    with Runtime(str(tmp_path)) as rt:
        agent=rt.wrap(model,namespace='alice')
        agent('Orion launch code is violet.')
        assert '<retrieved_evidence>' not in calls[0]
        agent('What is the Orion launch code?')
        assert 'Orion launch code is violet.' in calls[1]
        assert len(rt.timeline(namespace='alice'))==4
        assert not rt.retrieve_pack('Orion launch code',namespace='bob').items
    with Runtime(str(tmp_path)) as rt:
        rt.wrap(model,namespace='alice')('Recall Orion launch code')
        assert 'Orion launch code is violet.' in calls[2]
        assert len(calls)==3


def test_committed_memory_is_available_in_a_different_process(tmp_path):
    with Runtime(str(tmp_path)) as rt:
        eid=rt.append_event(namespace='alice',content='Orion launch code is violet.')
    program="""
import socket,sys,json
def denied(*a,**k):raise AssertionError('Network access')
socket.socket.connect=denied
from nsn import Runtime
with Runtime(sys.argv[1]) as rt:
 print(json.dumps([i.id for i in rt.retrieve_pack('Orion code',namespace='alice').items]))
"""
    result=subprocess.run([sys.executable,'-c',program,str(tmp_path)],capture_output=True,text=True,check=True)
    assert eid in json.loads(result.stdout)


def test_history_larger_than_model_context_recalls_old_identifier(tmp_path):
    with Runtime(str(tmp_path)) as rt:
        wanted=rt.append_event(namespace='agent',content='Kestrel-X7422 access token is violet-7812.')
        for i in range(750):
            rt.append_event(namespace='agent',content=f'Routine job {i} completed successfully with unrelated output.')
        rt.append_event(namespace='other',content='Kestrel-X7422 access token is private-gold.')
        assert sum(len(e['content']) for e in rt.timeline(namespace='agent',limit=1000))>2048
        pack=rt.retrieve_pack('Kestrel-X7422 access token',namespace='agent',
                              input_window_tokens=2048,existing_tokens=100,reserved_output_tokens=100,
                              tokenizer=lambda text:len(text.encode('utf-8')))
        assert wanted in {i.id for i in pack.items}
        assert 'violet-7812' in pack.rendered_text and 'private-gold' not in pack.rendered_text
        assert pack.token_count<=pack.token_budget<=int(.2*(2048-200))


@pytest.mark.parametrize('semantic',[False,True])
def test_historical_recall_excludes_future_events_before_limit(tmp_path,semantic):
    with Runtime(str(tmp_path),encoder=FastHashEncoder(dimension=16) if semantic else None) as rt:
        past=rt.append_event(content='Orion code violet',event_time='2025-01-01T00:00:00Z',observed_at='2025-01-01T00:00:00Z')
        for i in range(5):
            rt.append_event(content='Orion code gold',event_time='2026-01-01T00:00:00Z',observed_at=f'2026-01-01T00:00:0{i}Z')
        pack=rt.retrieve_pack('Orion code',as_of_time='2025-06-01T00:00:00Z',limit=1)
        assert {item.id for item in pack.items}=={past}
        assert 'gold' not in pack.rendered_text


def test_semantic_session_filter_applies_before_top_k(tmp_path):
    with Runtime(str(tmp_path),encoder=FastHashEncoder(dimension=16)) as rt:
        wanted=rt.append_event(session_id='wanted',content='Azure observatory')
        # These stronger vector matches must not consume the other session's k.
        for i in range(5):rt.append_event(session_id='other',content='Orion code',id=f'other-{i}')
        pack=rt.retrieve_pack('Orion code',session_id='wanted',limit=1)
        assert {item.id for item in pack.items}=={wanted}


@pytest.mark.parametrize('python_fallback',[False,True])
def test_dense_eligibility_keeps_scores_and_does_not_poison_cache(tmp_path,monkeypatch,python_fallback):
    import neurosleepnet.storage.compact_vector as compact
    if python_fallback:monkeypatch.setattr(compact,'np',None)
    with Runtime(str(tmp_path),encoder=FastHashEncoder(dimension=2)) as rt:
        store=rt.vector_store
        store.add_batch([('a',[1,0],'default','event'),('b',[0,1],'default','event'),('c',[.6,.8],'default','event')])
        assert store.search([1,0],limit=1)[0]['id']=='a'
        filtered=store.search([1,0],limit=1,eligible_ids={'c'})
        assert filtered[0]['id']=='c' and filtered[0]['score']==pytest.approx(.6)
        assert store.search([1,0],limit=1,eligible_ids=set())==[]
        assert store.search([1,0],limit=1)[0]['id']=='a'


def test_as_of_uses_event_time_offsets_and_separate_cache_entries(tmp_path):
    with Runtime(str(tmp_path)) as rt:
        past=rt.append_event(content='Orion code violet',event_time='2026-01-02T01:00:00+05:30')
        future=rt.append_event(content='Orion code gold',event_time='2026-01-01T22:00:00Z')
        assert {i.id for i in rt.retrieve_pack('Orion code').items}=={past,future}
        assert {i.id for i in rt.retrieve_pack('Orion code',as_of_time='2026-01-01T20:00:00Z').items}=={past}
        assert {i.id for i in rt.retrieve_pack('Orion code').items}=={past,future}


def test_encoder_failure_preserves_lexical_memory_and_recovers_index(tmp_path):
    class TransientEncoder(FastHashEncoder):
        broken=True
        def encode(self,text):
            if self.broken:raise RuntimeError('encoder unavailable')
            return super().encode(text)
    encoder=TransientEncoder(dimension=16)
    with Runtime(str(tmp_path),encoder=encoder) as rt:
        eid=rt.append_event(content='Orion code violet')
        pack=rt.retrieve_pack('Orion code')
        assert eid in {i.id for i in pack.items}
        assert 'encoder unavailable' in pack.diagnostic_explain['embedding_error']
        assert rt.vector_store.count('default')==0
        encoder.broken=False
        assert rt.process_index_jobs(worker_id='repair')==1
        assert rt.vector_store.count('default')==1
        assert rt.process_index_jobs(worker_id='repair')==0


def test_fact_correction_history_and_support_deletion(tmp_path):
    with Runtime(str(tmp_path)) as rt:
        first=rt.append_event(content='Orion port 8080',event_time='2025-01-01T00:00:00Z')
        old=rt.record_fact(subject='Orion',predicate='port',value=8080,supporting_event_ids=[first],valid_from='2025-01-01T00:00:00Z')
        second=rt.append_event(content='Orion port 9090',event_time='2026-01-01T00:00:00Z')
        new=rt.record_fact(subject='Orion',predicate='port',value=9090,supporting_event_ids=[second],valid_from='2026-01-01T00:00:00Z')
        facts=[i for i in rt.retrieve_pack('Orion port').items if i.kind=='fact']
        assert {i.id for i in facts}=={new.id}
        assert rt.facts.get_fact_as_of('Orion','port','2025-06-01T00:00:00Z','default').id==old.id
        rt.storage.delete_event(second,'default')
        assert new.id not in {i.id for i in rt.retrieve_pack('Orion port').items}
        assert rt.storage.get_event(first,'default') is not None


@pytest.mark.parametrize('operation',['replay','rebuild'])
def test_closed_runtime_cannot_mutate_derived_memory(tmp_path,operation):
    rt=Runtime(str(tmp_path),encoder=FastHashEncoder(dimension=16),defer_indexing=True)
    rt.append_event(content='Orion code violet')
    rt.close()
    with pytest.raises(RuntimeError,match='closed'):
        if operation=='replay':rt.process_index_jobs()
        else:rt.rebuild_vectors()


@pytest.mark.parametrize('operation',['relation','extract','pin'])
def test_closed_runtime_rejects_optional_memory_mutations(tmp_path,operation):
    rt=Runtime(str(tmp_path),graph=True)
    eid=rt.append_event(content='Alice works for Acme')
    rt.close()
    with pytest.raises(RuntimeError,match='closed'):
        if operation=='relation':rt.record_relation(subject='Alice',predicate='works_for',object='Acme',event_id=eid)
        elif operation=='extract':rt.extract_relations(eid)
        else:rt.pin(eid,source_type='event')


def test_namespace_backup_restore_and_delete_remain_scoped(tmp_path):
    with Runtime(str(tmp_path/'source')) as source:
        eid=source.append_event(namespace='alice',content='Orion code violet')
        snapshot=source.export_memory('alice')
    with Runtime(str(tmp_path/'target')) as target:
        other=target.append_event(namespace='bob',content='Orion code gold')
        target.import_memory(json.loads(json.dumps(snapshot)),namespace='alice')
        assert eid in {i.id for i in target.retrieve_pack('Orion code',namespace='alice').items}
        target.delete_namespace('alice')
        assert not target.retrieve_pack('Orion code',namespace='alice').items
        assert {i.id for i in target.retrieve_pack('Orion code',namespace='bob').items}=={other}


def test_real_onnx_session_and_history_scoping_without_network(tmp_path,monkeypatch):
    assets=Path(os.environ.get('NSN_ONNX_ASSETS','C:/Users/aviroop/AppData/Local/NSN/evaluation/minilm-onnx-1110a243'))
    if not assets.is_dir():pytest.skip('Prepared ONNX assets unavailable')
    pytest.importorskip('onnxruntime')
    import socket
    def denied(*a,**k):raise AssertionError('Unexpected network access')
    monkeypatch.setattr(socket.socket,'connect',denied)
    from neurosleepnet.embeddings.onnx_pooled import ONNXPooledEncoder
    with Runtime(str(tmp_path),encoder=ONNXPooledEncoder(assets)) as rt:
        for i in range(3):rt.append_event(session_id='other',content='Payment gateway production port 9090',event_time='2026-01-01T00:00:00Z')
        eid=rt.append_event(session_id='wanted',content='Payment gateway production port 8080',event_time='2025-01-01T00:00:00Z')
        pack=rt.retrieve_pack('Where does payment gateway listen?',session_id='wanted',as_of_time='2025-06-01T00:00:00Z',limit=1)
        assert {i.id for i in pack.items}=={eid}
        assert '8080' in pack.rendered_text and '9090' not in pack.rendered_text
