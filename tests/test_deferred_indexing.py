import sqlite3
from nsn import Runtime
from neurosleepnet.embeddings.pooled import FastHashEncoder


class SpyEncoder(FastHashEncoder):
    def __init__(self):
        super().__init__(dimension=16)
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return super().encode(text)


def test_deferred_events_and_facts_survive_restart_without_inline_encoding(tmp_path):
    encoder = SpyEncoder()
    with Runtime(tmp_path, encoder=encoder, defer_indexing=True) as rt:
        eid = rt.append_event(content='Gateway production port 8080.', idempotency_key='turn-1')
        assert rt.append_event(content='Retry altered content', idempotency_key='turn-1') == eid
        fact = rt.record_fact(subject='gateway', predicate='port', value=8080, supporting_event_ids=[eid])
        assert encoder.calls == 0
        assert rt.vector_store.count('default') == 0
        with sqlite3.connect(rt.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type='vector_index' AND status='pending'").fetchone()[0] == 2
        assert eid in {item.id for item in rt.retrieve_pack('Gateway port').items}
    with Runtime(tmp_path, encoder=encoder, defer_indexing=True) as rt:
        assert rt.process_index_jobs(worker_id='recovery') == 2
        assert rt.vector_store.count('default') == 2
        assert rt.process_index_jobs(worker_id='recovery') == 0


def test_deferred_deleted_and_superseded_targets_never_restore_vectors(tmp_path):
    with Runtime(tmp_path, encoder=SpyEncoder(), defer_indexing=True) as rt:
        deleted = rt.append_event(content='Deleted secret')
        first = rt.record_fact(subject='gateway', predicate='port', value=8080,
                               valid_from='2025-01-01T00:00:00Z')
        latest = rt.record_fact(subject='gateway', predicate='port', value=9090,
                                valid_from='2026-01-01T00:00:00Z')
        rt.storage.delete_event(deleted, 'default')
        rt.process_index_jobs(worker_id='recovery')
        ids = {row['id'] for row in rt.vector_store.search(rt.encoder.encode('gateway port'), limit=10)}
        assert latest.id in ids
        assert deleted not in ids and first.id not in ids


def test_default_semantic_writes_still_index_inline(tmp_path):
    encoder = SpyEncoder()
    with Runtime(tmp_path, encoder=encoder) as rt:
        rt.append_event(content='Gateway production port 8080.')
        assert encoder.calls == 1
        assert rt.vector_store.count('default') == 1
