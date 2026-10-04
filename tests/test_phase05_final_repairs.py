import sqlite3
import threading

import pytest
from nsn.runtime import Runtime
from neurosleepnet.embeddings.pooled import FastHashEncoder
from neurosleepnet.storage.compact_vector import CompactVectorStore


def test_target_validation_holds_writer_lock_and_deletion_invalidates_cache(tmp_path, monkeypatch):
    encoder = FastHashEncoder(dimension=16)
    rt = Runtime(str(tmp_path), encoder=encoder)
    fact = rt.record_fact(subject="service", predicate="status", value="ready")
    reader = CompactVectorStore.from_encoder(rt.db_path, encoder)
    assert reader.search(encoder.encode("service ready"))
    entered, release = threading.Event(), threading.Event()
    original = rt.vector_store._get_connection

    def traced():
        conn = original()
        def trace(sql):
            if sql.startswith("SELECT status FROM facts"):
                # BEGIN IMMEDIATE must already hold the write reservation.
                assert conn.in_transaction
                entered.set()
                assert release.wait(5)
        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(rt.vector_store, "_get_connection", traced)
    errors = []
    def add():
        try:
            rt.vector_store.add(fact.id, encoder.encode("service ready"))
        except Exception as exc:
            errors.append(exc)
    writer = threading.Thread(target=add)
    writer.start()
    assert entered.wait(5)
    deleted = threading.Event()
    def delete():
        conn = sqlite3.connect(rt.db_path)
        with conn:
            conn.execute("UPDATE facts SET status='deleted' WHERE id=?", (fact.id,))
        conn.close()
        deleted.set()
    deleter = threading.Thread(target=delete)
    deleter.start()
    assert not deleted.wait(0.05)
    release.set()
    writer.join(5)
    deleter.join(5)
    assert not errors and deleted.is_set()
    assert reader.search(encoder.encode("service ready")) == []
    rt.close()


def test_rebuild_empty_invalidates_cache_and_failed_encode_preserves_index(tmp_path):
    encoder = FastHashEncoder(dimension=16)
    rt = Runtime(str(tmp_path), encoder=encoder)
    eid = rt.append_event(content="stored event", role="user")
    assert rt.vector_store.search(encoder.encode("stored event"))
    class Broken:
        def encode(self, text):
            raise RuntimeError("broken")
    with pytest.raises(RuntimeError, match="broken"):
        rt.vector_store.rebuild(rt.db_path, Broken(), "default")
    assert rt.vector_store.count() == 1
    rt.storage.delete_event(eid, "default")
    assert rt.rebuild_vectors() == 0
    assert rt.vector_store.search(encoder.encode("stored event")) == []
    rt.close()
