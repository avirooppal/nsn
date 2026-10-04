import sqlite3
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from nsn import Runtime


def test_runtime_close_does_not_close_another_owner_or_lose_committed_events(tmp_path):
    one=Runtime(str(tmp_path));two=Runtime(str(tmp_path))
    event=one.append_event(namespace='agent',content='gateway port 8080')
    assert one.storage._wal_anchor is not None
    one.close();one.close()
    assert one.storage._wal_anchor is None and two.storage._wal_anchor is not None
    with pytest.raises(RuntimeError):one.wrap(lambda prompt:prompt)
    assert two.storage.get_event(event,'agent')['content']=='gateway port 8080'
    two.append_event(namespace='agent',content='gateway host internal')
    two.close()
    restarted=Runtime(str(tmp_path))
    assert restarted.retrieve_pack('gateway port',namespace='agent').items
    assert not restarted.retrieve_pack('gateway port',namespace='other').items
    restarted.close()


def test_owned_connection_serializes_appends_and_recovers_after_rollback(tmp_path):
    rt=Runtime(str(tmp_path))
    try:
        with pytest.raises(KeyError):
            rt.storage.append_event(namespace='agent',id='rolled-back',content='invalid',outbox_jobs=[{}])
        assert rt.storage.get_event('rolled-back','agent') is None
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids=list(pool.map(lambda i:rt.append_event(namespace='agent',content=f'gateway port {i}'),range(32)))
        assert len(set(ids))==32
        assert all(rt.storage.get_event(eid,'agent') for eid in ids)
        with sqlite3.connect(rt.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM events_fts WHERE namespace='agent'").fetchone()[0]==32
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    finally:rt.close()


def test_graph_read_restores_shared_connection_transaction_and_progress_handler(tmp_path):
    rt=Runtime(str(tmp_path),graph=True)
    try:
        source=rt.append_event(content='Alice works for Acme.')
        rt.extract_relations(source)
        assert rt.relationships.traverse('Alice',max_tokens=1000).paths
        conn=rt.storage._wal_anchor
        assert conn.row_factory is None and not conn.in_transaction
        # A lingering traversal deadline must not interrupt later SQLite work.
        assert conn.execute('WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<1000) SELECT sum(n) FROM x').fetchone()[0]==500500
        later=rt.append_event(content='Gateway port is 8080.')
        assert rt.storage.get_event(later,'default')
    finally:rt.close()


def test_concurrent_runtime_owners_share_authoritative_idempotency_result(tmp_path):
    owners=[Runtime(str(tmp_path)),Runtime(str(tmp_path))]
    barrier=threading.Barrier(2)
    def writer(owner):
        ids=[]
        for i in range(12):
            barrier.wait(timeout=10)
            ids.append(owner.storage.append_event(namespace='agent',content='gateway port',idempotency_key=str(i)))
        return ids
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first,second=[future.result() for future in [pool.submit(writer,owner) for owner in owners]]
        assert first==second and len(set(first))==12
        with sqlite3.connect(owners[0].db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM events WHERE namespace='agent'").fetchone()[0]==12
    finally:
        for owner in owners:owner.close()
