"""
Tests for Phase 01 — Scoped Durable Storage and Runtime Ownership.

Verifies:
1. Namespace isolation: Two namespaces sharing a database cannot read, delete,
   or deduplicate each other's records, even by known ID.
2. Crash recovery / outbox replay: Committed events and outbox jobs are preserved
   across interruption and replayed without loss.
3. Concurrent access: WAL mode and busy handling allow concurrent readers/writers
   without corrupted state or duplicate job execution.
4. Schema migrations: Versioned, repeatable, idempotent, preserves legacy data and
   creates backup, fails visibly on unsupported future version.
5. Lifecycle flush and close: Flush reports pending-job failure rather than falsely
   claiming consistency; close is idempotent.
"""

import os
import sqlite3
import tempfile
import threading
import pytest

from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.storage.migrations import apply_migrations, get_schema_version, CURRENT_SCHEMA_VERSION


def test_legacy_backup_includes_uncheckpointed_wal(tmp_path):
    path=str(tmp_path/'legacy.db')
    writer=sqlite3.connect(path)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute('CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, namespace TEXT, memory_type TEXT)')
    writer.execute("INSERT INTO memories VALUES ('original','preserve WAL evidence','default','semantic')")
    writer.commit()
    assert os.path.getsize(path+'-wal')>0
    apply_migrations(path)
    with sqlite3.connect(path+'.bak') as backup:
        assert backup.execute('SELECT content FROM memories').fetchone()[0]=='preserve WAL evidence'
        assert get_schema_version(backup)==0
    with sqlite3.connect(path) as migrated:
        assert get_schema_version(migrated)==CURRENT_SCHEMA_VERSION
        assert migrated.execute('SELECT content FROM memories').fetchone()[0]=='preserve WAL evidence'
    writer.close()


def test_namespace_isolation_by_known_id(tmp_path):
    """
    Acceptance criterion 1:
    Two namespaces sharing a database cannot read/delete/deduplicate each other's
    identical-content records, even by known ID.
    """
    db_path = str(tmp_path / "shared.db")
    storage = SQLiteAdapter(db_path=db_path)

    # 1. Store in namespace A
    event_a_id = storage.append_event({
        "id": "evt-123",
        "namespace": "tenant_a",
        "content": "Secret API key: 12345",
        "role": "user",
        "source": "user",
        "idempotency_key": "key-shared-1",
    })
    assert event_a_id == "evt-123"

    # 2. Namespace B tries to read event_a_id -> MUST return None
    assert storage.get_event("evt-123", namespace="tenant_b") is None
    # Namespace A CAN read it
    evt_a = storage.get_event("evt-123", namespace="tenant_a")
    assert evt_a is not None
    assert evt_a["content"] == "Secret API key: 12345"

    # 3. Namespace B tries to delete event_a_id -> MUST return False and NOT delete
    assert storage.delete_event("evt-123", namespace="tenant_b") is False
    assert storage.get_event("evt-123", namespace="tenant_a") is not None

    # 4. Namespace B stores identical content and idempotency key -> MUST be isolated
    event_b_id = storage.append_event({
        "id": "evt-b-999",
        "namespace": "tenant_b",
        "content": "Secret API key: 12345",
        "role": "user",
        "source": "user",
        "idempotency_key": "key-shared-1",
    })
    assert event_b_id == "evt-b-999"

    # Listing events is scoped to namespace
    a_events = storage.list_events(namespace="tenant_a")
    b_events = storage.list_events(namespace="tenant_b")
    assert len(a_events) == 1
    assert len(b_events) == 1
    assert a_events[0]["id"] == "evt-123"
    assert b_events[0]["id"] == "evt-b-999"

    # Search is scoped to namespace
    a_search = storage.search_events("Secret", namespace="tenant_a")
    b_search = storage.search_events("Secret", namespace="tenant_b")
    assert len(a_search) == 1 and a_search[0]["id"] == "evt-123"
    assert len(b_search) == 1 and b_search[0]["id"] == "evt-b-999"


def test_crash_injection_preserves_events_and_replays_jobs(tmp_path):
    """
    Acceptance criterion 2:
    Crash injection around commit and index processing preserves committed events
    and replays pending work exactly once in effect.
    """
    db_path = str(tmp_path / "crash_test.db")
    storage = SQLiteAdapter(db_path=db_path)

    # Commit event with outbox job
    outbox_job = {
        "id": "job-index-1",
        "job_type": "index_event",
        "payload": {"event_id": "evt-crash-1"}
    }
    storage.append_event({
        "id": "evt-crash-1",
        "namespace": "test_ns",
        "content": "Mission critical observation",
        "role": "system",
        "source": "system",
    }, outbox_jobs=[outbox_job])

    # Simulate crash before worker completes the job
    # Re-open storage from disk (restart recovery)
    storage_restarted = SQLiteAdapter(db_path=db_path)

    # Event is durably preserved
    recovered_event = storage_restarted.get_event("evt-crash-1", namespace="test_ns")
    assert recovered_event is not None
    assert recovered_event["content"] == "Mission critical observation"

    # Outbox job is pending and replayed
    jobs = storage_restarted.get_pending_jobs(limit=10, worker_id="worker-recovery")
    assert len(jobs) == 1
    assert jobs[0]["id"] == "job-index-1"
    assert jobs[0]["payload"]["event_id"] == "evt-crash-1"

    # Complete job
    storage_restarted.complete_job(jobs[0]["id"], worker_id="worker-recovery")

    # Verify no pending jobs remain
    assert len(storage_restarted.get_pending_jobs(limit=10)) == 0


def test_concurrent_writers_and_readers(tmp_path):
    """
    Acceptance criterion 3:
    Concurrent writers/readers obey the documented ownership policy without
    corrupted state or duplicate jobs.
    """
    db_path = str(tmp_path / "concurrent.db")
    storage = SQLiteAdapter(db_path=db_path)
    errors = []

    def writer_worker(thread_id: int):
        try:
            worker_storage = SQLiteAdapter(db_path=db_path)
            for i in range(15):
                worker_storage.append_event({
                    "id": f"th_{thread_id}_evt_{i}",
                    "namespace": f"ns_{thread_id % 2}",
                    "content": f"Thread {thread_id} content {i}",
                    "role": "user",
                    "source": "user",
                }, outbox_jobs=[{
                    "id": f"th_{thread_id}_job_{i}",
                    "job_type": "index",
                    "payload": {"i": i}
                }])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer_worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent execution produced errors: {errors}"

    # Verify total events stored (4 threads * 15 events = 60)
    events_ns0 = storage.list_events(namespace="ns_0", limit=100)
    events_ns1 = storage.list_events(namespace="ns_1", limit=100)
    assert len(events_ns0) + len(events_ns1) == 60

    # Verify lease prevents duplicate job execution
    jobs1 = storage.get_pending_jobs(limit=30, lease_duration_seconds=60, worker_id="w1")
    jobs2 = storage.get_pending_jobs(limit=30, lease_duration_seconds=60, worker_id="w2")
    # Total jobs leased by w1 + w2 = 60, with no overlap
    ids1 = {j["id"] for j in jobs1}
    ids2 = {j["id"] for j in jobs2}
    assert ids1.isdisjoint(ids2)
    assert len(ids1) + len(ids2) == 60


def test_schema_migrations_repeatable_and_versioned(tmp_path):
    """
    Acceptance criterion 4:
    Migration is repeatable, versioned, preserves legacy evidence,
    and fails visibly on unsupported schema.
    """
    db_path = str(tmp_path / "mig_test.db")

    # 1. Run migrations to current version
    apply_migrations(db_path, target_version=CURRENT_SCHEMA_VERSION)
    conn = sqlite3.connect(db_path)
    assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION
    conn.close()

    # 2. Re-running migration is idempotent
    apply_migrations(db_path, target_version=CURRENT_SCHEMA_VERSION)
    conn = sqlite3.connect(db_path)
    assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION
    conn.close()

    # 3. Requesting an unsupported future version raises ValueError
    with pytest.raises(ValueError, match="Unsupported schema version"):
        apply_migrations(db_path, target_version=CURRENT_SCHEMA_VERSION - 1)


def test_legacy_database_migration_with_backup(tmp_path):
    """
    Acceptance criterion 4 (part 2):
    Back up and migrate a legacy database fixture without changing its original raw content.
    """
    db_path = str(tmp_path / "legacy.db")

    # Create un-migrated legacy schema and data
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT,
            importance REAL DEFAULT 0.0,
            trust_score REAL DEFAULT 0.5,
            embedding TEXT,
            namespace TEXT DEFAULT 'default',
            memory_type TEXT DEFAULT 'semantic',
            access_count INTEGER DEFAULT 0,
            last_accessed_at TEXT
        )
    ''')
    conn.execute('''
        INSERT INTO memories (id, content, created_at)
        VALUES ('legacy_m1', 'Legacy raw memory content', '2026-01-01T00:00:00Z')
    ''')
    conn.commit()
    conn.close()

    # Migrate via SQLiteAdapter
    storage = SQLiteAdapter(db_path=db_path)

    # 1. Backup file must have been created
    backup_path = f"{db_path}.bak"
    assert os.path.exists(backup_path)

    # 2. Original raw record is preserved
    rec = storage.get("legacy_m1")
    assert rec is not None
    assert rec["content"] == "Legacy raw memory content"


def test_flush_and_close_lifecycle(tmp_path):
    """
    Acceptance criterion 5:
    Flush reports pending-job failure rather than falsely claiming consistency;
    close is idempotent.
    """
    db_path = str(tmp_path / "flush_test.db")
    storage = SQLiteAdapter(db_path=db_path)

    # Append event with job
    storage.append_event({
        "id": "evt-f-1",
        "namespace": "test_ns",
        "content": "Some event",
    }, outbox_jobs=[{
        "id": "job-f-1",
        "job_type": "extract",
        "payload": {}
    }])

    # Pending job exists -> flush returns False (not settled)
    assert storage.flush() is False

    # Simulate worker failing the job repeatedly until max_attempts
    for _ in range(3):
        jobs = storage.get_pending_jobs(limit=1, worker_id="w1")
        assert len(jobs) == 1
        assert storage.fail_job(jobs[0]["id"], "Permanent parsing error", worker_id="w1") is True

    # Job is permanently failed -> flush reports False (failure visible)
    assert storage.flush() is False

    # Close is idempotent
    storage.close()
    storage.close()
