import sqlite3
import json
import re
import uuid
import threading
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from .base import StorageAdapter

class EventID(str):
    """Event ID string carrying insertion status for idempotency preservation."""
    def __new__(cls, value: str, inserted: bool = True):
        obj = str.__new__(cls, value)
        obj.inserted = inserted
        return obj

    def __init__(self, value: str, inserted: bool = True):
        self.inserted = inserted


class SQLiteAdapter(StorageAdapter):
    """
    SQLite implementation of StorageAdapter.
    """
    def __init__(self, db_path: str = "neurosleepnet.db", keep_wal_open: bool = False):
        self.db_path = db_path
        self._initialize_db()
        # Keep this runtime attached to WAL. Closing the last connection after
        # every append otherwise checkpoints/deletes/reopens the WAL repeatedly.
        self._wal_anchor = None
        self._append_lock = threading.RLock()
        if keep_wal_open:
            self._wal_anchor = sqlite3.connect(db_path, check_same_thread=False)
            self._wal_anchor.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()

    @contextmanager
    def _runtime_connection(self):
        # Serialize use of the owner's connection, preserving SQLite FULL sync
        # while avoiding repeated schema parsing on short foreground writes.
        with self._append_lock:
            conn = self._wal_anchor or sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                if conn is not self._wal_anchor:
                    conn.close()

    def _initialize_db(self):
        from .migrations import apply_migrations
        apply_migrations(self.db_path)

    def store_graph_node(self, node_id: str, label: str, name: str, properties: str, created_at: str, namespace: str = "default"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO graph_nodes (id, label, name, properties, created_at, namespace)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (node_id, label, name, properties, created_at, namespace))
        conn.commit()
        conn.close()

    def store_graph_edge(self, edge_id: str, source_id: str, target_id: str, relation: str, properties: str, created_at: str, namespace: str = "default"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO graph_edges (id, source_id, target_id, relation, properties, created_at, namespace)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (edge_id, source_id, target_id, relation, properties, created_at, namespace))
        conn.commit()
        conn.close()

    def get_graph_node(self, node_id: str, namespace: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if namespace:
            cursor.execute('SELECT id, label, name, properties, created_at, namespace FROM graph_nodes WHERE id = ? AND namespace = ?', (node_id, namespace))
        else:
            cursor.execute('SELECT id, label, name, properties, created_at, namespace FROM graph_nodes WHERE id = ?', (node_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"id": row[0], "label": row[1], "name": row[2], "properties": json.loads(row[3]) if row[3] else {}, "created_at": row[4], "namespace": row[5] if len(row) > 5 else "default"}
        return None

    def query_graph(self, node_name: str, namespace: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if namespace:
            cursor.execute('SELECT id, label, properties FROM graph_nodes WHERE name = ? AND namespace = ?', (node_name, namespace))
        else:
            cursor.execute('SELECT id, label, properties FROM graph_nodes WHERE name = ?', (node_name,))
        node_row = cursor.fetchone()

        if not node_row:
            conn.close()
            return None

        node_id = node_row[0]
        result = {
            "node": {"id": node_id, "name": node_name, "label": node_row[1], "properties": json.loads(node_row[2]) if node_row[2] else {}},
            "edges": []
        }

        if namespace:
            cursor.execute('''
                SELECT e.relation, e.properties, n.id, n.name, n.label, n.properties
                FROM graph_edges e
                JOIN graph_nodes n ON e.target_id = n.id
                WHERE e.source_id = ? AND e.namespace = ? AND n.namespace = ?
            ''', (node_id, namespace, namespace))
        else:
            cursor.execute('''
                SELECT e.relation, e.properties, n.id, n.name, n.label, n.properties
                FROM graph_edges e
                JOIN graph_nodes n ON e.target_id = n.id
                WHERE e.source_id = ?
            ''', (node_id,))

        edge_rows = cursor.fetchall()
        for row in edge_rows:
            result["edges"].append({
                "relation": row[0],
                "edge_properties": json.loads(row[1]) if row[1] else {},
                "target": {
                    "id": row[2],
                    "name": row[3],
                    "label": row[4],
                    "properties": json.loads(row[5]) if row[5] else {}
                }
            })

        conn.close()
        return result

    def store(self, memory_id: str, content: str, created_at: str, metadata: str = "{}", importance: float = 0.0, trust_score: float = 0.5, embedding: str = "[]", namespace: str = "default", memory_type: str = "semantic"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO memories (id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (memory_id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type))

        cursor.execute('''
            INSERT OR REPLACE INTO memories_fts (id, content, namespace)
            VALUES (?, ?, ?)
        ''', (memory_id, content, namespace))

        conn.commit()
        conn.close()

    def get(self, memory_id: str, namespace: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if namespace:
            cursor.execute('''
                SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories WHERE id = ? AND namespace = ?
            ''', (memory_id, namespace))
        else:
            cursor.execute('''
                SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories WHERE id = ?
            ''', (memory_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10]
            }
        return None

    def delete(self, memory_id: str, namespace: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if namespace:
            cursor.execute('DELETE FROM memories WHERE id = ? AND namespace = ?', (memory_id, namespace))
            cursor.execute('DELETE FROM memories_fts WHERE id = ? AND namespace = ?', (memory_id, namespace))
        else:
            cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            cursor.execute('DELETE FROM memories_fts WHERE id = ?', (memory_id,))
        conn.commit()
        conn.close()

    def list(self, namespace: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if namespace:
            cursor.execute('''
                SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories WHERE namespace = ?
            ''', (namespace,))
        else:
            cursor.execute('''
                SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories
            ''')
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10]
            })
        return records

    def search_keyword(self, query: str, limit: int = 5, namespace: str = None):
        if not query or not query.strip():
            return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if caller explicitly requested phrase match
        q_strip = query.strip()
        if q_strip.startswith('"') and q_strip.endswith('"') and len(q_strip) > 2:
            safe_query = '"' + q_strip[1:-1].replace('"', '""') + '"'
        else:
            tokens = re.findall(r'[\w]+', q_strip)
            if not tokens:
                conn.close()
                return []
            safe_query = " OR ".join(f'"{t}"' for t in tokens)

        try:
            if namespace:
                cursor.execute('''
                    SELECT m.id, m.content, m.created_at, m.metadata, m.importance, m.trust_score, m.embedding, m.namespace, m.memory_type, m.access_count, m.last_accessed_at
                    FROM memories_fts fts
                    JOIN memories m ON fts.id = m.id
                    WHERE memories_fts MATCH ? AND fts.namespace = ?
                    ORDER BY bm25(memories_fts) ASC, m.created_at DESC
                    LIMIT ?
                ''', (safe_query, namespace, limit))
            else:
                cursor.execute('''
                    SELECT m.id, m.content, m.created_at, m.metadata, m.importance, m.trust_score, m.embedding, m.namespace, m.memory_type, m.access_count, m.last_accessed_at
                    FROM memories_fts fts
                    JOIN memories m ON fts.id = m.id
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25(memories_fts) ASC, m.created_at DESC
                    LIMIT ?
                ''', (safe_query, limit))
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10],
                "score": 1.0
            })
        return records

    def increment_access(self, memory_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute('''
            UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?
        ''', (now, memory_id))
        conn.commit()
        conn.close()

    def list_namespace(self, namespace: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at
            FROM memories
            WHERE namespace = ?
        ''', (namespace,))
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10]
            })
        return records

    def timeline(self, namespace: str, memory_type: str = None, limit: int = 20, ascending: bool = True) -> list:
        """Returns chronologically ordered memories, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = (
            "SELECT id, content, created_at, metadata, importance, trust_score, "
            "namespace, memory_type, access_count, last_accessed_at "
            "FROM memories WHERE namespace = ?"
        )
        params = [namespace]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        order = "ASC" if ascending else "DESC"
        query += f" ORDER BY created_at {order} LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": float(row[4]),
                "trust_score": float(row[5]),
                "namespace": row[6],
                "memory_type": row[7],
                "access_count": int(row[8]),
                "last_accessed_at": row[9],
            })
        return records

    def append_event(self, event: dict = None, outbox_jobs: list = None, **kwargs) -> str:
        """
        Atomically append an event and optional outbox jobs within a single transaction.
        Enforces namespace isolation and idempotency.
        """
        if event is None:
            event = kwargs
        elif kwargs:
            event = {**event, **kwargs}

        namespace = event.get("namespace")
        if not namespace:
            raise ValueError("Event must specify a namespace.")

        event_id = event.get("id") or str(uuid.uuid4())
        session_id = event.get("session_id")
        role = event.get("role", "user")
        content = event.get("content", "")
        source_identity = event.get("source_identity", event.get("source", "user"))
        observed_at = event.get("observed_at") or datetime.now(timezone.utc).isoformat()
        event_time = event.get("event_time")
        turn_status = event.get("turn_status", "completed")
        idempotency_key = event.get("idempotency_key")
        metadata_val = event.get("metadata", {})
        metadata_str = json.dumps(metadata_val) if isinstance(metadata_val, dict) else (metadata_val or "{}")
        retention_policy = event.get("retention_policy", "permanent")

        with self._runtime_connection() as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            cursor = conn.cursor()

            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if idempotency_key:
                    cursor.execute(
                        "SELECT id FROM events WHERE namespace = ? AND idempotency_key = ?",
                        (namespace, idempotency_key)
                    )
                    row = cursor.fetchone()
                    if row:
                        return EventID(row[0], inserted=False)

                cursor.execute('''
                    INSERT INTO events (
                        id, namespace, session_id, role, content, source_identity,
                        observed_at, event_time, turn_status, idempotency_key, metadata, retention_policy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_id, namespace, session_id, role, content, source_identity,
                    observed_at, event_time, turn_status, idempotency_key, metadata_str, retention_policy
                ))

                cursor.execute('''
                    INSERT INTO events_fts (id, content, namespace)
                    VALUES (?, ?, ?)
                ''', (event_id, content, namespace))

                if outbox_jobs:
                    now_str = datetime.now(timezone.utc).isoformat()
                    for job in outbox_jobs:
                        job_id = job.get("id") or str(uuid.uuid4())
                        job_type = job["job_type"]
                        payload_val = job.get("payload", {})
                        payload_str = json.dumps(payload_val) if isinstance(payload_val, dict) else (payload_val or "{}")
                        cursor.execute('''
                            INSERT INTO jobs (
                                id, namespace, job_type, payload, status, attempts, max_attempts, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, 'pending', 0, 3, ?, ?)
                        ''', (job_id, namespace, job_type, payload_str, now_str, now_str))

            return EventID(event_id, inserted=True)

    def get_event(self, event_id: str, namespace: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, namespace, session_id, role, content, source_identity,
                   observed_at, event_time, turn_status, idempotency_key, metadata, retention_policy
            FROM events WHERE id = ? AND namespace = ?
        ''', (event_id, namespace))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row[0],
            "namespace": row[1],
            "session_id": row[2],
            "role": row[3],
            "content": row[4],
            "source_identity": row[5],
            "observed_at": row[6],
            "event_time": row[7],
            "turn_status": row[8],
            "idempotency_key": row[9],
            "metadata": json.loads(row[10]) if row[10] else {},
            "retention_policy": row[11],
        }

    def delete_event(self, event_id: str, namespace: str) -> bool:
        from neurosleepnet.memory.facts import FactManager
        try:
            fm = FactManager(self)
            fm.invalidate_event_support(event_id, namespace)
        except Exception:
            pass

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        with conn:
            cursor.execute("DELETE FROM events WHERE id = ? AND namespace = ?", (event_id, namespace))
            affected = cursor.rowcount
            cursor.execute("DELETE FROM events_fts WHERE id = ? AND namespace = ?", (event_id, namespace))
        conn.close()
        return affected > 0

    def list_events(self, namespace: str, limit: int = 100, session_id: str = None) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if session_id:
            cursor.execute('''
                SELECT id, namespace, session_id, role, content, source_identity,
                       observed_at, event_time, turn_status, idempotency_key, metadata, retention_policy
                FROM events WHERE namespace = ? AND session_id = ?
                ORDER BY observed_at ASC LIMIT ?
            ''', (namespace, session_id, limit))
        else:
            cursor.execute('''
                SELECT id, namespace, session_id, role, content, source_identity,
                       observed_at, event_time, turn_status, idempotency_key, metadata, retention_policy
                FROM events WHERE namespace = ?
                ORDER BY observed_at ASC LIMIT ?
            ''', (namespace, limit))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "namespace": r[1],
                "session_id": r[2],
                "role": r[3],
                "content": r[4],
                "source_identity": r[5],
                "observed_at": r[6],
                "event_time": r[7],
                "turn_status": r[8],
                "idempotency_key": r[9],
                "metadata": json.loads(r[10]) if r[10] else {},
                "retention_policy": r[11],
            }
            for r in rows
        ]

    def search_events(self, query: str, namespace: str, limit: int = 10) -> list:
        if not query or not query.strip():
            return []
        q_strip = query.strip()
        if q_strip.startswith('"') and q_strip.endswith('"') and len(q_strip) > 2:
            safe_query = '"' + q_strip[1:-1].replace('"', '""') + '"'
        else:
            tokens = re.findall(r'[\w]+', q_strip)
            if not tokens:
                return []
            safe_query = " OR ".join(f'"{t}"' for t in tokens)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT e.id, e.namespace, e.session_id, e.role, e.content, e.source_identity,
                       e.observed_at, e.event_time, e.turn_status, e.metadata
                FROM events_fts fts
                JOIN events e ON fts.id = e.id
                WHERE events_fts MATCH ? AND fts.namespace = ?
                ORDER BY bm25(events_fts) ASC, e.observed_at DESC
                LIMIT ?
            ''', (safe_query, namespace, limit))
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()

        return [
            {
                "id": r[0],
                "namespace": r[1],
                "session_id": r[2],
                "role": r[3],
                "content": r[4],
                "source_identity": r[5],
                "observed_at": r[6],
                "event_time": r[7],
                "turn_status": r[8],
                "metadata": json.loads(r[9]) if r[9] else {},
            }
            for r in rows
        ]

    def enqueue_job(self, job_type: str, payload: dict, namespace: str = "default", max_attempts: int = 3) -> str:
        """Enqueue a durable background job for asynchronous or recoverable execution."""
        job_id = str(uuid.uuid4())
        now_str = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(payload)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        with conn:
            conn.execute('''
                INSERT INTO jobs (id, namespace, job_type, payload, status, attempts, max_attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)
            ''', (job_id, namespace, job_type, payload_str, max_attempts, now_str, now_str))
        conn.close()
        return job_id

    def get_pending_jobs(
        self,
        limit: int = 10,
        lease_duration_seconds: int = 30,
        worker_id: str = "default",
        job_type: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> list:
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        lease_expires = (now + timedelta(seconds=lease_duration_seconds)).isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()

        with conn:
            if namespace is not None:
                cursor.execute('''
                    SELECT id, namespace, job_type, payload, attempts, max_attempts FROM jobs
                    WHERE (status='pending' OR (status='processing' AND lease_expires_at < ?))
                      AND namespace=? AND (? IS NULL OR job_type=?) ORDER BY created_at ASC LIMIT ?
                ''', (now_str, namespace, job_type, job_type, limit))
            elif job_type:
                cursor.execute('''
                    SELECT id, namespace, job_type, payload, attempts, max_attempts
                    FROM jobs
                    WHERE (status = 'pending'
                       OR (status = 'processing' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?))
                      AND job_type = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                ''', (now_str, job_type, limit))
            else:
                cursor.execute('''
                    SELECT id, namespace, job_type, payload, attempts, max_attempts
                    FROM jobs
                    WHERE status = 'pending'
                       OR (status = 'processing' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
                    ORDER BY created_at ASC
                    LIMIT ?
                ''', (now_str, limit))
            rows = cursor.fetchall()

            leased_jobs = []
            for r in rows:
                job_id = r[0]
                # Atomic conditional claim: ensure state hasn't changed since candidate selection
                cursor.execute('''
                    UPDATE jobs
                    SET status = 'processing', leased_by = ?, lease_expires_at = ?, updated_at = ?
                    WHERE id = ?
                      AND (status = 'pending'
                           OR (status = 'processing' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?))
                ''', (worker_id, lease_expires, now_str, job_id, now_str))
                if cursor.rowcount > 0:
                    leased_jobs.append({
                        "id": job_id,
                        "namespace": r[1],
                        "job_type": r[2],
                        "payload": json.loads(r[3]) if r[3] else {},
                        "attempts": r[4],
                        "max_attempts": r[5],
                    })

        conn.close()
        return leased_jobs

    def list_jobs(self, status: Optional[str] = None) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if status:
            cursor.execute('''
                SELECT id, namespace, job_type, payload, status, attempts, max_attempts, last_error
                FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
            ''', (status,))
        else:
            cursor.execute('''
                SELECT id, namespace, job_type, payload, status, attempts, max_attempts, last_error
                FROM jobs
                ORDER BY created_at ASC
            ''')
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "namespace": r[1],
                "job_type": r[2],
                "payload": json.loads(r[3]) if r[3] else {},
                "status": r[4],
                "attempts": r[5],
                "max_attempts": r[6],
                "last_error": r[7],
            }
            for r in rows
        ]

    def complete_job(self, job_id: str, worker_id: Optional[str] = None) -> bool:
        now_str = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        with conn:
            cursor = conn.cursor()
            if worker_id:
                # Require lease ownership: job must be processing, leased by worker_id, and unexpired
                cursor.execute('''
                    UPDATE jobs
                    SET status = 'completed', lease_expires_at = NULL, leased_by = NULL, updated_at = ?
                    WHERE id = ? AND status = 'processing' AND leased_by = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at >= ?)
                ''', (now_str, job_id, worker_id, now_str))
            else:
                # Inline completion: only pending jobs; must NOT override another worker's lease
                cursor.execute('''
                    UPDATE jobs
                    SET status = 'completed', lease_expires_at = NULL, leased_by = NULL, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                ''', (now_str, job_id))
            updated = cursor.rowcount > 0
        conn.close()
        return updated

    def fail_job(self, job_id: str, error: str, worker_id: Optional[str] = None) -> bool:
        now_str = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        with conn:
            cursor = conn.cursor()
            if worker_id:
                # Require lease ownership: job must be processing, leased by worker_id, and unexpired
                cursor.execute('''
                    SELECT attempts, max_attempts, payload FROM jobs
                    WHERE id = ? AND status = 'processing' AND leased_by = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at >= ?)
                ''', (job_id, worker_id, now_str))
            else:
                # Inline failure: only for pending jobs, must NOT override another worker's lease
                cursor.execute("SELECT attempts, max_attempts, payload FROM jobs WHERE id = ? AND status = 'pending'", (job_id,))
            row = cursor.fetchone()
            if not row:
                return False
            attempts = row[0] + 1
            max_attempts = row[1]
            payload_str = row[2] or "{}"
            try:
                p_data = json.loads(payload_str)
            except Exception:
                p_data = {}
            p_data["error"] = str(error)
            new_payload = json.dumps(p_data)
            status = 'failed' if attempts >= max_attempts else 'pending'
            if worker_id:
                cursor.execute('''
                    UPDATE jobs
                    SET status = ?, attempts = ?, last_error = ?, payload = ?, lease_expires_at = NULL, leased_by = NULL, updated_at = ?
                    WHERE id = ? AND status = 'processing' AND leased_by = ?
                      AND (lease_expires_at IS NULL OR lease_expires_at >= ?)
                ''', (status, attempts, str(error), new_payload, now_str, job_id, worker_id, now_str))
            else:
                cursor.execute('''
                    UPDATE jobs
                    SET status = ?, attempts = ?, last_error = ?, payload = ?, lease_expires_at = NULL, leased_by = NULL, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                ''', (status, attempts, str(error), new_payload, now_str, job_id))
            updated = cursor.rowcount > 0
        conn.close()
        return updated

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed'")
        failed_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'processing')")
        pending_count = cursor.fetchone()[0]
        conn.close()

        if failed_count > 0:
            return False
        return pending_count == 0

    def close(self):
        """Idempotent close."""
        with self._append_lock:
            anchor = self._wal_anchor
            if anchor is not None:
                anchor.close()
                self._wal_anchor = None
