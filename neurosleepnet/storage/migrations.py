"""
Database schema migrations for NeuroSleepNet.
Provides versioned, transactional, idempotent migrations.
"""
import sqlite3
import datetime
import os
import logging

logger = logging.getLogger("neurosleepnet.migrations")

CURRENT_SCHEMA_VERSION = 7

MIGRATIONS = [
    (1, "Initialize legacy schema and migration tracking", [
        """CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS memories (
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
        )""",
        """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            content,
            namespace
        )""",
        """CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            name TEXT NOT NULL,
            properties TEXT,
            namespace TEXT DEFAULT 'default',
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            properties TEXT,
            namespace TEXT DEFAULT 'default',
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)",
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
    ]),
    (2, "Create events, jobs, index_state, and events_fts", [
        """CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            event_time TEXT,
            turn_status TEXT DEFAULT 'completed',
            idempotency_key TEXT,
            metadata TEXT,
            retention_policy TEXT DEFAULT 'permanent'
        )""",
        "CREATE INDEX IF NOT EXISTS idx_events_ns_time ON events(namespace, observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_events_ns_session ON events(namespace, session_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_ns_idempotency ON events(namespace, idempotency_key)",
        """CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
            id UNINDEXED,
            content,
            namespace
        )""",
        """CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            job_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 3,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            leased_by TEXT,
            lease_expires_at TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_jobs_pending ON jobs(status, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_ns ON jobs(namespace, status)",
        """CREATE TABLE IF NOT EXISTS index_state (
            name TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            last_processed_event_id TEXT,
            last_processed_at TEXT,
            revision INTEGER DEFAULT 1
        )""",
    ]),
    (3, "Create facts and fact_support tables", [
        """CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            value TEXT NOT NULL,
            value_type TEXT DEFAULT 'string',
            unit TEXT,
            qualifiers TEXT,
            valid_from TEXT,
            valid_to TEXT,
            recorded_at TEXT NOT NULL,
            invalidated_at TEXT,
            status TEXT NOT NULL,
            cardinality TEXT DEFAULT 'single',
            confidence REAL DEFAULT 1.0,
            source_authority REAL DEFAULT 1.0,
            conflict_group TEXT,
            metadata TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_facts_lookup ON facts(namespace, subject, predicate, status)",
        "CREATE INDEX IF NOT EXISTS idx_facts_validity ON facts(namespace, valid_from, valid_to)",
        "CREATE INDEX IF NOT EXISTS idx_facts_conflict ON facts(namespace, conflict_group)",
        """CREATE TABLE IF NOT EXISTS fact_support (
            id TEXT PRIMARY KEY,
            fact_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            source_span TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(fact_id) REFERENCES facts(id) ON DELETE CASCADE,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
        )""",
        "CREATE INDEX IF NOT EXISTS idx_fact_support_fact ON fact_support(fact_id)",
        "CREATE INDEX IF NOT EXISTS idx_fact_support_event ON fact_support(event_id)",
    ]),
    (4, "Create compact_vectors and vector_manifest tables", [
        """CREATE TABLE IF NOT EXISTS vector_manifest (
            index_name TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            preprocessing_fingerprint TEXT NOT NULL,
            index_revision INTEGER DEFAULT 1,
            updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS compact_vectors (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            target_type TEXT NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL,
            tombstoned INTEGER DEFAULT 0
        )""",
        "CREATE INDEX IF NOT EXISTS idx_compact_vec_ns ON compact_vectors(namespace, target_type, tombstoned)",
    ]),
    (5, "Source-backed relationships and vector cache coherence", [
        "CREATE TABLE vector_changes (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)",
        "INSERT INTO vector_changes VALUES (1, 0)",
        "CREATE TRIGGER compact_changed_insert AFTER INSERT ON compact_vectors BEGIN UPDATE vector_changes SET revision=revision+1 WHERE id=1; END",
        "CREATE TRIGGER compact_changed_update AFTER UPDATE ON compact_vectors BEGIN UPDATE vector_changes SET revision=revision+1 WHERE id=1; END",
        "CREATE TRIGGER compact_changed_delete AFTER DELETE ON compact_vectors BEGIN UPDATE vector_changes SET revision=revision+1 WHERE id=1; END",
        "CREATE TRIGGER fact_vector_invalidated AFTER UPDATE OF status ON facts WHEN NEW.status IN ('deleted','superseded') BEGIN UPDATE compact_vectors SET tombstoned=1 WHERE id=NEW.id AND namespace=NEW.namespace; END",
        "CREATE TRIGGER event_vector_deleted AFTER DELETE ON events BEGIN UPDATE compact_vectors SET tombstoned=1 WHERE id=OLD.id AND namespace=OLD.namespace; DELETE FROM relationship_support WHERE event_id=OLD.id AND namespace=OLD.namespace; DELETE FROM unresolved_relationships WHERE event_id=OLD.id AND namespace=OLD.namespace; END",
        "CREATE TRIGGER event_vector_cancelled AFTER UPDATE OF turn_status ON events WHEN NEW.turn_status IN ('cancelled','deleted') BEGIN UPDATE compact_vectors SET tombstoned=1 WHERE id=NEW.id AND namespace=NEW.namespace; END",
        """CREATE TABLE relationship_entities (
            namespace TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL,
            PRIMARY KEY(namespace, id), UNIQUE(namespace, name))""",
        """CREATE TABLE relationship_aliases (
            namespace TEXT NOT NULL, alias TEXT NOT NULL, entity_id TEXT NOT NULL,
            PRIMARY KEY(namespace, alias, entity_id),
            FOREIGN KEY(namespace, entity_id) REFERENCES relationship_entities(namespace, id) ON DELETE CASCADE)""",
        """CREATE TABLE relationships (
            namespace TEXT NOT NULL, id TEXT NOT NULL, subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL, object_id TEXT NOT NULL, kind TEXT NOT NULL,
            confidence REAL NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT,
            PRIMARY KEY(namespace, id),
            FOREIGN KEY(namespace, subject_id) REFERENCES relationship_entities(namespace, id),
            FOREIGN KEY(namespace, object_id) REFERENCES relationship_entities(namespace, id))""",
        "CREATE INDEX relationships_out ON relationships(namespace, subject_id, valid_from, valid_to)",
        """CREATE TABLE relationship_support (
            namespace TEXT NOT NULL, relation_id TEXT NOT NULL, event_id TEXT NOT NULL,
            span_start INTEGER NOT NULL, span_end INTEGER NOT NULL,
            PRIMARY KEY(namespace, relation_id, event_id, span_start, span_end),
            FOREIGN KEY(namespace, relation_id) REFERENCES relationships(namespace, id) ON DELETE CASCADE,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE)""",
        """CREATE TABLE unresolved_relationships (
            id TEXT PRIMARY KEY, namespace TEXT NOT NULL, event_id TEXT NOT NULL,
            text TEXT NOT NULL, reason TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE)""",
    ]),
    (6, "Incremental recoverable extractive consolidation", [
        "UPDATE compact_vectors SET target_type='summary' WHERE id IN (SELECT id FROM events WHERE role='summary')",
        """CREATE TABLE consolidation_sources (
            namespace TEXT NOT NULL, source_type TEXT NOT NULL, source_id TEXT NOT NULL,
            neighborhood TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
            processed_revision INTEGER NOT NULL DEFAULT 0,
            queued_revision INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(namespace, source_type, source_id))""",
        "CREATE INDEX consolidation_dirty ON consolidation_sources(namespace, neighborhood, source_type, processed_revision, revision)",
        "CREATE INDEX consolidation_pending ON consolidation_sources(namespace,source_type,source_id) WHERE revision>queued_revision",
        """CREATE TABLE derived_support (
            summary_id TEXT NOT NULL, namespace TEXT NOT NULL, source_type TEXT NOT NULL,
            source_id TEXT NOT NULL, revision INTEGER NOT NULL,
            PRIMARY KEY(summary_id, source_type, source_id))""",
        "CREATE INDEX derived_source ON derived_support(namespace, source_type, source_id)",
        "CREATE TABLE derived_cleanup (summary_id TEXT PRIMARY KEY, namespace TEXT NOT NULL)",
        "CREATE INDEX derived_cleanup_namespace ON derived_cleanup(namespace)",
        "CREATE TRIGGER consolidate_summary_cleanup BEFORE DELETE ON events WHEN OLD.role='summary' BEGIN INSERT OR IGNORE INTO derived_cleanup VALUES (OLD.id,OLD.namespace); END",
        """CREATE TABLE enrichment_proposals (
            id TEXT PRIMARY KEY, namespace TEXT NOT NULL, job_id TEXT NOT NULL,
            source_type TEXT NOT NULL, source_id TEXT NOT NULL, revision INTEGER NOT NULL,
            subject TEXT NOT NULL, predicate TEXT NOT NULL, value TEXT NOT NULL,
            span_start INTEGER NOT NULL, span_end INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'unconfirmed')""",
        """CREATE TABLE source_importance (
            namespace TEXT NOT NULL, source_type TEXT NOT NULL, source_id TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 1, pinned INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(namespace, source_type, source_id))""",
        """INSERT INTO consolidation_sources(namespace,source_type,source_id,neighborhood)
            SELECT namespace,'event',id,COALESCE(session_id,'') FROM events WHERE role!='summary'""",
        """INSERT INTO consolidation_sources(namespace,source_type,source_id,neighborhood)
            SELECT namespace,'fact',id,subject FROM facts""",
        """CREATE TRIGGER consolidate_event_insert AFTER INSERT ON events WHEN NEW.role!='summary' BEGIN
            INSERT INTO consolidation_sources(namespace,source_type,source_id,neighborhood,revision,processed_revision) VALUES (NEW.namespace,'event',NEW.id,COALESCE(NEW.session_id,''),1,0)
            ON CONFLICT(namespace,source_type,source_id) DO UPDATE SET revision=revision+1,neighborhood=excluded.neighborhood;
        END""",
        """CREATE TRIGGER consolidate_fact_insert AFTER INSERT ON facts BEGIN
            INSERT INTO consolidation_sources(namespace,source_type,source_id,neighborhood,revision,processed_revision) VALUES (NEW.namespace,'fact',NEW.id,NEW.subject,1,0)
            ON CONFLICT(namespace,source_type,source_id) DO UPDATE SET revision=revision+1,neighborhood=excluded.neighborhood;
        END""",
        """CREATE TRIGGER consolidate_event_update AFTER UPDATE ON events WHEN NEW.role!='summary' BEGIN
            DELETE FROM relationship_support WHERE namespace=NEW.namespace AND event_id=NEW.id;
            DELETE FROM fact_support WHERE event_id=NEW.id AND (OLD.content IS NOT NEW.content OR NEW.turn_status IN ('cancelled','deleted'));
            UPDATE consolidation_sources SET revision=revision+1,neighborhood=COALESCE(NEW.session_id,'') WHERE namespace=NEW.namespace AND source_type='event' AND source_id=NEW.id;
            DELETE FROM events_fts WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=NEW.namespace AND source_type='event' AND source_id=NEW.id);
            DELETE FROM events WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=NEW.namespace AND source_type='event' AND source_id=NEW.id);
            DELETE FROM enrichment_proposals WHERE namespace=NEW.namespace AND source_type='event' AND source_id=NEW.id;
        END""",
        """CREATE TRIGGER consolidate_fact_update AFTER UPDATE ON facts BEGIN
            UPDATE consolidation_sources SET revision=revision+1 WHERE namespace=NEW.namespace AND source_type='fact' AND source_id=NEW.id;
            DELETE FROM events_fts WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=NEW.namespace AND source_type='fact' AND source_id=NEW.id);
            DELETE FROM events WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=NEW.namespace AND source_type='fact' AND source_id=NEW.id);
            DELETE FROM enrichment_proposals WHERE namespace=NEW.namespace AND source_type='fact' AND source_id=NEW.id;
            DELETE FROM enrichment_proposals WHERE namespace=NEW.namespace AND source_type='event' AND source_id IN (SELECT event_id FROM fact_support WHERE fact_id=NEW.id);
        END""",
        """CREATE TRIGGER consolidate_source_event_deleted AFTER DELETE ON events BEGIN
            UPDATE consolidation_sources SET revision=revision+1 WHERE namespace=OLD.namespace AND source_type='event' AND source_id=OLD.id;
            DELETE FROM events_fts WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=OLD.namespace AND source_type='event' AND source_id=OLD.id);
            DELETE FROM events WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=OLD.namespace AND source_type='event' AND source_id=OLD.id);
            DELETE FROM enrichment_proposals WHERE namespace=OLD.namespace AND source_type='event' AND source_id=OLD.id;
            DELETE FROM events_fts WHERE id=OLD.id;
            DELETE FROM derived_support WHERE summary_id=OLD.id;
        END""",
        """CREATE TRIGGER consolidate_source_fact_deleted AFTER DELETE ON facts BEGIN
            UPDATE consolidation_sources SET revision=revision+1 WHERE namespace=OLD.namespace AND source_type='fact' AND source_id=OLD.id;
            DELETE FROM events_fts WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=OLD.namespace AND source_type='fact' AND source_id=OLD.id);
            DELETE FROM events WHERE id IN (SELECT summary_id FROM derived_support WHERE namespace=OLD.namespace AND source_type='fact' AND source_id=OLD.id);
            DELETE FROM enrichment_proposals WHERE namespace=OLD.namespace AND source_type='fact' AND source_id=OLD.id;
            DELETE FROM enrichment_proposals WHERE namespace=OLD.namespace AND source_type='event' AND source_id IN (SELECT event_id FROM fact_support WHERE fact_id=OLD.id);
        END""",
        """CREATE TRIGGER consolidate_fact_support_added AFTER INSERT ON fact_support BEGIN
            UPDATE consolidation_sources SET revision=revision+1 WHERE source_type='event' AND source_id=NEW.event_id AND namespace=(SELECT namespace FROM facts WHERE id=NEW.fact_id);
            DELETE FROM enrichment_proposals WHERE source_type='event' AND source_id=NEW.event_id;
            DELETE FROM events_fts WHERE id IN (SELECT summary_id FROM derived_support WHERE source_type='event' AND source_id=NEW.event_id);
            DELETE FROM events WHERE id IN (SELECT summary_id FROM derived_support WHERE source_type='event' AND source_id=NEW.event_id);
        END""",
        """CREATE TRIGGER consolidate_fact_support_removed AFTER DELETE ON fact_support BEGIN
            UPDATE consolidation_sources SET revision=revision+1 WHERE source_type='fact' AND source_id=OLD.fact_id;
            DELETE FROM events_fts WHERE id IN (SELECT summary_id FROM derived_support WHERE source_type='fact' AND source_id=OLD.fact_id);
            DELETE FROM events WHERE id IN (SELECT summary_id FROM derived_support WHERE source_type='fact' AND source_id=OLD.fact_id);
            DELETE FROM enrichment_proposals WHERE source_type='fact' AND source_id=OLD.fact_id;
            UPDATE facts SET status='deleted' WHERE id=OLD.fact_id AND NOT EXISTS(SELECT 1 FROM fact_support WHERE fact_id=OLD.fact_id);
        END""",
    ]),
]


def get_schema_version(conn: sqlite3.Connection) -> int:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(version) FROM schema_migrations")
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


MIGRATIONS.append((7, "Validated procedure traces", [
    "CREATE TABLE procedures (id TEXT PRIMARY KEY, namespace TEXT NOT NULL, title TEXT NOT NULL, steps TEXT NOT NULL, prerequisites TEXT NOT NULL, environment TEXT NOT NULL, tools TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL)",
    "CREATE TABLE procedure_traces (id TEXT PRIMARY KEY, namespace TEXT NOT NULL, procedure_id TEXT NOT NULL REFERENCES procedures(id) ON DELETE CASCADE, event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE, outcome TEXT NOT NULL, evidence TEXT NOT NULL, created_at TEXT NOT NULL)",
    "CREATE INDEX procedure_scope ON procedures(namespace)",
    "CREATE TRIGGER procedure_event_changed AFTER UPDATE OF content,turn_status ON events BEGIN DELETE FROM procedure_traces WHERE event_id=NEW.id AND namespace=NEW.namespace; END",
    "CREATE TRIGGER procedure_event_deleted AFTER DELETE ON events BEGIN DELETE FROM procedure_traces WHERE event_id=OLD.id AND namespace=OLD.namespace; END",
]))

def apply_migrations(db_path: str, target_version: int = CURRENT_SCHEMA_VERSION):
    """
    Apply migrations up to target_version transactionally.
    If db exists and has legacy tables without schema_migrations, creates a backup first.
    """
    if db_path != ":memory:" and os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        # Check if legacy db
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        has_memories = cur.fetchone() is not None
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
        has_migrations = cur.fetchone() is not None
        if has_memories and not has_migrations:
            backup_path = f"{db_path}.bak"
            if not os.path.exists(backup_path):
                # A file copy can omit committed records still in the WAL.
                with sqlite3.connect(backup_path) as backup_conn:
                    conn.backup(backup_conn)
                logger.info(f"Created legacy backup at {backup_path}")
        conn.close()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")

    current_ver = get_schema_version(conn)
    if current_ver > target_version:
        conn.close()
        raise ValueError(
            f"Unsupported schema version {current_ver}. Maximum supported version is {target_version}."
        )

    for ver, desc, statements in MIGRATIONS:
        if ver > target_version:
            break
        if ver <= current_ver:
            continue
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                # Another initializer may have migrated while this one waited.
                if get_schema_version(conn) >= ver:
                    continue
                cursor = conn.cursor()
                for stmt in statements:
                    cursor.execute(stmt)
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
                    (ver, now, desc)
                )
            logger.info(f"Applied migration {ver}: {desc}")
        except Exception as e:
            conn.close()
            raise RuntimeError(f"Migration {ver} failed: {e}") from e

    with conn:
        cursor = conn.cursor()
        for col_stmt in [
            "ALTER TABLE graph_nodes ADD COLUMN namespace TEXT DEFAULT 'default'",
            "ALTER TABLE graph_edges ADD COLUMN namespace TEXT DEFAULT 'default'",
        ]:
            try:
                cursor.execute(col_stmt)
            except sqlite3.OperationalError:
                pass

    conn.close()
