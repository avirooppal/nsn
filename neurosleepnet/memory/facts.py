"""
Temporal and Typed Fact Management for NeuroSleepNet.
Preserves facts, versions, provenance, conflict groups, and historical truth.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json
import re
import sqlite3
import uuid

QUESTION_STARTERS = ("what", "where", "when", "which", "who", "why", "how", "is", "are", "can", "does", "do")


@dataclass
class Fact:
    id: str
    namespace: str
    subject: str
    predicate: str
    value: Any
    value_type: str = "string"          # 'string', 'number', 'boolean', 'json', 'set'
    unit: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[str] = None    # ISO UTC
    valid_to: Optional[str] = None      # ISO UTC
    recorded_at: Optional[str] = None   # ISO UTC
    invalidated_at: Optional[str] = None
    status: str = "asserted"            # 'asserted', 'derived', 'confirmed', 'superseded', 'disputed', 'deleted'
    cardinality: str = "single"         # 'single' or 'set'
    confidence: float = 1.0
    source_authority: float = 1.0       # system=1.0, user=0.8, assistant=0.3
    conflict_group: Optional[str] = None
    supporting_event_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FactManager:
    """
    Manages facts in SQLite with strict namespace isolation, temporal validity,
    versioning, cardinality, and conflict detection.
    """
    def __init__(self, storage, vector_store: Optional[Any] = None, encoder: Optional[Any] = None, defer_indexing: bool = False):
        self.storage = storage
        self.db_path = storage.db_path
        self.vector_store = vector_store
        self.encoder = encoder
        self.defer_indexing = defer_indexing

    def record_fact(
        self,
        subject: str,
        predicate: str,
        value: Any,
        namespace: str,
        value_type: str = "string",
        unit: Optional[str] = None,
        qualifiers: Optional[dict] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        status: Optional[str] = None,
        cardinality: str = "single",
        confidence: float = 1.0,
        source_authority: float = 1.0,
        supporting_event_ids: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
    ) -> Fact:
        now_str = datetime.now(timezone.utc).isoformat()
        qualifiers = qualifiers or {}
        supporting_event_ids = supporting_event_ids or []
        metadata = metadata or {}

        # Default status based on authority / source
        if status is None:
            if source_authority >= 0.8:
                status = "confirmed"
            elif source_authority >= 0.5:
                status = "asserted"
            else:
                status = "derived"

        # Check existing facts for (namespace, subject, predicate, unit)
        existing_facts = self._get_active_facts(namespace, subject, predicate, unit, qualifiers)

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()

        fact_id = id or str(uuid.uuid4())
        conflict_group = None
        return_id = None

        superseded_ids = []
        with conn:
            if cardinality == "single":
                # Handle single-valued update / conflict / out-of-order event
                for old in existing_facts:
                    old_id = old["id"]
                    old_val = old["value"]
                    old_from = old["valid_from"]
                    old_auth = old["source_authority"]

                    # Same value -> corroboration/noop
                    if str(old_val).strip() == str(value).strip():
                        # Update support
                        fact_id = old_id
                        cursor.execute(
                            "UPDATE facts SET recorded_at = ? WHERE id = ?", (now_str, fact_id)
                        )
                        for eid in supporting_event_ids:
                            cursor.execute(
                                "INSERT OR IGNORE INTO fact_support (id, fact_id, event_id, created_at) VALUES (?, ?, ?, ?)",
                                (str(uuid.uuid4()), fact_id, eid, now_str)
                            )
                        return_id = fact_id
                        break

                    # Contradicting value
                    # 1. Compare temporal validity if known
                    if valid_from and old_from:
                        if valid_from > old_from:
                            # New fact is newer: supersede old fact
                            cursor.execute('''
                                UPDATE facts
                                SET status = 'superseded', valid_to = ?, invalidated_at = ?
                                WHERE id = ?
                            ''', (valid_from, now_str, old_id))
                            superseded_ids.append(old_id)
                        elif valid_from < old_from:
                            # New fact is chronologically older: mark new fact as already superseded
                            status = "superseded"
                            valid_to = old_from
                        else:
                            # Equal timestamps, different values -> conflict!
                            if abs(source_authority - old_auth) < 0.1:
                                conflict_group = old.get("conflict_group") or f"conflict_{subject}_{predicate}_{uuid.uuid4().hex[:8]}"
                                cursor.execute(
                                    "UPDATE facts SET status = 'disputed', conflict_group = ? WHERE id = ?",
                                    (conflict_group, old_id)
                                )
                                status = "disputed"
                            elif source_authority > old_auth:
                                cursor.execute("UPDATE facts SET status = 'superseded', invalidated_at = ? WHERE id = ?", (now_str, old_id))
                                superseded_ids.append(old_id)
                            else:
                                status = "superseded"
                    else:
                        # Unknown timestamps: compare source authority
                        if source_authority > old_auth:
                            cursor.execute("UPDATE facts SET status = 'superseded', invalidated_at = ? WHERE id = ?", (now_str, old_id))
                            superseded_ids.append(old_id)
                        elif abs(source_authority - old_auth) < 0.1:
                            # Equal authority unresolved conflict!
                            conflict_group = old.get("conflict_group") or f"conflict_{subject}_{predicate}_{uuid.uuid4().hex[:8]}"
                            cursor.execute(
                                "UPDATE facts SET status = 'disputed', conflict_group = ? WHERE id = ?",
                                (conflict_group, old_id)
                            )
                            status = "disputed"
                        else:
                            status = "superseded"

            elif cardinality == "set":
                # Set-valued membership: combine with existing set if present
                for old in existing_facts:
                    old_id = old["id"]
                    try:
                        old_set = json.loads(old["value"]) if isinstance(old["value"], str) else old["value"]
                        if not isinstance(old_set, list):
                            old_set = [old_set]
                    except Exception:
                        old_set = [old["value"]]

                    if value not in old_set:
                        old_set.append(value)
                        cursor.execute(
                            "UPDATE facts SET value = ?, recorded_at = ? WHERE id = ?",
                            (json.dumps(old_set), now_str, old_id)
                        )
                    for eid in supporting_event_ids:
                        cursor.execute(
                            "INSERT OR IGNORE INTO fact_support (id, fact_id, event_id, created_at) VALUES (?, ?, ?, ?)",
                            (str(uuid.uuid4()), old_id, eid, now_str)
                        )
                    return_id = old_id
                    break

                if return_id is None:
                    # First member of set
                    value = [value]
                    value_type = "set"

            if return_id is None:
                # Insert new fact
                val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                qual_str = json.dumps(qualifiers)
                meta_str = json.dumps(metadata)

                cursor.execute('''
                    INSERT INTO facts (
                        id, namespace, subject, predicate, value, value_type, unit,
                        qualifiers, valid_from, valid_to, recorded_at, status,
                        cardinality, confidence, source_authority, conflict_group, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fact_id, namespace, subject, predicate, val_str, value_type, unit,
                    qual_str, valid_from, valid_to, now_str, status,
                    cardinality, confidence, source_authority, conflict_group, meta_str
                ))

                for eid in supporting_event_ids:
                    cursor.execute(
                        "INSERT INTO fact_support (id, fact_id, event_id, created_at) VALUES (?, ?, ?, ?)",
                        (str(uuid.uuid4()), fact_id, eid, now_str)
                    )
                return_id = fact_id

            indexing_job_id = None
            if self.vector_store is not None and return_id is not None and status not in ("superseded", "deleted"):
                indexing_job_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO jobs (
                        id, namespace, job_type, payload, status, attempts, max_attempts, created_at, updated_at
                    ) VALUES (?, ?, 'vector_index', ?, 'pending', 0, 3, ?, ?)
                ''', (
                    indexing_job_id,
                    namespace,
                    json.dumps({
                        "target_id": return_id,
                        "target_type": "fact",
                    }),
                    now_str,
                    now_str,
                ))

        conn.close()

        # Update vector store
        if self.vector_store is not None:
            for sid in superseded_ids:
                try:
                    self.vector_store.tombstone(sid, namespace=namespace)
                except Exception:
                    pass

        fact_obj = self.get_fact(return_id, namespace)
        if not self.defer_indexing and self.vector_store is not None and self.encoder is not None and fact_obj and fact_obj.status not in ("superseded", "deleted"):
            desc = f"{fact_obj.subject} {fact_obj.predicate} is {fact_obj.value}"
            if fact_obj.unit:
                desc += f" {fact_obj.unit}"
            try:
                vec = self.encoder.encode(desc)
                persisted = self.vector_store.add(fact_obj.id, vec, namespace=namespace, target_type="fact", validate_target=True)
                if indexing_job_id:
                    self.storage.complete_job(indexing_job_id, worker_id=None)
            except Exception as e:
                # Synchronous indexing failed: update job with error trace so it is visible and recoverable
                if indexing_job_id:
                    self.storage.fail_job(indexing_job_id, str(e), worker_id=None)

        return fact_obj

    def delete_fact(self, fact_id: str, namespace: str) -> bool:
        """Mark a fact as deleted and tombstone its vector."""
        now_str = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        with conn:
            cursor.execute(
                "UPDATE facts SET status = 'deleted', invalidated_at = ? WHERE id = ? AND namespace = ?",
                (now_str, fact_id, namespace)
            )
            affected = cursor.rowcount
            # Cancel any pending indexing jobs for this fact so replay will not restore
            cursor.execute(
                "UPDATE jobs SET status = 'completed', updated_at = ? WHERE namespace = ? AND job_type = 'vector_index' AND payload LIKE ?",
                (now_str, namespace, f'%"{fact_id}"%')
            )
        conn.close()
        if self.vector_store is not None:
            try:
                self.vector_store.tombstone(fact_id, namespace=namespace)
            except Exception:
                pass
        return affected > 0

    def _get_active_facts(self, namespace: str, subject: str, predicate: str, unit: Optional[str], qualifiers: dict) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, namespace, subject, predicate, value, value_type, unit,
                   qualifiers, valid_from, valid_to, recorded_at, invalidated_at,
                   status, cardinality, confidence, source_authority, conflict_group, metadata
            FROM facts
            WHERE namespace = ? AND subject = ? AND predicate = ?
              AND status IN ('asserted', 'confirmed', 'disputed')
        ''', (namespace, subject, predicate))
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            r_unit = r[6]
            r_qual = json.loads(r[7]) if r[7] else {}
            # Only match if units match and qualifiers match
            if r_unit == unit and r_qual == qualifiers:
                results.append(self._row_to_dict(r))
        return results

    def _row_to_dict(self, r: tuple) -> dict:
        v = r[4]
        v_type = r[5]
        if v_type in ("json", "set"):
            try:
                v = json.loads(v)
            except Exception:
                pass
        elif v_type == "number":
            try:
                v = float(v) if "." in v else int(v)
            except Exception:
                pass
        elif v_type == "boolean":
            v = v.lower() in ("true", "1")

        return {
            "id": r[0],
            "namespace": r[1],
            "subject": r[2],
            "predicate": r[3],
            "value": v,
            "value_type": v_type,
            "unit": r[6],
            "qualifiers": json.loads(r[7]) if r[7] else {},
            "valid_from": r[8],
            "valid_to": r[9],
            "recorded_at": r[10],
            "invalidated_at": r[11],
            "status": r[12],
            "cardinality": r[13],
            "confidence": float(r[14]),
            "source_authority": float(r[15]),
            "conflict_group": r[16],
            "metadata": json.loads(r[17]) if r[17] else {},
        }

    def get_fact(self, fact_id: str, namespace: str) -> Optional[Fact]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, namespace, subject, predicate, value, value_type, unit,
                   qualifiers, valid_from, valid_to, recorded_at, invalidated_at,
                   status, cardinality, confidence, source_authority, conflict_group, metadata
            FROM facts WHERE id = ? AND namespace = ?
        ''', (fact_id, namespace))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        cursor.execute("SELECT event_id FROM fact_support WHERE fact_id = ?", (fact_id,))
        support_rows = cursor.fetchall()
        conn.close()

        d = self._row_to_dict(row)
        d["supporting_event_ids"] = [sr[0] for sr in support_rows]
        return Fact(**d)

    def get_current_fact(self, subject: str, predicate: str, namespace: str) -> Optional[Fact]:
        """Get the current, active fact for (subject, predicate)."""
        facts = self.list_facts(namespace, subject=subject, predicate=predicate, include_superseded=False)
        active = [f for f in facts if f.status in ("asserted", "confirmed", "disputed")]
        if not active:
            return None
        # Return highest authority / newest recorded
        active.sort(key=lambda f: (f.source_authority, f.recorded_at or ""), reverse=True)
        return active[0]

    def get_fact_as_of(self, subject: str, predicate: str, as_of_time: str, namespace: str) -> Optional[Fact]:
        """
        Query historical truth: fact valid at `as_of_time`.
        """
        facts = self.list_facts(namespace, subject=subject, predicate=predicate, include_superseded=True)
        candidates = []
        for f in facts:
            if f.status == "deleted":
                continue
            vf = f.valid_from
            vt = f.valid_to
            if vf and vf > as_of_time:
                continue
            if vt and vt <= as_of_time:
                continue
            candidates.append(f)

        if not candidates:
            return None
        candidates.sort(key=lambda f: (f.valid_from or "", f.source_authority), reverse=True)
        return candidates[0]

    def list_facts(
        self,
        namespace: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        status: Optional[str] = None,
        include_superseded: bool = False,
    ) -> List[Fact]:

        query = "SELECT id, namespace, subject, predicate, value, value_type, unit, qualifiers, valid_from, valid_to, recorded_at, invalidated_at, status, cardinality, confidence, source_authority, conflict_group, metadata FROM facts WHERE namespace = ?"
        params: List[Any] = [namespace]

        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)
        if status:
            query += " AND status = ?"
            params.append(status)
        elif not include_superseded:
            query += " AND status NOT IN ('superseded', 'deleted')"

        with self.storage._runtime_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        result = []
        for r in rows:
            d = self._row_to_dict(r)
            result.append(Fact(**d))
        return result

    def invalidate_event_support(self, event_id: str, namespace: str):
        """
        When an event is deleted, remove support links and mark facts with
        no remaining support as 'deleted'.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        with conn:
            cursor.execute("SELECT fact_id FROM fact_support WHERE event_id = ?", (event_id,))
            affected_facts = [r[0] for r in cursor.fetchall()]
            cursor.execute("DELETE FROM fact_support WHERE event_id = ?", (event_id,))

            for fid in affected_facts:
                cursor.execute("SELECT COUNT(*) FROM fact_support WHERE fact_id = ?", (fid,))
                remaining = cursor.fetchone()[0]
                if remaining == 0:
                    cursor.execute(
                        "UPDATE facts SET status = 'deleted', invalidated_at = ? WHERE id = ? AND namespace = ?",
                        (datetime.now(timezone.utc).isoformat(), fid, namespace)
                    )
        conn.close()

    def extract_conservative_facts(self, event: dict) -> List[Fact]:
        """
        Conservative deterministic heuristic fact extraction from raw events.
        Never extracts facts from questions or unconfirmed assistant text.
        """
        content = event.get("content", "").strip()
        role = event.get("role", "user")
        namespace = event.get("namespace", "default")
        event_id = event.get("id")
        source = event.get("source_identity", event.get("source", "user"))

        # Questions are inquiries, never assertions
        if content.endswith("?") or any(content.lower().startswith(q) for q in QUESTION_STARTERS):
            return []

        # Assistant output is not an authoritative fact by default
        if role == "assistant":
            source_authority = 0.3
        elif source == "system":
            source_authority = 1.0
        elif source == "user":
            source_authority = 0.8
        else:
            source_authority = 0.5

        facts = []
        # Pattern 1: Day X: <Service> production port [configured to|updated to|migrated to] <Port>
        match_port_update = re.search(
            r'(?:Day\s+(\d+):\s+)?(\w+)\s+(?:production\s+)?port\s+(?:is\s+configured\s+to|updated\s+to|migrated\s+to|is\s+set\s+to|is|=)\s+(\d+)',
            content,
            re.IGNORECASE
        )
        if match_port_update:
            day_str = match_port_update.group(1)
            service = match_port_update.group(2)
            port_val = int(match_port_update.group(3))
            valid_from = f"Day {day_str}" if day_str else event.get("event_time") or event.get("observed_at")
            fact = self.record_fact(
                subject=service,
                predicate="production_port",
                value=port_val,
                namespace=namespace,
                value_type="number",
                unit="port",
                valid_from=valid_from,
                source_authority=source_authority,
                supporting_event_ids=[event_id] if event_id else [],
            )
            facts.append(fact)
            return facts

        # Pattern 2: The system <attr> is (not )?<val>.
        match_sys = re.search(
            r'(?:The\s+)?system\s+([\w\s]+?)\s+is\s+(not\s+)?([\w\.\-]+)',
            content,
            re.IGNORECASE
        )
        if match_sys:
            attr = match_sys.group(1).strip()
            is_neg = bool(match_sys.group(2))
            val = match_sys.group(3).strip()
            if not is_neg:
                fact = self.record_fact(
                    subject="system",
                    predicate=attr,
                    value=val,
                    namespace=namespace,
                    value_type="string",
                    valid_from=event.get("event_time") or event.get("observed_at"),
                    source_authority=source_authority,
                    supporting_event_ids=[event_id] if event_id else [],
                )
                facts.append(fact)
            return facts

        return facts
