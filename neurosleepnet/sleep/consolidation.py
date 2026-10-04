"""Explicit, incremental, non-destructive extractive consolidation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class Consolidator:
    def __init__(self, runtime):
        self.runtime = runtime
        self.storage = runtime.storage

    def _connect(self):
        conn = sqlite3.connect(self.runtime.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA recursive_triggers=ON")
        return conn

    @staticmethod
    def _source(conn, ref, namespace):
        kind, sid = ref["source_type"], ref["source_id"]
        version = conn.execute("SELECT revision FROM consolidation_sources WHERE namespace=? AND source_type=? AND source_id=?", (namespace, kind, sid)).fetchone()
        if not version or version[0] != ref["revision"]:
            return None
        table = "facts" if kind == "fact" else "events"
        row = conn.execute(f"SELECT * FROM {table} WHERE namespace=? AND id=?", (namespace, sid)).fetchone()
        if not row:
            return None
        row = dict(row)
        if kind == "fact":
            if row["status"] in ("deleted", "superseded"):
                return None
            # Exact raw values and qualifiers, including conflicts and validity.
            text = json.dumps({k: row[k] for k in ("subject", "predicate", "value", "unit", "qualifiers", "status", "conflict_group", "valid_from", "valid_to", "confidence", "source_authority", "cardinality")}, ensure_ascii=False, sort_keys=True)
        else:
            if row["role"] == "summary" or row["turn_status"] in ("cancelled", "deleted"):
                return None
            text = row["content"]
        dependencies = []
        if kind == "fact":
            supports = conn.execute("""SELECT s.event_id,c.revision,e.turn_status FROM fact_support s
                LEFT JOIN events e ON e.id=s.event_id AND e.namespace=?
                LEFT JOIN consolidation_sources c ON c.namespace=e.namespace AND c.source_type='event' AND c.source_id=e.id
                WHERE s.fact_id=? LIMIT 33""", (namespace, sid)).fetchall()
            if len(supports) > 32:
                raise ValueError("source_dependency_limit")
            dependencies = [dict(source_type="event",source_id=s["event_id"],revision=s["revision"]) for s in supports
                            if s["revision"] is not None and s["turn_status"] not in ("cancelled", "deleted")]
            if supports and not dependencies:
                return None
        if kind == "event":
            linked = conn.execute("""SELECT f.id,f.status,c.revision FROM fact_support s JOIN facts f ON f.id=s.fact_id
                JOIN consolidation_sources c ON c.namespace=f.namespace AND c.source_type='fact' AND c.source_id=f.id
                WHERE s.event_id=? AND f.namespace=? LIMIT 33""", (sid, namespace)).fetchall()
            if len(linked) > 32:
                raise ValueError("source_dependency_limit")
            if any(r["status"] in ("deleted", "superseded") for r in linked):
                return None
            dependencies = [dict(source_type="fact", source_id=r["id"], revision=r["revision"]) for r in linked]
        return dict(ref, text=text, session_id=row.get("session_id"),
                    time=row.get("valid_from") or row.get("event_time") or row.get("observed_at"),
                    source=row.get("source_identity", "structured_fact"), dependencies=dependencies)

    def pin(self, source_id, source_type="fact", namespace=None, pinned=True):
        ns = namespace or self.runtime.default_namespace
        if source_type not in ("event", "fact"):
            raise ValueError("source_type must be event or fact")
        conn = self._connect()
        try:
            with conn:
                if not conn.execute("SELECT 1 FROM consolidation_sources WHERE namespace=? AND source_type=? AND source_id=?", (ns, source_type, source_id)).fetchone():
                    raise ValueError("Source is missing or outside namespace")
                conn.execute("INSERT INTO source_importance VALUES (?,?,?,1,?) ON CONFLICT(namespace,source_type,source_id) DO UPDATE SET pinned=excluded.pinned", (ns, source_type, source_id, int(pinned)))
        finally:
            conn.close()

    def _enqueue(self, namespace, max_sources, settings):
        conn = self._connect()
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                first = conn.execute("SELECT * FROM consolidation_sources WHERE namespace=? AND revision>queued_revision ORDER BY source_type,source_id LIMIT 1", (namespace,)).fetchone()
                if not first:
                    return None
                rows = conn.execute("""SELECT source_type,source_id,revision FROM consolidation_sources
                    WHERE namespace=? AND neighborhood=? AND source_type=? AND revision>queued_revision
                    ORDER BY source_id LIMIT ?""", (namespace, first["neighborhood"], first["source_type"], max_sources)).fetchall()
                payload = {"refs": [dict(r) for r in rows], "settings": settings}
                encoded = json.dumps(payload, sort_keys=True)
                jid = "sleep:" + hashlib.sha256((namespace + encoded).encode()).hexdigest()
                conn.execute("INSERT OR IGNORE INTO jobs(id,namespace,job_type,payload,status,attempts,max_attempts,created_at,updated_at) VALUES (?,?,'consolidate',?,'pending',0,3,?,?)", (jid, namespace, encoded, now(), now()))
                for r in rows:
                    conn.execute("UPDATE consolidation_sources SET queued_revision=? WHERE namespace=? AND source_type=? AND source_id=?", (r["revision"], namespace, r["source_type"], r["source_id"]))
                return jid
        finally:
            conn.close()

    @staticmethod
    def _enrich(records, settings, remaining_seconds, report):
        command = settings.get("enrichment_command")
        if command is None or not records:
            return []
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Optional enrichment caps require pip install 'nsn[enrichment]'") from exc
        data = json.dumps({"records": records}, ensure_ascii=False).encode("utf-8")
        if len(data) > settings["max_model_input_bytes"]:
            raise ValueError("enrichment_input_limit")
        report["model_calls"] += 1
        report["model_input_bytes"] += len(data)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nsn_enrich_") as directory:
            inp, out, err = [Path(directory) / n for n in ("input", "output", "error")]
            inp.write_bytes(data)
            with inp.open("rb") as fi, out.open("wb") as fo, err.open("wb") as fe:
                process = subprocess.Popen(command, stdin=fi, stdout=fo, stderr=fe, shell=False)
                tracked = None
                descendants = {}
                try:
                    try:
                        tracked = psutil.Process(process.pid)
                    except psutil.NoSuchProcess:
                        pass
                    deadline = started + min(settings["enrichment_timeout_seconds"], remaining_seconds)
                    while process.poll() is None:
                        try:
                            rss = tracked.memory_info().rss if tracked else 0
                            for child in tracked.children(recursive=True) if tracked else []:
                                descendants[child.pid] = child
                            for child in descendants.values():
                                try:
                                    rss += child.memory_info().rss
                                except psutil.NoSuchProcess:
                                    pass
                            report["model_max_sampled_rss_bytes"] = max(report["model_max_sampled_rss_bytes"], rss)
                            if rss > settings["max_model_rss_mb"] * 1024 * 1024:
                                raise ValueError("enrichment_memory_limit")
                        except psutil.NoSuchProcess:
                            pass
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("enrichment_timeout")
                        if out.stat().st_size + err.stat().st_size > settings["max_model_output_bytes"]:
                            raise ValueError("enrichment_output_limit")
                        time.sleep(0.005)
                    if process.returncode:
                        raise ValueError(f"enrichment_exit_{process.returncode}")
                finally:
                    # Terminate descendants as well as the command on failure/cancellation.
                    try:
                        for child in tracked.children(recursive=True) if tracked else []:
                            descendants[child.pid] = child
                        for child in descendants.values():
                            try:
                                if child.is_running():
                                    child.kill()
                            except psutil.Error:
                                pass
                    except psutil.Error:
                        pass
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    # Windows venv launchers have child interpreters holding these
                    # files. Reap them before TemporaryDirectory cleanup can mask
                    # the original timeout/output-limit failure.
                    psutil.wait_procs(list(descendants.values()), timeout=1)
                    report["model_output_bytes"] += out.stat().st_size
            if out.stat().st_size + err.stat().st_size > settings["max_model_output_bytes"]:
                raise ValueError("enrichment_output_limit")
            raw = out.read_bytes()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or set(parsed) != {"proposals"} or not isinstance(parsed["proposals"], list) or len(parsed["proposals"]) > 8:
                raise ValueError("enrichment_schema")
            by_id = {(r["source_type"], r["source_id"]): r for r in records}
            fields = {"source_type", "source_id", "subject", "predicate", "value", "span_start", "span_end"}
            proposals = []
            for proposal in parsed["proposals"]:
                if not isinstance(proposal, dict) or set(proposal) != fields:
                    raise ValueError("enrichment_schema")
                if any(not isinstance(proposal[k], str) or not proposal[k] for k in ("source_type", "source_id", "subject", "predicate", "value")):
                    raise ValueError("enrichment_schema")
                source = by_id.get((proposal["source_type"], proposal["source_id"]))
                lo, hi = proposal["span_start"], proposal["span_end"]
                if source is None or type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= len(source["text"]) or source["text"][lo:hi] != proposal["value"]:
                    raise ValueError("enrichment_unbacked_proposal")
                proposals.append(dict(proposal, revision=source["revision"]))
            return proposals

    def _commit(self, job, records, content, proposals, worker_id):
        conn = self._connect()
        ns, jid = job["namespace"], job["id"]
        summary_id = "summary:" + jid.split(":", 1)[1]
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                lease = conn.execute("SELECT 1 FROM jobs WHERE id=? AND status='processing' AND leased_by=? AND lease_expires_at>=?", (jid, worker_id, now())).fetchone()
                if not lease:
                    raise RuntimeError("consolidation_lease_lost")
                # Revalidate under the same writer lock as all derived output.
                current = [self._source(conn, ref, ns) for ref in job["payload"]["refs"]]
                current = [r for r in current if r is not None]
                if current != records:
                    raise RuntimeError("consolidation_source_changed")
                if content:
                    sessions = {r["session_id"] for r in records}
                    session = sessions.pop() if len(sessions) == 1 else None
                    metadata = json.dumps({"derived": "extractive_summary", "job_id": jid, "support": [{k: r[k] for k in ("source_type", "source_id", "revision")} for r in records]})
                    conn.execute("""INSERT OR IGNORE INTO events(id,namespace,session_id,role,content,source_identity,observed_at,turn_status,metadata,retention_policy)
                        VALUES (?,?,?,'summary',?,'nsn_extractive',?,'completed',?,'derived')""", (summary_id, ns, session, content, now(), metadata))
                    conn.execute("INSERT INTO events_fts(id,content,namespace) SELECT ?,?,? WHERE NOT EXISTS(SELECT 1 FROM events_fts WHERE id=?)", (summary_id, content, ns, summary_id))
                    for r in records:
                        conn.execute("INSERT OR IGNORE INTO derived_support VALUES (?,?,?,?,?)", (summary_id, ns, r["source_type"], r["source_id"], r["revision"]))
                        for dep in r["dependencies"]:
                            conn.execute("INSERT OR IGNORE INTO derived_support VALUES (?,?,?,?,?)", (summary_id, ns, dep["source_type"], dep["source_id"], dep["revision"]))
                    if self.runtime.vector_store is not None:
                        vid = "vector:" + summary_id
                        conn.execute("INSERT OR IGNORE INTO jobs(id,namespace,job_type,payload,status,attempts,max_attempts,created_at,updated_at) VALUES (?,?,'vector_index',?,'pending',0,3,?,?)", (vid, ns, json.dumps({"target_id": summary_id, "target_type": "summary"}), now(), now()))
                for i, proposal in enumerate(proposals):
                    conn.execute("INSERT OR IGNORE INTO enrichment_proposals VALUES (?,?,?,?,?,?,?,?,?,?,?,'unconfirmed')", (jid+f":p{i}", ns, jid, proposal["source_type"], proposal["source_id"], proposal["revision"], proposal["subject"], proposal["predicate"], proposal["value"], proposal["span_start"], proposal["span_end"]))
                for ref in job["payload"]["refs"]:
                    conn.execute("UPDATE consolidation_sources SET processed_revision=? WHERE namespace=? AND source_type=? AND source_id=? AND revision=?", (ref["revision"], ns, ref["source_type"], ref["source_id"], ref["revision"]))
                    conn.execute("INSERT OR IGNORE INTO source_importance VALUES (?,?,?,1,0)", (ns, ref["source_type"], ref["source_id"]))
                    decay = job["payload"]["settings"]["importance_decay"]
                    conn.execute("UPDATE source_importance SET score=score*? WHERE namespace=? AND source_type=? AND source_id=? AND pinned=0", (decay, ns, ref["source_type"], ref["source_id"]))
                conn.execute("UPDATE jobs SET status='completed',leased_by=NULL,lease_expires_at=NULL,updated_at=? WHERE id=?", (now(), jid))
            return summary_id if content else None
        finally:
            conn.close()

    def sleep(self, namespace=None, max_jobs=8, max_sources=4, time_budget_ms=1000,
              max_summary_bytes=4096, connect=False, enrichment_command=None,
              enrichment_timeout_seconds=2, max_model_input_bytes=8192,
              max_model_output_bytes=8192, max_model_rss_mb=512,
              importance_decay=1.0, worker_id=None):
        if not 1 <= max_jobs <= 100 or not 1 <= max_sources <= 32 or not 0 < time_budget_ms <= 60000:
            raise ValueError("Sleep requires 1..100 jobs, 1..32 sources, and a 0..60000 ms budget")
        if not 128 <= max_summary_bytes <= 65536 or not 0 <= importance_decay <= 1:
            raise ValueError("Invalid summary or decay limit")
        if enrichment_command is not None and (not isinstance(enrichment_command, list) or not enrichment_command or not all(isinstance(x, str) for x in enrichment_command)):
            raise ValueError("enrichment_command must be an explicit argument list")
        if not 0 < enrichment_timeout_seconds <= 10 or not 1 <= max_model_input_bytes <= 65536 or not 1 <= max_model_output_bytes <= 65536 or not 1 <= max_model_rss_mb <= 4096:
            raise ValueError("Invalid enrichment resource caps")
        if connect and self.runtime.relationships is None:
            raise ValueError("connect=True requires graph=True")
        ns = namespace or self.runtime.default_namespace
        worker = (worker_id or "sleep") + "-" + str(uuid.uuid4())
        settings = dict(max_summary_bytes=max_summary_bytes, connect=connect, enrichment_command=enrichment_command,
                        enrichment_timeout_seconds=enrichment_timeout_seconds, max_model_input_bytes=max_model_input_bytes,
                        max_model_output_bytes=max_model_output_bytes, max_model_rss_mb=max_model_rss_mb, importance_decay=importance_decay)
        report = dict(enqueued=0, processed=0, failed=0, skipped=0, summaries=[], reasons=[], model_calls=0,
                      model_input_bytes=0, model_output_bytes=0, model_max_sampled_rss_bytes=0,
                      stages={s: 0 for s in ("enrich", "connect", "summarize", "validate", "compact")})
        start = time.perf_counter()
        deadline = start + time_budget_ms / 1000
        for _ in range(max_jobs):
            if time.perf_counter() >= deadline:
                report["reasons"].append("time_budget_ms")
                break
            report["enqueued"] += int(self._enqueue(ns, max_sources, settings) is not None)
            jobs = self.storage.get_pending_jobs(limit=1, worker_id=worker, job_type="consolidate", namespace=ns,
                                                 lease_duration_seconds=int(time_budget_ms/1000)+5)
            if not jobs:
                break
            job = jobs[0]
            try:
                conn = self._connect()
                try:
                    records = [self._source(conn, ref, ns) for ref in job["payload"]["refs"]]
                    records = [r for r in records if r is not None]
                finally:
                    conn.close()
                # Whole-source extraction, never cut identifiers, units, or conflicts.
                content = "\n".join(f"[{r['source_type']} {r['source_id']} revision={r['revision']} time={r['time']} source={r['source']}]\n{r['text']}" for r in records)
                if len(content.encode("utf-8")) > job["payload"]["settings"]["max_summary_bytes"]:
                    raise ValueError("summary_byte_limit")
                proposals = self._enrich(records, job["payload"]["settings"], deadline-time.perf_counter(), report)
                if time.perf_counter() >= deadline:
                    raise TimeoutError("sleep_time_budget_ms")
                if job["payload"]["settings"]["connect"]:
                    for r in records:
                        if r["source_type"] == "event":
                            self.runtime.extract_relations(r["source_id"], ns)
                            report["stages"]["connect"] += 1
                summary_id = self._commit(job, records, content, proposals, worker)
                report["processed"] += 1
                report["stages"]["validate"] += len(records)
                report["stages"]["enrich"] += len(proposals)
                if summary_id:
                    report["summaries"].append(summary_id)
                    report["stages"]["summarize"] += 1
                else:
                    report["skipped"] += 1
            except Exception as exc:
                self.storage.fail_job(job["id"], str(exc), worker_id=worker)
                report["failed"] += 1
                report["reasons"].append(str(exc))
                # One attempt per invocation; do not consume all retries immediately.
                break
        # Bounded cleanup of lineage for summaries already invalidated by triggers.
        conn = self._connect()
        try:
            with conn:
                cleanup = conn.execute("SELECT summary_id FROM derived_cleanup WHERE namespace=? LIMIT ?", (ns, max_jobs*max_sources)).fetchall()
                for row in cleanup:
                    conn.execute("DELETE FROM derived_support WHERE summary_id=?", (row[0],))
                    conn.execute("DELETE FROM derived_cleanup WHERE summary_id=?", (row[0],))
                report["stages"]["compact"] = len(cleanup)
            report["pending"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE namespace=? AND job_type='consolidate' AND status IN ('pending','processing')", (ns,)).fetchone()[0]
            report["terminal_failed"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE namespace=? AND job_type='consolidate' AND status='failed'", (ns,)).fetchone()[0]
            report["pending_index_jobs"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE namespace=? AND job_type='vector_index' AND status IN ('pending','processing')", (ns,)).fetchone()[0]
        finally:
            conn.close()
        report["elapsed_ms"] = round((time.perf_counter()-start)*1000, 3)
        return report
