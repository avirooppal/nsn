import json
import sqlite3
import sys
import threading

import pytest
from nsn import Runtime
from neurosleepnet.embeddings.pooled import FastHashEncoder
from neurosleepnet.sleep.consolidation import Consolidator


def test_summary_metadata_excludes_unavailable_queued_source(tmp_path):
    rt = Runtime(str(tmp_path))
    removed = rt.append_event(content='Unavailable queued observation')
    retained = rt.append_event(content='Retained queued observation')
    worker = Consolidator(rt)
    enqueue = worker._enqueue
    def enqueue_then_delete(*args):
        result = enqueue(*args)
        if result:
            rt.storage.delete_event(removed,'default')
        return result
    worker._enqueue = enqueue_then_delete
    report = worker.sleep(max_jobs=1)
    assert report['summaries']
    summary = rt.storage.get_event(report['summaries'][0],'default')
    metadata = summary['metadata']
    if isinstance(metadata,str):
        metadata = json.loads(metadata)
    assert {r['source_id'] for r in metadata['support']}=={retained}


def query(rt, sql, args=()):
    conn = sqlite3.connect(rt.db_path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def test_repeat_sleep_and_restart_lease_recovery(tmp_path):
    rt = Runtime(str(tmp_path))
    eid = rt.append_event(content="Production port 8080, duration 7 ms, uuid-42.")
    worker = Consolidator(rt)
    real_commit = worker._commit
    def crash(*args):
        raise KeyboardInterrupt("crash before output commit")
    worker._commit = crash
    with pytest.raises(KeyboardInterrupt):
        worker.sleep()
    assert query(rt, "SELECT COUNT(*) FROM events WHERE role='summary'")[0][0] == 0
    conn = sqlite3.connect(rt.db_path)
    with conn:
        conn.execute("UPDATE jobs SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE job_type='consolidate'")
    conn.close()
    rt.close()
    rt = Runtime(str(tmp_path))
    report = rt.sleep()
    assert report["processed"] == 1 and report["failed"] == 0
    assert rt.sleep()["processed"] == 0
    content = rt.storage.get_event(report["summaries"][0], "default")["content"]
    assert "8080" in content and "7 ms" in content and "uuid-42" in content and eid in content
    assert rt.storage.get_event(eid, "default") is not None
    rt.close()


def test_fact_correction_deletion_lineage_vectors_and_asof(tmp_path):
    rt = Runtime(str(tmp_path), encoder=FastHashEncoder(dimension=16))
    source = rt.append_event(content="Production gateway port 8080.")
    first = rt.record_fact(subject="gateway", predicate="port", value=8080, unit="port",
                           valid_from="2025-01-01T00:00:00Z", supporting_event_ids=[source], value_type="number")
    report = rt.sleep()
    rt.process_index_jobs()
    summary_ids = report["summaries"]
    assert summary_ids
    # Warm an independent runtime's cache, then correct the fact.
    other = Runtime(str(tmp_path), encoder=FastHashEncoder(dimension=16))
    other.vector_store.search(other.encoder.encode("gateway port"))
    latest = rt.record_fact(subject="gateway", predicate="port", value=9090, unit="port",
                            valid_from="2026-01-01T00:00:00Z", value_type="number")
    assert all(rt.storage.get_event(s, "default") is None for s in summary_ids)
    assert not any(r["id"] in summary_ids for r in other.vector_store.search(other.encoder.encode("gateway port")))
    assert not query(rt, "SELECT id FROM events_fts WHERE id IN (" + ",".join("?" for _ in summary_ids) + ")", summary_ids)
    rt.sleep()
    historical = rt.retrieve_pack("gateway port", as_of_time="2025-06-01T00:00:00Z")
    assert any(i.kind == "fact" and i.value == 8080 for i in historical.items)
    assert not any(i.kind == "summary" for i in historical.items)
    assert any(i.kind == "fact" and i.value == 9090 for i in rt.retrieve_pack("gateway port").items)
    rt.fact_manager.delete_fact(latest.id, "default")
    assert not query(rt, "SELECT content FROM events WHERE role='summary' AND content LIKE '%9090%'")
    other.close()
    rt.close()


def test_delete_raw_source_removes_summary_fts_and_does_not_replay_stale_job(tmp_path):
    rt = Runtime(str(tmp_path))
    a = rt.append_event(content="private uuid-secret")
    b = rt.append_event(content="another source")
    summary = rt.sleep()["summaries"][0]
    rt.storage.delete_event(a, "default")
    assert rt.storage.get_event(summary, "default") is None
    assert query(rt, "SELECT COUNT(*) FROM events_fts WHERE id=?", (summary,))[0][0] == 0
    assert rt.storage.get_event(b, "default")
    assert rt.sleep()["failed"] == 0
    assert not query(rt, "SELECT content FROM events WHERE role='summary' AND content LIKE '%uuid-secret%'")
    rt.close()


def test_conflicts_units_pin_decay_and_namespace_isolation(tmp_path):
    rt = Runtime(str(tmp_path))
    one = rt.record_fact(subject="port", predicate="value", value=8080, unit="tcp", source_authority=0.8, valid_from="2025-01-01T00:00:00Z")
    two = rt.record_fact(subject="port", predicate="value", value=9090, unit="tcp", source_authority=0.8, valid_from="2025-01-01T00:00:00Z")
    rt.pin(one.id)
    rt.append_event(content="tenant B secret", namespace="b")
    report = rt.sleep(importance_decay=0.5)
    contents = [rt.storage.get_event(s, "default")["content"] for s in report["summaries"]]
    assert any("8080" in s and "9090" in s and "tcp" in s and "disputed" in s and "conflict_group" in s for s in contents)
    scores = dict(query(rt, "SELECT source_id,score FROM source_importance WHERE source_type='fact'"))
    assert scores[one.id] == 1 and scores[two.id] == 0.5
    assert not query(rt, "SELECT id FROM events WHERE namespace='b' AND role='summary'")
    assert not any("tenant B" in s for s in contents)
    rt.close()


def command(code):
    return [sys.executable, "-c", code]


def test_optional_enrichment_unconfirmed_only_and_fabrication_rejected(tmp_path):
    rt = Runtime(str(tmp_path))
    eid = rt.append_event(content="Paris")
    backend = command("import sys,json; r=json.load(sys.stdin)['records'][0]; print(json.dumps({'proposals':[{'source_type':r['source_type'],'source_id':r['source_id'],'subject':'person','predicate':'city','value':r['text'],'span_start':0,'span_end':len(r['text'])}]}))")
    report = rt.sleep(enrichment_command=backend)
    assert report["model_calls"] == 1 and report["model_input_bytes"] > 0 and report["model_output_bytes"] > 0
    assert query(rt, "SELECT value,status FROM enrichment_proposals") == [("Paris", "unconfirmed")]
    assert query(rt, "SELECT COUNT(*) FROM facts")[0][0] == 0
    assert not any(i.kind == "fact" for i in rt.retrieve_pack("person city").items)
    rt.storage.delete_event(eid, "default")
    assert not query(rt, "SELECT * FROM enrichment_proposals")
    rt.append_event(content="Berlin", session_id="new")
    fabricated = backend.copy()
    fabricated[-1] = fabricated[-1].replace("'value':r['text']", "'value':'invented'")
    bad = rt.sleep(enrichment_command=fabricated)
    assert bad["failed"] == 1 and "enrichment_unbacked_proposal" in bad["reasons"]
    assert not query(rt, "SELECT * FROM enrichment_proposals")
    rt.close()


def test_timeout_byte_caps_retry_limits_and_other_work_not_starved(tmp_path):
    rt = Runtime(str(tmp_path))
    rt.append_event(content="oversized " * 200)
    for _ in range(3):
        assert rt.sleep(max_summary_bytes=128)["failed"] == 1
    assert rt.sleep()["terminal_failed"] == 1
    rt.append_event(content="small", session_id="next")
    report = rt.sleep(enrichment_command=command("import time; time.sleep(3)"), enrichment_timeout_seconds=0.05)
    assert "enrichment_timeout" in report["reasons"]
    assert report["elapsed_ms"] < 2500
    # Existing failed job retains its settings; another neighborhood can still be queued.
    rt.append_event(content="ordinary source", session_id="third")
    assert query(rt, "SELECT COUNT(*) FROM events WHERE role!='summary'")[0][0] == 3
    rt.close()


@pytest.mark.parametrize("caps,reason", [
    ({"max_model_input_bytes": 1}, "enrichment_input_limit"),
    ({"max_model_output_bytes": 64}, "enrichment_output_limit"),
    ({"max_model_rss_mb": 1}, "enrichment_memory_limit"),
])
def test_enrichment_resource_caps_leave_no_derived_output(tmp_path, caps, reason):
    rt = Runtime(str(tmp_path))
    rt.append_event(content="Source data")
    backend = command("import time; print('x'*10000,flush=True); time.sleep(.2)")
    report = rt.sleep(enrichment_command=backend, **caps)
    assert report["failed"] == 1 and reason in report["reasons"]
    assert query(rt, "SELECT COUNT(*) FROM events WHERE role='summary'")[0][0] == 0
    assert query(rt, "SELECT COUNT(*) FROM enrichment_proposals")[0][0] == 0
    rt.close()


def test_concurrent_sleep_one_output_per_source_and_changed_snapshot(tmp_path, monkeypatch):
    rt = Runtime(str(tmp_path))
    eid = rt.append_event(content="initial")
    c = Consolidator(rt)
    original = c._commit
    def changed(job, records, content, proposals, worker):
        conn = sqlite3.connect(rt.db_path)
        with conn:
            conn.execute("UPDATE events SET content='corrected' WHERE id=?", (eid,))
        conn.close()
        return original(job, records, content, proposals, worker)
    monkeypatch.setattr(c, "_commit", changed)
    result = c.sleep()
    assert result["failed"] == 1 and "consolidation_source_changed" in result["reasons"]
    assert not query(rt, "SELECT id FROM events WHERE role='summary'")
    reports = []
    threads = [threading.Thread(target=lambda: reports.append(rt.sleep())) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join(5)
    assert len(reports) == 3
    assert query(rt, "SELECT COUNT(*) FROM events WHERE role='summary'")[0][0] == 1
    assert "corrected" in query(rt, "SELECT content FROM events WHERE role='summary'")[0][0]
    rt.close()


def test_phase06_retry_safe_connect_and_rebuild_preserves_concurrent_insert(tmp_path):
    class Encoder(FastHashEncoder):
        callback = None
        def encode(self, text):
            if self.callback:
                callback, self.callback = self.callback, None
                callback()
            return super().encode(text)
    encoder = Encoder(dimension=16)
    rt = Runtime(str(tmp_path), graph=True, encoder=encoder)
    eid = rt.append_event(content="Alice works for Acme.")
    assert rt.extract_relations(eid) == rt.extract_relations(eid)
    assert query(rt, "SELECT COUNT(*) FROM relationships")[0][0] == 1
    assert rt.sleep(connect=True)["failed"] == 0
    assert query(rt, "SELECT COUNT(*) FROM relationships")[0][0] == 1
    conn = sqlite3.connect(rt.db_path)
    with conn:
        conn.execute("UPDATE events SET content='Alice works for Beta.' WHERE id=?", (eid,))
    conn.close()
    assert rt.relationships.traverse("Alice", max_tokens=1000).paths == []
    assert rt.extract_relations(eid)
    assert rt.relationships.traverse("Alice", max_tokens=1000).paths[0]["edges"][0]["object"] == "beta"
    added = []
    encoder.callback = lambda: added.append(rt.append_event(content="Concurrent source"))
    rt.rebuild_vectors()
    assert query(rt, "SELECT COUNT(*) FROM compact_vectors WHERE id=? AND tombstoned=0", (added[0],))[0][0] == 1
    rt.close()


def test_summary_retrieval_opt_in_and_wrapper_options_not_forwarded(tmp_path):
    rt = Runtime(str(tmp_path))
    rt.append_event(content="Gateway status is ready.")
    rt.sleep()
    assert not any(i.kind == "summary" for i in rt.retrieve_pack("Gateway status").items)
    assert any(i.kind == "summary" for i in rt.retrieve_pack("Gateway status", use_summaries=True).items)
    def strict_provider(prompt):
        return prompt
    wrapped = rt.wrap(strict_provider, auto_observe_inputs=False, auto_observe_outputs=False)
    text = wrapped("Gateway status", use_summaries=True, use_graph=False, graph_limits={"max_depth": 1})
    assert "EXTRACTIVE SUMMARY" in text
    rt.close()


def test_summary_vector_filter_happens_before_candidate_limit(tmp_path):
    rt = Runtime(str(tmp_path), encoder=FastHashEncoder(dimension=16))
    eid = rt.append_event(content="Gateway status ready")
    summary_id = rt.sleep()["summaries"][0]
    rt.process_index_jobs()
    vec = rt.encoder.encode("Gateway status ready")
    assert any(r["id"] == summary_id for r in rt.vector_store.search(vec, limit=10))
    filtered = rt.vector_store.search(vec, limit=1, exclude_summaries=True)
    assert len(filtered) == 1 and filtered[0]["id"] == eid
    assert not any(i.kind == "summary" for i in rt.retrieve_pack("Gateway status", limit=1).items)
    rt.close()


def test_cancelled_support_invalidates_fact_based_summary_without_fact_manager(tmp_path):
    rt = Runtime(str(tmp_path))
    eid = rt.append_event(content="Production database host postgres.internal")
    rt.record_fact(subject="database", predicate="host", value="postgres.internal", supporting_event_ids=[eid])
    summaries = rt.sleep()["summaries"]
    conn = sqlite3.connect(rt.db_path)
    with conn:
        conn.execute("UPDATE events SET turn_status='cancelled' WHERE id=?", (eid,))
    conn.close()
    assert all(rt.storage.get_event(s, "default") is None for s in summaries)
    assert query(rt, "SELECT status FROM facts")[0][0] == "deleted"
    assert rt.sleep()["failed"] == 0
    assert not query(rt, "SELECT id FROM events WHERE role='summary'")
    rt.close()
