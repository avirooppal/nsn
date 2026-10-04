"""Matched graph on/off development evidence ablation, with no model or network."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import time

from nsn.runtime import Runtime


DATASET = Path(__file__).parent / "datasets" / "relationship_dev.json"


def run(dataset=DATASET):
    raw = Path(dataset).read_bytes()
    cases = json.loads(raw)["cases"]
    report = {"dataset_sha256": hashlib.sha256(raw).hexdigest(), "scope": "development evidence retrieval; no generated-answer accuracy", "cases": [], "summary": {}}
    for enabled in (False, True):
        mode = "graph_on" if enabled else "graph_off"
        mode_rows = []
        with tempfile.TemporaryDirectory(prefix="nsn_graph_ablation_") as directory:
            rt = Runtime(directory, graph=enabled)
            for case in cases:
                ns = case["id"]
                refs = []
                for i, (subject, predicate, obj) in enumerate(case["relations"]):
                    eid = rt.append_event(id=f"{ns}_{i}", namespace=ns, content=f"{subject} {predicate} {obj}.")
                    refs.append(eid)
                    if enabled:
                        rt.record_relation(subject=subject, predicate=predicate, object=obj, event_id=eid, namespace=ns, valid_from="2025-01-01T00:00:00Z")
                if case.get("correction"):
                    subject, predicate, obj = case["correction"]
                    eid = rt.append_event(id=f"{ns}_update", namespace=ns, content=f"{subject} {predicate} {obj}.")
                    if enabled:
                        rt.record_relation(subject=subject, predicate=predicate, object=obj, event_id=eid, namespace=ns, valid_from="2026-01-01T00:00:00Z", replace=True)
                    if not case.get("as_of_time"):
                        refs[-1] = eid
                if case.get("delete_last"):
                    rt.storage.delete_event(refs[-1], namespace=ns)
                if case.get("direct"):
                    eid = rt.append_event(id=f"{ns}_direct", namespace=ns, content=case["direct"])
                    refs = [eid]
                # Matched unrelated distractors across both modes.
                for i in range(5):
                    rt.append_event(namespace=ns, content=f"Unrelated project {i} operates in Vienna.")
                latencies = []
                for _ in range(20):
                    started = time.perf_counter()
                    pack = rt.retrieve_pack(case["query"], namespace=ns, as_of_time=case.get("as_of_time"), max_tokens=512)
                    latencies.append((time.perf_counter() - started) * 1000)
                source_ids = set()
                for item in pack.items:
                    source_ids.update(item.support_refs)
                    if item.kind == "event":
                        source_ids.add(item.id)
                paths = [p for p in pack.items if p.kind == "relation_path"]
                expected = case["terminal"]
                # The complete-path metric requires every gold source AND the terminal.
                # Apply the same criterion to raw-event and graph evidence packs.
                # A baseline can recover a complete chain without a graph object.
                complete = expected is not None and set(refs) <= source_ids and any(expected in p.content.casefold() for p in pack.items)
                coverage = len(set(refs) & source_ids) / len(refs) if expected or case.get("direct") else None
                row = {"mode": mode, "id": ns, "complete_path": complete if expected else None,
                       "gold_evidence_coverage": coverage, "unsupported_full_path": bool(paths and any(len(p.support_refs) >= 2 for p in paths)) if expected is None and not case.get("direct") else False,
                       "direct_recall": bool(set(refs) <= source_ids) if case.get("direct") else None,
                       "latency_p50_ms": round(statistics.median(latencies), 3),
                       "latency_p95_ms": round(sorted(latencies)[18], 3)}
                report["cases"].append(row)
                mode_rows.append(row)
            conn = sqlite3.connect(rt.db_path)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            try:
                graph_used_bytes = conn.execute("SELECT SUM(pgsize-unused) FROM dbstat WHERE name LIKE 'relationship%' OR name='unresolved_relationships'").fetchone()[0]
            except sqlite3.OperationalError:
                graph_used_bytes = None
            tables = ("relationship_entities", "relationship_aliases", "relationships", "relationship_support", "unresolved_relationships")
            # Portable logical payload measurement, explicitly distinct from disk pages.
            graph_payload_bytes = sum(len(json.dumps(tuple(row), separators=(",", ":")).encode("utf-8"))
                                      for table in tables for row in conn.execute(f"SELECT * FROM {table}"))
            relation_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            conn.close()
            scored = [r for r in mode_rows if r["complete_path"] is not None]
            coverage = [r["gold_evidence_coverage"] for r in mode_rows if r["gold_evidence_coverage"] is not None]
            report["summary"][mode] = {
                "complete_path_rate": sum(r["complete_path"] for r in scored) / len(scored),
                "mean_gold_evidence_coverage": sum(coverage) / len(coverage),
                "unsupported_full_paths": sum(r["unsupported_full_path"] for r in mode_rows),
                "direct_recall": all(r["direct_recall"] for r in mode_rows if r["direct_recall"] is not None),
                "mean_query_p50_ms": round(statistics.mean(r["latency_p50_ms"] for r in mode_rows), 3),
                "database_bytes": Path(rt.db_path).stat().st_size,
                "relationship_used_page_bytes": graph_used_bytes,
                "relationship_json_payload_bytes": graph_payload_bytes,
                "stored_relations": relation_count,
            }
            rt.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
