"""Matched explicit sleep on/off; measured evidence, foreground cost, sampled RSS."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import threading
import time

import psutil
from nsn import Runtime


DATASET = Path(__file__).parent / "datasets" / "consolidation_dev.json"


def run():
    raw = DATASET.read_bytes()
    groups = json.loads(raw)["groups"]
    results = {"dataset_sha256":hashlib.sha256(raw).hexdigest(), "scope":"development extractive evidence, lexical-only, no generated-answer metric", "modes":{}}
    for enabled in (False, True):
        with tempfile.TemporaryDirectory(prefix="nsn_sleep_ablation_") as directory:
            rt = Runtime(directory)
            for group in groups:
                for i, text in enumerate(group["sources"]):
                    rt.append_event(id=f"{group['id']}-{i}", namespace=group["id"], session_id=group["id"], content=text)
            process = psutil.Process()
            rss = [process.memory_info().rss]
            stopped = threading.Event()
            def sample():
                while not stopped.wait(.002):
                    rss.append(process.memory_info().rss)
            sampler = threading.Thread(target=sample)
            sampler.start()
            started = time.perf_counter()
            reports = []
            try:
                if enabled:
                    reports = [rt.sleep(namespace=g["id"]) for g in groups]
            finally:
                stopped.set()
                sampler.join()
                rss.append(process.memory_info().rss)
            sleep_ms = (time.perf_counter()-started)*1000
            cases = []
            for g in groups:
                times = []
                for _ in range(20):
                    begin = time.perf_counter()
                    pack = rt.retrieve_pack(g["query"], namespace=g["id"], limit=1, max_tokens=512, use_summaries=enabled)
                    times.append((time.perf_counter()-begin)*1000)
                text = pack.rendered_text
                sources = set()
                for item in pack.items:
                    if item.kind == "event": sources.add(item.id)
                    sources.update(item.support_refs)
                gold_ids = {f"{g['id']}-{i}" for i in range(len(g["sources"]))}
                cases.append(dict(id=g["id"], complete_values=all(v in text for v in g["gold"]),
                                  gold_source_coverage=len(gold_ids & sources)/len(gold_ids),
                                  foreground_p50_ms=round(statistics.median(times),3), foreground_p95_ms=round(sorted(times)[18],3)))
            conn = sqlite3.connect(rt.db_path)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            original_count = conn.execute("SELECT COUNT(*) FROM events WHERE role!='summary'").fetchone()[0]
            summaries = conn.execute("SELECT COUNT(*) FROM events WHERE role='summary'").fetchone()[0]
            conn.close()
            results["modes"]["sleep_on" if enabled else "sleep_off"] = dict(
                cases=cases, complete_value_rate=sum(c["complete_values"] for c in cases)/len(cases),
                mean_gold_source_coverage=statistics.mean(c["gold_source_coverage"] for c in cases),
                mean_foreground_p50_ms=round(statistics.mean(c["foreground_p50_ms"] for c in cases),3),
                sleep_elapsed_ms=round(sleep_ms,3), rss_start_bytes=rss[0], max_sampled_rss_bytes=max(rss),
                max_sampled_rss_increase_bytes=max(rss)-rss[0], original_events=original_count, summaries=summaries,
                database_bytes=Path(rt.db_path).stat().st_size, model_calls=sum(r["model_calls"] for r in reports),
                sleep_reports=reports)
            rt.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{m:v for m,v in data.items() if m not in ("cases","sleep_reports")} for k,data in results["modes"].items()},indent=2))
