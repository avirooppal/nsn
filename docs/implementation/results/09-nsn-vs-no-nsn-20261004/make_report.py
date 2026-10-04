"""Render this frozen paired benchmark from comparison JSON and raw rows."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.public_eval.jsonl import read_rows

root=Path(__file__).resolve().parent
comparison=json.loads((root/'comparison.json').read_text(encoding='utf-8'))
labels=['qwen2b','qwen4b','qwen8b','gemma31b','qwen4b-nothink']
profiles=['stateless','rolling','full_context','bm25','hybrid','nsn_lexical','nsn_semantic']
lines=['# NSN versus non-NSN: measured paired benchmark','',
       'Completed bounded comparison: 15 frozen public questions × 7 configurations × 4 real answering models = **420 original rows**, plus **105 separately pinned template-change attempt 4B rows**. Nine LoCoMo questions cover all five categories; six full-S LongMemEval questions cover its six categories. This is an exploratory sample, not the complete public campaign or a production/superiority qualification.','',
       'Same 2,048-token input window, 32 output tokens, 512 bytes of memory evidence, seed 42 and decoding per model/configuration. Cloud 31B is a separate reference. LongMemEval uses an independent Cloud GPT-OSS 120B judge; human calibration remains pending. See [protocol](protocol.md) and [frozen sample](sample.json).','',
       'Native Qwen3 4B has an unconditional opening thinking block in its installed template. Despite requesting `think=false`, it spent the answer budget on reasoning; its native scores are retained but not qualified as a no-thinking model-size comparison. The additional `nsn-benchmark-qwen3-4b-nothink:latest` run uses the same weights with the installed 8B Qwen3 no-thinking template, new digest and unchanged original model. GGUF metadata identifies these as Qwen3-4B-Thinking-2507 weights. The template attempt still emits reasoning and exhausts the budget, so both 4B configurations are unqualified for a no-thinking comparison. See [failed template-attempt provenance](qwen4b-template-repair.json). No samples or budgets changed.','']
extra={}
for dataset in ('locomo','longmemeval'):
    lines+=['## '+('LoCoMo upstream F1 (%)' if dataset=='locomo' else 'LongMemEval independent-judge correctness (%) — provisional'),'',
            '| Model | No memory | Recent context | Full context | BM25 | Hybrid RAG | NSN lexical | NSN semantic |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in comparison['models']:
        values=[]
        for profile in profiles:
            quality=model['profiles'][profile]['quality'][dataset]
            values.append(f"{quality['mean']*100:.1f}" if quality['mean'] is not None else f"{quality['lower']*100:.1f}–{quality['upper']*100:.1f} (ungraded bounds)")
        lines.append('| '+model['model']['name']+' | '+' | '.join(values)+' |')
    lines.append('')
lines+=['## Measured latency, evidence and tokens','',
        'Retrieval includes first-use vector materialization and shared-cache effects. Retrieval plus generation excludes offline native-corpus preparation. Percentiles use the nearest observed order statistic; only 15 observations per configuration. Full-context failure paths are not useful inference throughput.','',
        '| Model | Configuration | Evidence recall | Evidence precision | Complete evidence | Retrieval p50 / p95 ms | Generation p50 / p95 ms | Retrieval + generation p95 ms | Prompt / output tokens | Failures |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for label,model in zip(labels,comparison['models']):
    rows=read_rows(root/(label+'-graded')/'rows.jsonl')
    extra[label]={}
    for profile,m in model['profiles'].items():
        lines.append(f"| {model['model']['name']} | {profile} | {m['evidence_recall']:.3f} | {m['evidence_precision']:.3f} | {m['complete_evidence_rate']:.3f} | {m['retrieval_p50_ms']:.1f} / {m['retrieval_p95_ms']:.1f} | {m['generation_p50_ms']:.1f} / {m['generation_p95_ms']:.1f} | {m['retrieval_plus_generation_p95_ms']:.1f} | {m['prompt_tokens_known']} / {m['output_tokens_known']} | {m['failures']} |")
        selected=[r for r in rows if r['profile']==profile]
        expected=[r for r in selected if (r['dataset']=='locomo' and r['category']=='5') or '_abs' in r['case_id']]
        categories={}
        for category in sorted({r['dataset']+':'+str(r['category']) for r in selected}):
            group=[r for r in selected if r['dataset']+':'+str(r['category'])==category]
            categories[category]={'count':len(group),'score_mean':statistics.mean(r['score'] for r in group) if all(r['score'] is not None for r in group) else None,'failures':sum(bool(r['failure']) for r in group)}
        evidence=[r for r in selected if r['gold_count']]
        rr=[];ndcg=[]
        for row in evidence:
            relevant=set(row['gold_ids']);ids=list(dict.fromkeys(row['retrieved_ids']))
            rr.append(next((1/rank for rank,id in enumerate(ids,1) if id in relevant),0))
            dcg=sum(1/math.log2(rank+1) for rank,id in enumerate(ids,1) if id in relevant)
            ideal=sum(1/math.log2(rank+1) for rank in range(1,min(len(ids),len(relevant))+1))
            ndcg.append(dcg/ideal if ideal else 0)
        extra[label][profile]={'categories':categories,'selected_evidence_mrr':statistics.mean(rr) if rr else None,
            'selected_evidence_ndcg':statistics.mean(ndcg) if ndcg else None,
            'expected_abstention_questions':len(expected),'expected_abstention_rate':sum(r.get('abstained',False) for r in expected)/len(expected) if expected else None,
            'answerable_abstentions':sum(r.get('abstained',False) for r in selected if r not in expected),
            'length_terminated_answers':sum(r.get('generation_response',{}).get('done_reason')=='length' for r in selected),
            'successful_provider_calls':sum('generation_response' in r for r in selected),
            'nsn_internal_retrieval_p50_ms':statistics.median(r['nsn_diagnostics']['timings']['total_retrieve_ms'] for r in selected if r.get('nsn_diagnostics',{}).get('timings')) if any(r.get('nsn_diagnostics',{}).get('timings') for r in selected) else None}
lines+=['','## Resource costs and interpretation','',
        'Per-model client RSS/disk, token-count failures and management-call counts are in [comparison.json](comparison.json). The shared harness stores the native corpus even for the no-memory profile; those RSS/disk samples cannot measure incremental NSN overhead. Every configuration here has extraction/consolidation disabled. Original ingestion/embedding/cache-build costs remain in each raw row and regenerated run summary.','',
        'Separate previously measured installed-wheel 10k profiles (same SDK candidate, different workload): minimal append p95 **2.71ms**, repeated retrieval **2.13ms**, fresh-query retrieval **85.88ms**; optional ONNX/deferred append **5.52ms**, repeated retrieval **48.35ms**, fresh-query retrieval **143.88ms**, incremental memory **231.12MiB**. Fresh-query targets still fail; deferred indexing does not qualify default inline append. See [minimal](../10-completion-minimal-profile-20261004.json) and [semantic](../10-completion-onnx-profile-20261004.json). These are not new baseline-versus-NSN process-isolation measurements.','',
        '## Uncertainty and unavailable metrics','',
        'All paired differences and cluster intervals are in comparison.json. LoCoMo has only two sampled conversation clusters, so its intervals are exploratory and weak for generalization. LongMemEval has one connected cluster and no valid conversation-cluster confidence interval. Fixed profile order, shared caches and model loading affect wall latency; it is not a randomized performance qualification. No comparison establishes general superiority.','',
        'Abstention uses literal canonical phrases, not a human-validated semantic classifier. Additional category scores, abstention, MRR and NDCG over the selected evidence appear in [additional-metrics.json](additional-metrics.json). Native raw answers, full prompts, provider token counts/timings and grading responses are retained in the model directories. Human review queues are unfilled.','',
        'Validated hallucination/stale/conflict rates and agent task success are unavailable in this sample. Remote memory, billed dollar cost and time-to-first-token are unavailable. Failures and missing scores/counts remain visible rather than being silently omitted. Core retrieval, not full SDK-wrapper throughput or optional graph/consolidation/procedure behavior, is measured.']
(root/'additional-metrics.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
(root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() and path.name in ('manifest.json','rows.jsonl','sample.json','comparison.json','additional-metrics.json','report.md','make_report.py','protocol.md','human-review-queue.json','regenerated-summary.json','provenance-verification.json','qwen4b-template-repair.json','tests.txt')}
(root/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print('Rendered report and checksums from raw-artifact comparison.')
