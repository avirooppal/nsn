"""Paired NSN versus non-NSN metrics regenerated from complete raw runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.public_eval.grading import load_run
from benchmarks.public_eval.report import paired_interval


def percentile(values,q):
    if not values:return None
    return sorted(values)[round((len(values)-1)*q)]


def compare(run):
    manifest,rows=load_run(run)
    if not manifest.get('independent_judge') and 'self-judge' in manifest.get('judge',''):
        raise ValueError('Historical answering-model self-judge scores cannot enter comparison claims')
    result={'model':manifest['model'],'provider':manifest.get('provider','local'),
            'manifest_sha256':manifest['manifest_sha256'],
            'raw_rows_sha256':hashlib.sha256((Path(run)/'rows.jsonl').read_bytes()).hexdigest(),
            'independent_judge':manifest.get('independent_judge'),'profiles':{},'paired':{},
            'scope':'Bounded core-retrieval comparison, not complete evaluation or SDK-wrapper throughput qualification'}
    for profile in manifest['profiles']:
        selected=[r for r in rows if r['profile']==profile]
        metrics={}
        for dataset in sorted({r['dataset'] for r in selected}):
            group=[r for r in selected if r['dataset']==dataset]
            scores=[r['score'] for r in group]
            metrics[dataset]={'metric':'upstream category-aware F1' if dataset=='locomo' else 'independent judge correctness; human calibration pending',
                              'mean':statistics.mean(scores) if all(s is not None for s in scores) else None,
                              'lower':sum(s or 0 for s in scores)/len(scores),
                              'upper':sum(1 if s is None else s for s in scores)/len(scores),'count':len(scores)}
        evidence=[r for r in selected if r['gold_count']]
        prompt_missing=sum(r.get('answer_llm_calls',0) and 'prompt_eval_count' not in r for r in selected)
        result['profiles'][profile]={'questions':len(selected),'quality':metrics,
            'failures':sum(bool(r['failure']) for r in selected),'failure_reasons':[r['failure'] for r in selected if r['failure']],
            'evidence_recall':statistics.mean(r['evidence_coverage'] for r in evidence) if evidence else None,
            'complete_evidence_rate':statistics.mean(r['complete_evidence'] for r in evidence) if evidence else None,
            'evidence_precision':statistics.mean(len(set(r['retrieved_ids']) & set(r['gold_ids']))/len(set(r['retrieved_ids'])) if r['retrieved_ids'] else 0 for r in evidence) if evidence else None,
            'retrieval_p50_ms':percentile([r['retrieval_ms'] for r in selected],.5),
            'retrieval_p95_ms':percentile([r['retrieval_ms'] for r in selected],.95),
            'generation_p50_ms':percentile([r['generation_ms'] for r in selected],.5),
            'generation_p95_ms':percentile([r['generation_ms'] for r in selected],.95),
            'retrieval_plus_generation_p95_ms':percentile([r['retrieval_ms']+r['generation_ms'] for r in selected],.95),
            'prompt_tokens_known':sum(r.get('prompt_eval_count',0) for r in selected),
            'output_tokens_known':sum(r.get('eval_count',0) for r in selected),'token_counts_missing_calls':prompt_missing,
            'abstentions':sum(r.get('abstained',False) for r in selected),
            'client_rss_max_bytes':max(r['client_rss_bytes'] for r in selected),
            'harness_disk_max_bytes':max(r.get('disk_bytes',0) for r in selected),
            'memory_management_llm_calls':sum(r.get('management_llm_calls',0) for r in selected)}
    for dataset in sorted({r['dataset'] for r in rows}):
        part=[r for r in rows if r['dataset']==dataset]
        for target in ('nsn_lexical','nsn_semantic'):
            for baseline in ('stateless','rolling','bm25','hybrid','full_context'):
                if target in manifest['profiles'] and baseline in manifest['profiles']:
                    result['paired'][dataset+':'+target+'-'+baseline]=paired_interval(part,baseline,target)
    result['limitations']=[
        'Small predeclared sample; cluster intervals exploratory and unreliable for broad generalization; one LongMemEval cluster has no valid interval.',
        'Client RSS/disk include the shared benchmark harness and cannot be interpreted as incremental NSN overhead; isolated installed profiles are separate.',
        'First-use semantic retrieval includes vector materialization and shared cache effects, not isolated steady-state latency.',
        'Retrieval plus generation excludes offline corpus build/ingestion; raw artifacts retain build costs.',
        'Stale/conflict rates, hallucination rates and agent task success need validated annotations/task suites; unavailable here.',
        'Provider dollar cost, server peak memory and time-to-first-token are unavailable; token counts and nonstreaming wall latency are reported.',
        'Core retrieval is evaluated; SDK wrapping and optional graph/consolidation/procedures are not exercised by this QA workload.']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    reports=[compare(run) for run in args.runs]
    manifests=[json.loads((Path(run)/'manifest.json').read_text()) for run in args.runs]
    for key in ('case_ids','profiles','context_tokens','reserved_output_tokens','memory_budget_utf8_bytes','sources','embedding','memory_layer_source_sha256','system_prompt','answer_prompt_template'):
        if any(m[key]!=manifests[0][key] for m in manifests):raise ValueError('Unmatched comparison input: '+key)
    Path(args.output).write_text(json.dumps({'publishable_superiority':False,'models':reports},indent=2),encoding='utf-8')
