"""Raw-artifact summaries with paired conversation-cluster bootstrap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

def paired_interval(rows, baseline, target, samples=2000, seed=42):
    by_id=defaultdict(dict)
    for r in rows: by_id[r['case_id']][r['profile']]=r
    clusters=defaultdict(list)
    for q,profiles in by_id.items():
        if baseline not in profiles or target not in profiles:
            raise ValueError('Unpaired question coverage')
        a,b=profiles[baseline],profiles[target]
        if a.get('score') is None or b.get('score') is None:
            continue
        clusters[a['cluster']].append(b['score']-a['score'])
    if not clusters: return {'delta':None,'ci95':None,'clusters':0}
    keys=sorted(clusters);rng=random.Random(seed);draws=[]
    for _ in range(samples):
        values=[v for key in rng.choices(keys,k=len(keys)) for v in clusters[key]]
        draws.append(statistics.mean(values))
    draws.sort()
    return {'delta':statistics.mean([v for values in clusters.values() for v in values]),'ci95':[draws[int(.025*samples)],draws[min(samples-1,int(.975*samples))]],'clusters':len(keys),'scored_pairs':sum(map(len,clusters.values())),'samples':samples,'seed':seed,'multiple_comparisons':'exploratory unadjusted; no confirmatory winner claim'}

def summarize(rows,manifest):
    expected=set(manifest['case_ids'])
    result={'status':'evaluation_incomplete_qualification' if manifest.get('split')=='evaluation' else 'development_smoke_only','publishable_superiority':False,'manifest_sha256':manifest['manifest_sha256'],'profiles':{},'comparisons':{}}
    for profile in manifest['profiles']:
        selected=[r for r in rows if r['profile']==profile]
        if len(selected)!=len(expected) or {r['case_id'] for r in selected}!=expected:
            raise ValueError('Question coverage differs from frozen manifest')
        categories=defaultdict(list)
        for r in selected: categories[r['dataset']+':'+r['category']].append(r)
        result['profiles'][profile]={'questions':len(selected),'failures':sum(r['failure'] is not None for r in selected),'answer_metrics':{},'complete_evidence_rate':statistics.mean(r['complete_evidence'] for r in selected if r['gold_count']) if any(r['gold_count'] for r in selected) else None,'categories':{k:{'count':len(v),'failures':sum(r['failure'] is not None for r in v)} for k,v in categories.items()},'generation_ms_total':sum(r['generation_ms'] for r in selected),'prompt_tokens_total':sum(r.get('prompt_eval_count',0) for r in selected),'output_tokens_total':sum(r.get('eval_count',0) for r in selected),'sampled_client_rss_max':max(r['client_rss_bytes'] for r in selected),'retrieval_ms_median':statistics.median(r['retrieval_ms'] for r in selected),'judge_calls':sum(r.get('judge_called',False) for r in selected)}
        for dataset in {r['dataset'] for r in selected}:
            group=[r for r in selected if r['dataset']==dataset]
            scored=[r for r in group if r['score'] is not None]
            result['profiles'][profile]['answer_metrics'][dataset]={'score':statistics.mean(r['score'] for r in group) if len(scored)==len(group) else None,'score_lower_bound':sum(r['score'] or 0 for r in group)/len(group),'score_upper_bound':sum(1 if r['score'] is None else r['score'] for r in group)/len(group),'scored':len(scored),'total':len(group),'judge_missing':sum(r.get('judge_failure') is not None for r in group),'scorer':group[0]['scorer']}
        expected_abstentions=[r for r in selected if (r['dataset']=='locomo' and r['category']=='5') or (r['dataset']=='longmemeval' and '_abs' in r['case_id'])]
        result['profiles'][profile]['abstention']={'expected_questions':len(expected_abstentions),'abstained_rate':sum(r.get('abstained',False) for r in expected_abstentions)/len(expected_abstentions) if expected_abstentions else None,'answerable_abstentions':sum(r.get('abstained',False) for r in selected if r not in expected_abstentions)}
        result['profiles'][profile]['unsupported_metrics']=['stale-answer/conflict rates require validated old-answer/conflict annotations','task success requires an agent task suite; QA F1 is not task success']
    for dataset in sorted({r['dataset'] for r in rows}):
        part=[r for r in rows if r['dataset']==dataset]
        for target in manifest['profiles']:
            if target!='bm25': result['comparisons'][dataset+':'+target+'-bm25']=paired_interval(part,'bm25',target)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_directory');parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.run_directory)
    summary=summarize([json.loads(line) for line in (root/'rows.jsonl').read_text(encoding='utf-8').splitlines()],json.loads((root/'manifest.json').read_text(encoding='utf-8')))
    text=json.dumps(summary,indent=2)
    if args.output: Path(args.output).write_text(text,encoding='utf-8')
    else: print(text)
