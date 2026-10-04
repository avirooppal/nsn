"""Separate pinned local judging and tamper-bound human-review queues."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from benchmarks.public_eval.datasets import digest
from benchmarks.public_eval.runner import Ollama,JUDGE_SYSTEM
from benchmarks.public_eval.scoring import official_longmem_prompt
from benchmarks.public_eval.report import summarize


def load_run(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest['manifest_sha256']:
        raise ValueError('Saved manifest checksum mismatch')
    rows=[json.loads(line) for line in (root/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    summarize(rows,manifest)  # Require complete paired input, not a favorable subset.
    return manifest,rows


def independent_identity(answer_manifest,judge):
    if answer_manifest['model']['digest']==judge.identity['digest']:
        raise ValueError('Independent judge cannot use the answering checkpoint')


def grade(run_directory,output,assets,model,checkpoint):
    manifest,rows=load_run(run_directory)
    client=Ollama(model,checkpoint,output=64)
    independent_identity(manifest,client)
    assets=Path(assets)
    sources=json.loads((assets/'sources.json').read_text(encoding='utf-8'))
    name='longmemeval__src__evaluation__evaluate_qa.py'
    prompt_fn=official_longmem_prompt(assets/name,sources['files'][name]['sha256'])
    destination=Path(output);destination.mkdir(parents=True,exist_ok=False)
    original_sha=manifest['manifest_sha256']
    original_rows_sha=hashlib.sha256((Path(run_directory)/'rows.jsonl').read_bytes()).hexdigest()
    manifest={**manifest,'answer_manifest_sha256':original_sha,'judge':'independent local checkpoint, upstream prompt; not official reference judge','independent_judge':{'identity':client.identity,'show':client.show,'system':JUDGE_SYSTEM,'output_tokens':64,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    manifest['answer_rows_sha256']=original_rows_sha
    manifest['manifest_sha256']=digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for row in rows:
        if row['dataset']=='longmemeval':
            row['previous_judge_score_excluded']=True
            for key in ('judge_response','judge_prompt','judge_system','judge_ms'):
                row.pop(key,None)
            row['score']=0 if row['failure'] else None
            row['judge_failure']=None;row['judge_called']=False
            row['scorer']='upstream prompt, independent local judge; reference/human validation pending'
            if not row['failure']:
                prompt=prompt_fn(row['category'],row['question'],row['gold_answer'],row['answer'],abstention='_abs' in row['case_id'])
                row['judge_called']=True;row['judge_prompt']=prompt;row['judge_system']=JUDGE_SYSTEM
                started=time.perf_counter()
                try:
                    text,response=client.generate(prompt,output=64,system=JUDGE_SYSTEM)
                    row['judge_response']=response
                    if text.casefold() not in ('yes','no'):raise ValueError('Nonbinary independent judge response')
                    row['score']=int(text.casefold()=='yes')
                except Exception as exc:row['judge_failure']=repr(exc)
                row['judge_ms']=(time.perf_counter()-started)*1000
        with (destination/'rows.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(row['case_id'],row['profile'],row['score'],flush=True)
    (destination/'summary.json').write_text(json.dumps(summarize(rows,manifest),indent=2),encoding='utf-8')


def review_queue(manifest,rows,per_category=5):
    if per_category<1:raise ValueError('Review sample must be positive')
    groups={}
    for row in sorted(rows,key=lambda r:digest([manifest['manifest_sha256'],r['case_id'],r['profile']])):
        group=row['dataset']+':'+row['category']
        bucket=groups.setdefault(group,[])
        if len(bucket)>=per_category:continue
        bucket.append({'case_id':row['case_id'],'profile':row['profile'],'row_sha256':digest(row),'question':row['question'],'reference_answer':row['gold_answer'],'predicted_answer':row['answer'],'failure':row['failure'],'label':None,'rationale':None})
    return {'manifest_sha256':manifest['manifest_sha256'],'per_category':per_category,'sampling':'fixed digest order per dataset/category, includes failures','reviews':[review for bucket in groups.values() for review in bucket]}


def validate_reviews(manifest,rows,queue):
    if queue['manifest_sha256']!=manifest['manifest_sha256']:raise ValueError('Review manifest changed')
    expected=review_queue(manifest,rows,queue['per_category'])
    required={(r['case_id'],r['profile']):r for r in expected['reviews']}
    lookup={(r['case_id'],r['profile']):r for r in rows};seen=set()
    for review in queue['reviews']:
        key=(review['case_id'],review['profile'])
        if key in seen or key not in lookup or review['row_sha256']!=digest(lookup[key]):raise ValueError('Invalid/duplicate/stale review row')
        if key not in required or any(review.get(field)!=value for field,value in required[key].items() if field not in ('label','rationale')):
            raise ValueError('Human review content or frozen sample changed')
        if review['label'] not in ('correct','incorrect','uncertain') or not isinstance(review['rationale'],str) or not review['rationale'].strip():raise ValueError('Human label and rationale required')
        seen.add(key)
    if not seen or seen!=set(required):raise ValueError('Incomplete frozen human review sample')
    return {'reviewed':len(seen),'correct':sum(r['label']=='correct' for r in queue['reviews']),'uncertain':sum(r['label']=='uncertain' for r in queue['reviews']),'qualification':'human sample only; not full-dataset accuracy'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--output',required=True);parser.add_argument('--assets',default='benchmarks/assets/phase09');parser.add_argument('--judge-model');parser.add_argument('--judge-digest');parser.add_argument('--reviews')
    args=parser.parse_args()
    if args.judge_model:
        if not args.judge_digest:parser.error('--judge-digest required')
        grade(args.run,args.output,args.assets,args.judge_model,args.judge_digest)
    else:
        manifest,rows=load_run(args.run)
        result=validate_reviews(manifest,rows,json.loads(Path(args.reviews).read_text(encoding='utf-8'))) if args.reviews else review_queue(manifest,rows)
        Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
