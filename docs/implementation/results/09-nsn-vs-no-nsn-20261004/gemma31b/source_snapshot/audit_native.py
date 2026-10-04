"""Validate native full haystacks in bounded memory without generating answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from benchmarks.public_eval.datasets import iter_native


def audit(path,dataset):
    counts=Counter();ids=set();turns=0;gold=0
    for case in iter_native(path,dataset):
        if case['id'] in ids:raise ValueError('Duplicate question ID')
        ids.add(case['id']);counts[case['category']]+=1
        events={e['id']:e for e in case['events']}
        if len(events)!=len(case['events']):raise ValueError('Duplicate native turn ID')
        if not set(case['gold_ids'])<=set(events):raise ValueError('Gold evidence outside native corpus')
        if dataset=='longmemeval' and not set(case['answer_session_ids'])<={e['session'] for e in case['events']}:raise ValueError('Answer session outside haystack')
        turns+=len(events);gold+=len(case['gold_ids'])
    hasher=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):hasher.update(chunk)
    return {'file':str(path),'sha256':hasher.hexdigest(),'questions':len(ids),'question_ids':sorted(ids),'categories':dict(counts),'native_turns_across_questions':turns,'annotated_evidence_turns':gold,'conversion':'streaming native; session/date/speaker/role preservation; no generation','quality_evaluated':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('file');parser.add_argument('--dataset',default='longmemeval');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.file,args.dataset)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(result['questions'],result['categories'])
