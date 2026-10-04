"""Regenerate a factual completion checkpoint without inventing partial accuracy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from benchmarks.public_eval.datasets import digest
from benchmarks.public_eval.report import summarize
from benchmarks.public_eval.jsonl import parse_rows


def coverage(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    raw=(root/'rows.jsonl').read_bytes()
    rows=parse_rows(raw)
    expected={(case,profile) for case in manifest['case_ids'] for profile in manifest['profiles']}
    observed=set()
    for row in rows:
        pair=(row['case_id'],row['profile'])
        if pair not in expected or pair in observed:raise ValueError('Unknown or duplicate row')
        observed.add(pair)
    complete=observed==expected
    if complete:summarize(rows,manifest)
    return {'run':str(root),'manifest_sha256':manifest['manifest_sha256'],
            'rows_sha256_at_checkpoint':hashlib.sha256(raw).hexdigest(),
            'expected_rows':len(expected),'observed_rows':len(rows),'missing_rows':len(expected-observed),
            'completed_cases':sum(all((case,p) in observed for p in manifest['profiles']) for case in manifest['case_ids']),
            'answer_failures':sum(r.get('failure') is not None for r in rows),
            'missing_scores':sum(r.get('score') is None for r in rows),'complete_coverage':complete,
            'publishable_superiority':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='append',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result={'observed_utc':datetime.now(timezone.utc).isoformat(),'runs':[coverage(root) for root in args.run],
            'qualification':'Coverage checkpoint only; does not close quality, human review, competitor or release gates.'}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
