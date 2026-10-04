import json
import pytest
from benchmarks.public_eval.datasets import digest
from scripts.audit_completion import coverage


def test_partial_coverage_counts_failures_and_missing_scores(tmp_path):
    manifest={'case_ids':['a','b'],'profiles':['nsn','bm25']}
    manifest['manifest_sha256']=digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    rows=[{'case_id':'a','profile':'nsn','failure':'timeout','score':0},
          {'case_id':'a','profile':'bm25','failure':None,'score':None}]
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result=coverage(tmp_path)
    assert result['expected_rows']==4 and result['missing_rows']==2
    assert result['completed_cases']==1 and result['answer_failures']==1
    assert result['missing_scores']==1
    assert not result['complete_coverage'] and not result['publishable_superiority']
    (tmp_path/'rows.jsonl').write_text(json.dumps(rows[0])+'\n'+json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='duplicate'):coverage(tmp_path)
