import json
import pytest
from benchmarks.public_eval import compare


def test_comparison_keeps_failures_unknown_grades_and_resource_scope(tmp_path,monkeypatch):
    rows=[]
    for profile in ('stateless','nsn_semantic'):
        for case,score,failure in (('a',1,None),('b',0,'timeout')):
            rows.append({'case_id':case,'profile':profile,'dataset':'locomo','cluster':'one',
                'score':score,'failure':failure,'gold_count':1,'gold_ids':['gold'],
                'retrieved_ids':['gold','irrelevant'] if case=='a' else [],
                'evidence_coverage':1 if case=='a' else 0,'complete_evidence':case=='a',
                'retrieval_ms':1,'generation_ms':10,'answer_llm_calls':1,
                'client_rss_bytes':100,'disk_bytes':200,'management_llm_calls':0})
        rows.append({**rows[-1],'case_id':'c','dataset':'longmemeval','score':None,'failure':None})
    manifest={'model':{'name':'real-model-pin'},'manifest_sha256':'pin','profiles':['stateless','nsn_semantic']}
    monkeypatch.setattr(compare,'load_run',lambda run:(manifest,rows))
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result=compare.compare(tmp_path)
    nsn=result['profiles']['nsn_semantic']
    assert nsn['questions']==3 and nsn['failures']==1
    assert nsn['quality']['locomo']['mean']==.5
    assert nsn['quality']['longmemeval']['mean'] is None
    assert nsn['quality']['longmemeval']['lower']==0 and nsn['quality']['longmemeval']['upper']==1
    assert nsn['token_counts_missing_calls']==3 and nsn['prompt_tokens_known']==0
    assert nsn['evidence_precision']==pytest.approx(1/6)
    assert result['paired']['locomo:nsn_semantic-stateless']['ci95'] is None
    assert any('incremental NSN overhead' in limit for limit in result['limitations'])


def test_percentiles_preserve_empty_and_actual_observed_values():
    assert compare.percentile([],.95) is None
    assert compare.percentile([4,1,2,3],.5)==3
    assert compare.percentile([4,1,2,3],.95)==4


def test_comparison_rejects_historical_answering_model_self_judge(tmp_path,monkeypatch):
    manifest={'judge':'local answering-model self-judge, NOT official reference judge',
              'model':{'name':'historical-model'},'profiles':['stateless'],'manifest_sha256':'pin'}
    rows=[{'case_id':'a','profile':'stateless','dataset':'longmemeval','cluster':'one',
           'score':1,'failure':None,'gold_count':0,'retrieval_ms':1,'generation_ms':10,
           'client_rss_bytes':100}]
    (tmp_path/'rows.jsonl').write_text(json.dumps(rows[0])+'\n')
    monkeypatch.setattr(compare,'load_run',lambda run:(manifest,rows))
    with pytest.raises(ValueError,match='self-judge'):
        compare.compare(tmp_path)
