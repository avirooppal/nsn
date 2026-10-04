import copy
import json
from types import SimpleNamespace
import pytest
from benchmarks.public_eval import grading


def fixture_run(tmp_path, monkeypatch,question='port?'):
    source=tmp_path/'answer';source.mkdir()
    name='longmemeval__src__evaluation__evaluate_qa.py'
    pins={'files':{name:{'sha256':'pinned'}}}
    (tmp_path/'sources.json').write_text(json.dumps(pins))
    rows=[{'case_id':str(i),'cluster':str(i),'dataset':'longmemeval','category':'single-session-user',
           'profile':p,'question':question,'gold_answer':'8080','answer':'8080','score':None,
           'failure':None,'judge_failure':None,'judge_called':False,'scorer':'unconfigured',
           'complete_evidence':False,'gold_count':1,'generation_ms':0,'client_rss_bytes':10,'retrieval_ms':1}
          for i in range(2) for p in ('bm25','nsn')]
    (source/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'case_ids':['0','1'],'profiles':['bm25','nsn'],'manifest_sha256':'answer-pin',
              'model':{'name':'answer','digest':'answer-digest'},'sources':pins}
    monkeypatch.setattr(grading,'load_run',lambda root:(copy.deepcopy(manifest),copy.deepcopy(rows)))
    monkeypatch.setattr(grading,'official_longmem_prompt',lambda *a:lambda *a,**k:'fixed rubric')
    calls=[]
    def generate(*a,**k):
        calls.append(a)
        if len(calls)==1:raise TimeoutError('retained judge timeout')
        return 'yes',{'message':{'content':'yes'}}
    client=SimpleNamespace(identity={'name':'judge','digest':'judge-digest'},show={},generate=generate)
    monkeypatch.setattr(grading,'model_client',lambda *a,**k:client)
    return source,calls


def test_grading_resume_preserves_unicode_content_and_failed_grade(tmp_path,monkeypatch):
    source,calls=fixture_run(tmp_path,monkeypatch,question='port?\u2028second\u2029third')
    out=tmp_path/'graded'
    grading.grade(source,out,tmp_path,'judge','judge-digest',max_new_rows=1)
    original=(out/'rows.jsonl').read_bytes()
    assert '\u2028'.encode() in original
    grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    from benchmarks.public_eval.jsonl import read_rows
    rows=read_rows(out/'rows.jsonl')
    assert len(calls)==4 and len(rows)==4
    assert all(row['question']=='port?\u2028second\u2029third' for row in rows)
    assert rows[0]['score'] is None and rows[0]['judge_failure']
    assert (out/'rows.jsonl').read_bytes().startswith(original)


def test_judge_resume_keeps_failures_and_never_duplicates_model_calls(tmp_path, monkeypatch):
    source,calls=fixture_run(tmp_path,monkeypatch)
    out=tmp_path/'graded'
    grading.grade(source,out,tmp_path,'judge','judge-digest',max_new_rows=1)
    assert not (out/'summary.json').exists()
    assert json.loads((out/'progress.json').read_text())['missing_rows']==3
    grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    assert len(calls)==4
    rows=[json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==4 and len({(r['case_id'],r['profile']) for r in rows})==4
    assert rows[0]['score'] is None and 'timeout' in rows[0]['judge_failure']
    summary=json.loads((out/'summary.json').read_text())
    assert summary['profiles']['bm25']['answer_metrics']['longmemeval']['score'] is None
    grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    assert len(calls)==4


def test_judge_resume_rejects_changed_budget_duplicates_and_altered_answers(tmp_path, monkeypatch):
    source,calls=fixture_run(tmp_path,monkeypatch)
    out=tmp_path/'graded'
    grading.grade(source,out,tmp_path,'judge','judge-digest',max_new_rows=1)
    with pytest.raises(ValueError,match='changed judging'):
        grading.grade(source,out,tmp_path,'judge','judge-digest',output_tokens=512,resume=True)
    path=out/'rows.jsonl';original=path.read_text()
    path.write_text(original+original)
    with pytest.raises(ValueError,match='duplicate'):
        grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    row=json.loads(original);row['answer']='altered'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='changed graded row'):
        grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    assert len(calls)==1


def test_judge_cannot_change_upstream_prompt_pin_between_answering_and_grading(tmp_path, monkeypatch):
    source,calls=fixture_run(tmp_path,monkeypatch)
    (tmp_path/'sources.json').write_text(json.dumps({'files':{'longmemeval__src__evaluation__evaluate_qa.py':{'sha256':'different'}}}))
    with pytest.raises(ValueError,match='source pin'):
        grading.grade(source,tmp_path/'graded',tmp_path,'judge','judge-digest')
    assert not calls


def test_resume_rejects_score_that_contradicts_preserved_judge_response(tmp_path,monkeypatch):
    source,calls=fixture_run(tmp_path,monkeypatch)
    out=tmp_path/'graded'
    grading.grade(source,out,tmp_path,'judge','judge-digest',max_new_rows=2)
    path=out/'rows.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]['score']==1 and rows[1]['judge_response']['message']['content']=='yes'
    rows[1]['score']=0
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='contradicts logged outcome'):
        grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    assert len(calls)==2


def test_resume_rejects_modified_official_rubric_before_judging(tmp_path,monkeypatch):
    source,calls=fixture_run(tmp_path,monkeypatch)
    out=tmp_path/'graded'
    grading.grade(source,out,tmp_path,'judge','judge-digest',max_new_rows=1)
    path=out/'rows.jsonl';row=json.loads(path.read_text())
    row['judge_prompt']='altered rubric'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='contradicts logged outcome'):
        grading.grade(source,out,tmp_path,'judge','judge-digest',resume=True)
    assert len(calls)==1
