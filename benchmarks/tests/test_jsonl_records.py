import json
import subprocess
import sys
import pytest

from benchmarks.public_eval.datasets import digest
from benchmarks.public_eval.ledger import open_run
from benchmarks.public_eval import grading
from scripts.audit_completion import coverage
from benchmarks.public_eval.jsonl import parse_rows


def test_unicode_content_survives_resume_grading_and_coverage(tmp_path,monkeypatch):
    tmp_path=tmp_path/'run'
    manifest={'case_ids':['q','missing'],'profiles':['nsn']}
    manifest['manifest_sha256']=digest(manifest)
    frozen={'split':{'q':'evaluation','missing':'evaluation'}}
    open_run(tmp_path,manifest,frozen)
    row={'case_id':'q','profile':'nsn','failure':None,'score':None,
         'answer':'first\u2028second\u2029third\x85fourth'}
    path=tmp_path/'rows.jsonl'
    path.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
    original=path.read_bytes()
    assert open_run(tmp_path,manifest,frozen,resume=True)[1]==[row]
    monkeypatch.setattr(grading,'summarize',lambda rows,manifest:None)
    assert grading.load_run(tmp_path)[1]==[row]
    result=coverage(tmp_path)
    assert result['observed_rows']==1 and result['missing_rows']==1
    assert path.read_bytes()==original


@pytest.mark.parametrize('suffix',[b'{"answer":"unfinished',b'{bad}\n',b'\n'])
def test_resume_and_audit_reject_malformed_or_truncated_records(tmp_path,suffix):
    tmp_path=tmp_path/'run'
    manifest={'case_ids':['q'],'profiles':['nsn']}
    manifest['manifest_sha256']=digest(manifest)
    frozen={'split':{'q':'evaluation'}}
    open_run(tmp_path,manifest,frozen)
    path=tmp_path/'rows.jsonl'
    path.write_bytes(suffix)
    for read in (lambda:open_run(tmp_path,manifest,frozen,resume=True),lambda:coverage(tmp_path)):
        with pytest.raises(json.JSONDecodeError):read()
    assert path.read_bytes()==suffix


@pytest.mark.parametrize('raw',[b'',b'{"id":1}',b'{"id":1}\r\n'])
def test_empty_and_valid_final_records(raw):
    assert parse_rows(raw)==([] if not raw else [{'id':1}])


def test_report_cli_handles_unicode_records_without_changing_raw_artifact(tmp_path):
    manifest={'case_ids':['q'],'profiles':['bm25']}
    manifest['manifest_sha256']=digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    row={'case_id':'q','profile':'bm25','cluster':'q','dataset':'locomo','category':'1',
         'answer':'alpha\u2028beta\u2029gamma','failure':None,'score':1,'scorer':'fixture',
         'complete_evidence':True,'gold_count':1,'generation_ms':0,'retrieval_ms':0,'client_rss_bytes':0}
    path=tmp_path/'rows.jsonl'
    path.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
    original=path.read_bytes();output=tmp_path/'report.json'
    result=subprocess.run([sys.executable,'-m','benchmarks.public_eval.report',str(tmp_path),
                           '--output',str(output)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(output.read_text())['profiles']['bm25']['answer_metrics']['locomo']['score']==1
    assert path.read_bytes()==original
