import json
from pathlib import Path
import pytest
from benchmarks.public_eval.datasets import load_native,freeze,smoke_cases
from benchmarks.public_eval.runner import Ollama,packed,lexical,SYSTEM,JUDGE_SYSTEM
from benchmarks.public_eval.report import summarize,paired_interval
from benchmarks.public_eval.scoring import functions
from nsn import Runtime
from benchmarks.public_eval.ledger import select_cases,open_run
from benchmarks.public_eval.datasets import digest
from benchmarks.public_eval.corpus import freeze_stream

def test_streaming_freeze_matches_existing_split_and_rejects_duplicate_questions():
    cases=[{'id':str(i),'group':str(i),'dataset':'longmemeval','events':[{'id':str(i),'session':str(i),'date':'date','role':'user','speaker':'user','content':'same' if i<2 else str(i)}]} for i in range(5)]
    frozen,index=freeze_stream(iter(cases))
    assert frozen==freeze(cases)
    assert len(index)==5 and all('events' not in c for c in index)
    with pytest.raises(ValueError):freeze_stream(iter(cases+cases[:1]))
from benchmarks.public_eval.datasets import convert_native

def test_full_haystack_repeated_session_ids_preserve_dates_and_gold_identity():
    data=[{'question_id':'q','question_type':'temporal-reasoning','question':'when?','answer':'second','question_date':'today','haystack_dates':['first','second'],'haystack_session_ids':['s','s'],'haystack_sessions':[[{'role':'user','content':'same'}],[{'role':'user','content':'same','has_answer':True}]],'answer_session_ids':['s']}]
    case=convert_native(data,'longmemeval')[0]
    assert len({e['id'] for e in case['events']})==2
    assert [e['date'] for e in case['events']]==['first','second']
    assert case['gold_ids']==[case['events'][1]['id']]

def test_resume_preserves_failed_rows_and_rejects_changed_inputs(tmp_path):
    root=tmp_path/'run';frozen={'split':{'q':'evaluation'}}
    manifest={'case_ids':['q'],'profiles':['bm25'],'model':'pinned','hardware':{'available':1}}
    manifest['manifest_sha256']=digest(manifest)
    open_run(root,manifest,frozen)
    row={'case_id':'q','profile':'bm25','failure':'timeout'}
    (root/'rows.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    changed={**manifest,'hardware':{'available':2}}
    saved,rows=open_run(root,changed,frozen,resume=True)
    assert rows==[row] and saved==manifest
    with pytest.raises(ValueError):open_run(root,{**changed,'model':'different'},frozen,resume=True)
    (root/'rows.jsonl').write_text((json.dumps(row)+'\n')*2,encoding='utf-8')
    with pytest.raises(ValueError):open_run(root,manifest,frozen,resume=True)

def test_full_selection_uses_only_the_frozen_partition():
    cases=[{'id':'b'},{'id':'a'}];frozen={'split':{'a':'development','b':'evaluation'}}
    assert select_cases(cases,frozen,'evaluation',2)==[cases[0]]
    assert select_cases(cases,frozen,'development',2)==[cases[1]]

def test_native_conversion_preserves_roles_dates_speakers_and_gold(tmp_path):
    locomo=[{'sample_id':'a','conversation':{'speaker_a':'Alice','speaker_b':'Bob','session_1_date_time':'yesterday','session_1':[{'speaker':'Alice','dia_id':'D1:1','text':'port 8080','blip_caption':'image evidence'}]},'qa':[{'question':'port?','answer':8080,'category':4,'evidence':['D1:1']}]}]
    path=tmp_path/'data.json';path.write_text(json.dumps(locomo))
    c=load_native(path,'locomo')[0]
    assert c['gold_ids']==['D1:1'] and c['events'][0]['date']=='yesterday'
    assert c['events'][0]['speaker']=='Alice' and c['events'][0]['caption']=='image evidence'
    longmem=[{'question_id':'q1','question_type':'single-session-assistant','question':'port?','answer':8080,'question_date':'today','haystack_dates':['yesterday'],'haystack_session_ids':['s'],'haystack_sessions':[[{'role':'assistant','content':'port 8080','has_answer':True}]],'answer_session_ids':['s']}]
    path.write_text(json.dumps(longmem));c=load_native(path,'longmemeval')[0]
    assert c['events'][0]['role']=='assistant' and c['gold_ids']==['s:0']
    assert c['question_date']=='today'

def test_freeze_clusters_shared_content_and_preserves_holdout():
    cases=[]
    for i,content in enumerate(('same','same','different','fourth','fifth')):
        cases.append({'id':str(i),'group':str(i),'dataset':'longmemeval','events':[{'id':str(i),'session':str(i),'date':'date','role':'user','speaker':'user','content':content}]})
    frozen=freeze(cases)
    assert frozen['clusters']['0']==frozen['clusters']['1']
    assert frozen['split']['0']==frozen['split']['1']
    assert frozen==freeze(cases) and frozen['tuning_trials']==0
    assert any(v=='evaluation' for v in frozen['split'].values())
    assert all(frozen['split'][c['id']]=='development' for c in smoke_cases(cases,frozen))

def test_real_generation_never_falls_back_and_checks_identity(monkeypatch):
    def request(self,path,payload=None):
        if path=='/api/tags':return {'models':[{'name':'m','digest':'d'}]}
        if path in ('/api/show','/api/version'):return {}
        return {'model':'m','done':False,'message':{'content':'fake incomplete'}}
    monkeypatch.setattr(Ollama,'request',request)
    with pytest.raises(RuntimeError):Ollama('m','wrong')
    model=Ollama('m','d')
    with pytest.raises(RuntimeError):model.generate('question')
    with pytest.raises(ValueError):model.generate('x'*3000)

def test_upstream_scoring_source_pinned_before_execution(tmp_path):
    path=tmp_path/'score.py';path.write_text('def scorer(): return 1')
    with pytest.raises(ValueError):functions(path,['scorer'],'wrong')

def test_judge_uses_grading_instruction_without_answering_constraint(monkeypatch):
    requests=[]
    def request(self,path,payload=None):
        if path=='/api/tags':return {'models':[{'name':'m','digest':'d'}]}
        if path in ('/api/show','/api/version'):return {}
        requests.append(payload)
        return {'model':'m','done':True,'message':{'content':'yes'}}
    monkeypatch.setattr(Ollama,'request',request)
    client=Ollama('m','d')
    client.generate('answer this')
    client.generate('grade this',output=8,system=JUDGE_SYSTEM)
    assert requests[0]['messages'][0]['content']==SYSTEM
    assert requests[1]['messages'][0]['content']==JUDGE_SYSTEM
    assert 'only the reference conversation' not in JUDGE_SYSTEM

def test_native_id_mapping_and_budgets(tmp_path):
    rt=Runtime(str(tmp_path))
    rt.append_event(id='D1:1',namespace='development',content='gateway port 8080')
    assert lexical(rt,'gateway port')==['D1:1']
    events=[{'id':'D1:1','session':'s','date':'today','speaker':'A','role':'user','content':'x'*1000},{'id':'D1:2','session':'s','date':'today','speaker':'A','role':'user','content':'port 8080'}]
    context,ids,omitted=packed(events,100)
    assert ids==['D1:2'] and omitted==['D1:1'] and len(context.encode())<=100

def rows():
    return [{'case_id':str(i),'cluster':str(i//2),'dataset':'locomo','category':'2','profile':p,'score':float(p=='nsn'),'failure':None if p=='nsn' else 'generation error','complete_evidence':False,'gold_count':1,'generation_ms':0,'prompt_eval_count':0,'client_rss_bytes':10,'retrieval_ms':1,'scorer':'official'} for i in range(4) for p in ('bm25','nsn')]

def test_report_failure_denominators_cluster_pairing_and_rerun():
    data=rows();manifest={'case_ids':['0','1','2','3'],'profiles':['bm25','nsn'],'manifest_sha256':'pinned'}
    summary=summarize(data,manifest)
    assert summary['profiles']['bm25']['failures']==4
    assert summary['profiles']['bm25']['answer_metrics']['locomo']['score']==0
    assert summary==summarize(data,manifest)
    assert paired_interval(data,'bm25','nsn')['ci95']==[1,1]
    assert paired_interval(data,'bm25','nsn')['clusters']==2
    with pytest.raises(ValueError):summarize(data[:-1],manifest)


def test_missing_judge_never_inflates_accuracy_denominator():
    data=rows();data[0]['score']=None;data[0]['judge_failure']='judge unavailable'
    manifest={'case_ids':['0','1','2','3'],'profiles':['bm25','nsn'],'manifest_sha256':'pinned'}
    result=summarize(data,manifest)['profiles']['bm25']['answer_metrics']['locomo']
    assert result['score'] is None and result['total']==4
    assert result['score_lower_bound']==0 and result['score_upper_bound']==.25
    interval=paired_interval(data,'bm25','nsn')
    assert interval['ci95'] is None and interval['unscored_pairs']==1


def test_single_conversation_cannot_produce_generalization_interval():
    data=rows()
    for row in data:row['cluster']='shared'
    result=paired_interval(data,'bm25','nsn')
    assert result['ci95'] is None and result['clusters']==1


def test_historical_self_judge_scores_are_excluded_without_mutating_raw_rows():
    data=rows()
    for row in data:row.update(dataset='longmemeval',failure=None)
    manifest={'case_ids':['0','1','2','3'],'profiles':['bm25','nsn'],'manifest_sha256':'pinned','judge':'local answering-model self-judge'}
    result=summarize(data,manifest)
    assert result['excluded_self_judge_scores']==8
    assert result['profiles']['nsn']['answer_metrics']['longmemeval']['score'] is None
    assert result['comparisons']['longmemeval:nsn-bm25']['ci95'] is None
    assert all(row['score'] is not None for row in data)


def test_unknown_profiles_cannot_be_hidden_by_summary():
    data=rows();manifest={'case_ids':['0','1','2','3'],'profiles':['bm25','nsn'],'manifest_sha256':'pinned'}
    with pytest.raises(ValueError):summarize(data+[{**data[0],'profile':'unknown'}],manifest)


def test_cached_build_costs_are_not_multiplied_by_profile_rows():
    data=rows()
    for row in data:
        row['corpus_cache']={'key':'shared','ingestion_build_ms':10,'embedding_build_ms':20,
                             'hit':row['case_id']!='0','restore_ms':0 if row['case_id']=='0' else 2}
    manifest={'case_ids':['0','1','2','3'],'profiles':['bm25','nsn'],'manifest_sha256':'pinned'}
    costs=summarize(data,manifest)['corpus_cache_costs']
    assert costs['distinct_native_builds']==1 and costs['observed_builds']==1
    assert costs['observed_build_ms']==30 and costs['original_embedding_build_ms']==20
    assert costs['private_copy_restore_ms']==6


def test_declared_primary_uses_its_paired_baseline_and_keeps_unknown_grades_visible():
    data=rows()
    manifest={'case_ids':['0','1','2','3'],'profiles':['bm25','nsn'],'manifest_sha256':'pinned',
              'split':'evaluation','model':{'name':'answer-model'},
              'analysis_plan':{'sha256':'policy-pin','policy':{'primary':{
                  'dataset':'locomo','model':'answer-model','comparison':['nsn','bm25'],
                  'target_absolute_gain':.05}}}}
    result=summarize(data,manifest)
    assert result['declared_primary']['interval']['delta']==-1
    assert not result['publishable_superiority']
    data[0]['score']=None
    assert summarize(data,manifest)['declared_primary']['interval']['ci95'] is None


def test_local_provider_output_overrun_is_a_failure_not_a_usable_answer():
    client=object.__new__(Ollama)
    client.model='pinned-model'; client.context=2048; client.output=32
    client.request=lambda *a,**k: {'model':'pinned-model','done':True,
                                   'message':{'content':'answer'},'eval_count':33}
    with pytest.raises(RuntimeError,match='output token budget'):
        client.generate('question')
