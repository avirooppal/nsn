from types import SimpleNamespace
import json
import pytest
from benchmarks.public_eval.grading import independent_identity,review_queue,validate_reviews,load_run


def test_independent_judge_rejects_same_checkpoint_even_with_alias():
    manifest={'model':{'digest':'same'}}
    with pytest.raises(ValueError):independent_identity(manifest,SimpleNamespace(identity={'name':'alias','digest':'same'}))
    independent_identity(manifest,SimpleNamespace(identity={'digest':'different'}))


def test_review_requires_real_labels_and_rejects_stale_answers():
    manifest={'manifest_sha256':'pinned'}
    rows=[{'case_id':'q','profile':'nsn','dataset':'locomo','category':'1','question':'port?','gold_answer':'8080','answer':'9090','failure':None}]
    queue=review_queue(manifest,rows)
    assert queue==review_queue(manifest,rows)
    with pytest.raises(ValueError):validate_reviews(manifest,rows,queue)
    queue['reviews'][0].update(label='incorrect',rationale='Wrong port.')
    assert validate_reviews(manifest,rows,queue)['reviewed']==1
    rows[0]['answer']='8080'
    with pytest.raises(ValueError):validate_reviews(manifest,rows,queue)


def test_judge_rejects_tampered_answer_manifest_before_scoring(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'model':{'digest':'altered'},'manifest_sha256':'original'}),encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):load_run(tmp_path)


def test_human_review_cannot_drop_failed_rows_or_edit_presented_answers():
    manifest={'manifest_sha256':'pinned'}
    rows=[{'case_id':str(i),'profile':'nsn','dataset':'locomo','category':'1','question':'port?','gold_answer':'8080','answer':'9090','failure':'timeout' if i else None} for i in range(2)]
    queue=review_queue(manifest,rows)
    for review in queue['reviews']:review.update(label='incorrect',rationale='Incorrect or failed.')
    assert validate_reviews(manifest,rows,queue)['reviewed']==2
    queue['reviews'].pop()
    with pytest.raises(ValueError,match='Incomplete'):validate_reviews(manifest,rows,queue)
    queue=review_queue(manifest,rows)
    for review in queue['reviews']:review.update(label='incorrect',rationale='Incorrect or failed.')
    queue['reviews'][0]['predicted_answer']='8080'
    with pytest.raises(ValueError,match='content'):validate_reviews(manifest,rows,queue)
def test_same_checkpoint_cloud_prefix_and_name_alias_are_not_independent():
    from types import SimpleNamespace
    from benchmarks.public_eval.grading import independent_identity
    answer = {'model': {'name': 'model:20b', 'digest': 'abcdef123456' + '0' * 52}}
    with pytest.raises(ValueError, match='answering checkpoint'):
        independent_identity(answer, SimpleNamespace(identity={'name': 'alias', 'digest': 'abcdef123456'}))
    with pytest.raises(ValueError, match='answering checkpoint'):
        independent_identity(answer, SimpleNamespace(identity={'name': 'model:20b-cloud', 'digest': 'other'}))


def test_human_agreement_keeps_uncertain_and_failed_judges_visible():
    manifest={'manifest_sha256':'pinned','independent_judge':{'identity':{'digest':'separate'}}}
    rows=[{'case_id':str(i),'profile':'nsn','dataset':'longmemeval','category':'single',
           'question':'port?','gold_answer':'8080','answer':'8080','failure':None,
           'judge_called':True,'judge_failure':None,'score':score}
          for i,score in enumerate([1,0,1,1,None,0])]
    rows[4]['judge_failure']='timeout'
    rows[5].update(judge_called=False,failure='answer timeout')
    queue=review_queue(manifest,rows,per_category=6)
    labels={'0':'correct','1':'incorrect','2':'incorrect','3':'uncertain','4':'correct','5':'incorrect'}
    for review in queue['reviews']:
        review.update(label=labels[review['case_id']],rationale='Test fixture review.')
    result=validate_reviews(manifest,rows,queue)
    agreement=result['judge_human_agreement']
    assert result['reviewed']==6 and result['uncertain']==1
    assert agreement['binary_pairs']==3
    assert agreement['agreement_rate']==pytest.approx(2/3)
    assert agreement['cohen_kappa']==pytest.approx(.4)
    assert agreement['judge_missing_or_failed']==1
    assert agreement['not_independently_judged']==1
    assert agreement['disagreements']==[{'case_id':'2','profile':'nsn'}]


def test_no_binary_judgements_does_not_imply_perfect_agreement():
    manifest={'manifest_sha256':'pinned'}
    rows=[{'case_id':'q','profile':'nsn','dataset':'locomo','category':'1',
           'question':'port?','gold_answer':'8080','answer':'8080','failure':None,'score':1}]
    queue=review_queue(manifest,rows)
    queue['reviews'][0].update(label='correct',rationale='Test fixture review.')
    agreement=validate_reviews(manifest,rows,queue)['judge_human_agreement']
    assert agreement['binary_pairs']==0
    assert agreement['agreement_rate'] is None and agreement['cohen_kappa'] is None


def test_human_agreement_does_not_validate_historical_self_judging():
    manifest={'manifest_sha256':'pinned','judge':'diagnostic self-judge'}
    rows=[{'case_id':'q','profile':'nsn','dataset':'longmemeval','category':'1',
           'question':'port?','gold_answer':'8080','answer':'8080','failure':None,
           'score':1,'judge_called':True,'judge_failure':None}]
    queue=review_queue(manifest,rows)
    queue['reviews'][0].update(label='correct',rationale='Test fixture review.')
    result=validate_reviews(manifest,rows,queue)['judge_human_agreement']
    assert result['binary_pairs']==0 and result['agreement_rate'] is None

