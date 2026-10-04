import hashlib
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.public_eval.datasets import digest


def test_resume_rejects_modified_snapshot_before_executing_it(tmp_path):
    snapshot=tmp_path/'source_snapshot';snapshot.mkdir()
    original=b'print("original")'
    manifest={'script_sha256':{'runner.py':hashlib.sha256(original).hexdigest()}}
    manifest['manifest_sha256']=digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (snapshot/'runner.py').write_text('raise AssertionError("must not execute")',encoding='utf-8')
    script=Path(__file__).resolve().parents[2]/'scripts/resume_evaluation.py'
    result=subprocess.run([sys.executable,str(script),'--run',str(tmp_path)],cwd=tmp_path,text=True,capture_output=True,timeout=10)
    assert result.returncode!=0 and 'source snapshot changed' in result.stderr
    assert 'must not execute' not in result.stderr


def test_resume_cannot_override_existing_model_or_dataset(tmp_path):
    script=Path(__file__).resolve().parents[2]/'scripts/resume_evaluation.py'
    for override in (['--model','other','--digest','different'],['--dataset','locomo']):
        result=subprocess.run([sys.executable,str(script),'--run',str(tmp_path),*override],
                              cwd=tmp_path,text=True,capture_output=True,timeout=10)
        assert result.returncode==2
        assert 'overrides require --new-run' in result.stderr


def test_new_campaign_requires_both_model_and_digest(tmp_path):
    script=Path(__file__).resolve().parents[2]/'scripts/resume_evaluation.py'
    result=subprocess.run([sys.executable,str(script),'--run',str(tmp_path),
                           '--new-run',str(tmp_path/'new'),'--model','other'],
                          cwd=tmp_path,text=True,capture_output=True,timeout=10)
    assert result.returncode==2 and 'supplied together' in result.stderr


def test_new_campaign_uses_verified_snapshot_and_separate_output(tmp_path):
    snapshot=tmp_path/'source_snapshot';snapshot.mkdir()
    original=b'import json,sys; print(json.dumps(sys.argv[1:]))'
    manifest={'script_sha256':{'runner.py':hashlib.sha256(original).hexdigest()},
              'model':{'name':'old','digest':'old-pin'},'selection':'evaluation',
              'provider':'local','context_tokens':2048,'reserved_output_tokens':32,
              'dataset_selection':'longmemeval','longmemeval_variant':'s'}
    manifest['manifest_sha256']=digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (snapshot/'runner.py').write_bytes(original)
    script=Path(__file__).resolve().parents[2]/'scripts/resume_evaluation.py'
    new=tmp_path/'new'
    result=subprocess.run([sys.executable,str(script),'--run',str(tmp_path),'--new-run',str(new),
                           '--model','qwen3:4b','--digest','new-pin','--dataset','locomo'],
                          cwd=tmp_path,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    command=json.loads(result.stdout)
    assert '--resume' not in command
    assert command[command.index('--output')+1]==str(new.resolve())
    assert command[command.index('--model')+1]=='qwen3:4b'
    assert command[command.index('--dataset')+1]=='locomo'
    assert not new.exists()  # Fixture runner only reports arguments.


def test_judge_snapshot_tampering_rejected_before_execution(tmp_path):
    snapshot=tmp_path/'grading_source_snapshot';snapshot.mkdir()
    original=b'print("original")'
    manifest={'independent_judge':{'helper_sha256':{'grading.py':hashlib.sha256(original).hexdigest()}}}
    manifest['manifest_sha256']=digest(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (snapshot/'grading.py').write_text('raise AssertionError("must not execute")',encoding='utf-8')
    script=Path(__file__).resolve().parents[2]/'scripts/resume_grading.py'
    result=subprocess.run([sys.executable,str(script),'--graded-run',str(tmp_path),
                           '--answer-run',str(tmp_path/'answers')],cwd=tmp_path,
                          text=True,capture_output=True,timeout=10)
    assert result.returncode!=0 and 'Grader source snapshot changed' in result.stderr
    assert 'must not execute' not in result.stderr
