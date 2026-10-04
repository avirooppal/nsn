"""Run the declared Linux core matrix against the same built wheel, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--wheel');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];wheel=Path(args.wheel).resolve() if args.wheel else repo/'dist/nsn-0.3.0-py3-none-any.whl';results={}
    suites=['test_durable_storage','test_facts_history','test_retrieval_context','test_sdk_facade','test_audit_repairs','test_bounded_relationships','test_consolidation','test_advanced_memory','test_runtime_wal_lifecycle','test_event_retrieval_status','test_retrieval_cache_consistency','test_deferred_indexing','test_onnx_encoder','test_memory_hardening','test_memory_layer_contract']
    with tempfile.TemporaryDirectory(prefix='nsn_linux_matrix_') as directory:
        root=Path(directory);shutil.copy2(wheel,root/wheel.name);shutil.copy2(repo/'scripts/verify_install.py',root/'verify_install.py')
        tests=root/'tests';tests.mkdir()
        for name in suites:shutil.copy2(repo/'tests'/(name+'.py'),tests/(name+'.py'))
        for version in ('3.9','3.10','3.12'):
            (root/'Dockerfile').write_text(f'FROM python:{version}-slim\nCOPY {wheel.name} /tmp/\nRUN pip install --no-cache-dir /tmp/{wheel.name} pytest pytest-asyncio psutil\nCOPY verify_install.py /checks/\nCOPY tests/ /checks/tests/\nWORKDIR /tmp\n',encoding='utf-8')
            tag='nsn-matrix:'+version
            commands=[['docker','build','-t',tag,str(root)],['docker','run','--rm','--network','none',tag,'python','/checks/verify_install.py'],['docker','run','--rm','--network','none',tag,'python','-m','pytest','/checks/tests','-k','not rest_authentication and not legacy_deprecation','-q','-p','no:cacheprovider']]
            attempts=[]
            for command in commands:
                process=subprocess.run(command,cwd=directory,text=True,encoding='utf-8',errors='backslashreplace',capture_output=True)
                attempts.append({'command':command,'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr})
                if process.returncode:break
            results[version]={'passed':all(a['exit_code']==0 for a in attempts),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'attempts':attempts}
            Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8');print(version,results[version]['passed'],flush=True)


if __name__=='__main__':main()
