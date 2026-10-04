"""Exercise locally available Windows Python versions using installed wheels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--wheel');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];wheel=Path(args.wheel).resolve() if args.wheel else repo/'dist/nsn-0.3.0-py3-none-any.whl';results={}
    suites=['test_durable_storage','test_facts_history','test_retrieval_context','test_sdk_facade','test_audit_repairs','test_bounded_relationships','test_consolidation','test_advanced_memory','test_runtime_wal_lifecycle','test_event_retrieval_status','test_retrieval_cache_consistency','test_deferred_indexing','test_onnx_encoder','test_memory_hardening','test_memory_layer_contract']
    for version in ('3.9','3.10','3.12'):
        with tempfile.TemporaryDirectory(prefix='nsn_matrix_') as directory:
            env=Path(directory)/'venv';python=env/'Scripts/python.exe'
            commands=[['uv','venv','--python',version,str(env)],['uv','pip','install','--python',str(python),str(wheel),'pytest','pytest-asyncio','psutil'],[str(python),str(repo/'scripts/verify_install.py')],[str(python),'-m','pytest',*[str(repo/'tests'/(name+'.py')) for name in suites],'-k','not rest_authentication and not legacy_deprecation','-q','-p','no:cacheprovider']]
            attempts=[]
            for command in commands:
                completed=subprocess.run(command,cwd=directory,text=True,encoding='utf-8',errors='backslashreplace',capture_output=True)
                attempts.append({'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr})
                if completed.returncode:break
            results[version]={'passed':all(a['exit_code']==0 for a in attempts),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'attempts':attempts}
        Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(version,results[version]['passed'],flush=True)

if __name__=='__main__':main()
