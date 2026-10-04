"""Resume independent grading with its verified source snapshot and raw binding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--graded-run',required=True)
    parser.add_argument('--answer-run',required=True)
    parser.add_argument('--assets',default='benchmarks/assets/phase09')
    parser.add_argument('--max-new-rows',type=int)
    args=parser.parse_args()
    root=Path(args.graded_run).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    canonical={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    if hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=manifest['manifest_sha256']:
        raise ValueError('Grading manifest checksum changed')
    judge=manifest['independent_judge']
    with tempfile.TemporaryDirectory(prefix='nsn_frozen_grade_') as directory:
        package=Path(directory)/'benchmarks/public_eval';package.mkdir(parents=True)
        (package.parent/'__init__.py').write_text('')
        for name,sha in judge['helper_sha256'].items():
            source=root/'grading_source_snapshot'/name
            if Path(name).name!=name or hashlib.sha256(source.read_bytes()).hexdigest()!=sha:
                raise ValueError('Grader source snapshot changed')
            shutil.copy2(source,package/name)
        command=[sys.executable,'-m','benchmarks.public_eval.grading',str(Path(args.answer_run).resolve()),
                 '--output',str(root),'--assets',str(Path(args.assets).resolve()),'--judge-model',judge['identity']['name'],
                 '--judge-digest',judge['identity']['digest'],'--judge-provider',judge['provider'],
                 '--judge-output-tokens',str(judge['output_tokens']),'--resume']
        if args.max_new_rows is not None:command+=['--max-new-rows',str(args.max_new_rows)]
        result=subprocess.run(command,cwd=directory,env={**os.environ,'PYTHONPATH':directory})
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
