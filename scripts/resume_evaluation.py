"""Continue a frozen run using its verified evaluation source snapshot.

The runner still verifies datasets, model, encoder and SDK source identity. A
changed SDK requires a new run, not permission to combine different experiments.
"""
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
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--assets',default='benchmarks/assets/phase09');parser.add_argument('--max-new-cases',type=int)
    parser.add_argument('--new-run');parser.add_argument('--model');parser.add_argument('--digest')
    parser.add_argument('--dataset',choices=('both','locomo','longmemeval'))
    parser.add_argument('--longmemeval-variant',choices=('oracle','s','m'))
    parser.add_argument('--provider',choices=('local','cloud'))
    args=parser.parse_args();root=Path(args.run).resolve()
    if not args.new_run and any((args.model,args.digest,args.dataset,args.longmemeval_variant,args.provider)):
        parser.error('Experiment overrides require --new-run; existing rows cannot be changed')
    if bool(args.model)!=bool(args.digest):parser.error('New model and digest must be supplied together')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})!=manifest['manifest_sha256']:raise ValueError('Manifest checksum changed')
    for name,checksum in manifest['script_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Evaluation source snapshot changed')
    with tempfile.TemporaryDirectory(prefix='nsn_frozen_eval_') as directory:
        package=Path(directory)/'benchmarks/public_eval';package.mkdir(parents=True)
        (package.parent/'__init__.py').write_text('',encoding='utf-8')
        for name in manifest['script_sha256']:shutil.copy2(root/'source_snapshot'/name,package/name)
        command=[sys.executable,'-m','benchmarks.public_eval.runner','--assets',str(Path(args.assets).resolve()),'--output',str(Path(args.new_run).resolve()) if args.new_run else str(root),'--model',args.model or manifest['model']['name'],'--digest',args.digest or manifest['model']['digest'],'--selection',manifest['selection']]
        if not args.new_run:command+=['--resume']
        if manifest.get('embedding'):command+=['--embedding-path',manifest['embedding']['path']]
        if args.longmemeval_variant or manifest.get('longmemeval_variant'):command+=['--longmemeval-variant',args.longmemeval_variant or manifest['longmemeval_variant']]
        if args.dataset or manifest.get('dataset_selection'):command+=['--dataset',args.dataset or manifest['dataset_selection']]
        if args.provider or manifest.get('provider'):command+=['--provider',args.provider or manifest['provider'],'--context',str(manifest['context_tokens']),'--output-tokens',str(manifest['reserved_output_tokens'])]
        if manifest.get('encoder_backend'):command+=['--encoder-backend',manifest['encoder_backend']]
        if manifest.get('corpus_cache_path'):command+=['--corpus-cache',manifest['corpus_cache_path']]
        if manifest.get('analysis_plan'):command+=['--analysis-plan',manifest['analysis_plan']['path']]
        if 'self-judge' in manifest.get('judge',''):command+=['--local-judge']
        if args.max_new_cases is not None:command+=['--max-new-cases',str(args.max_new_cases)]
        result=subprocess.run(command,cwd=directory,env={**os.environ,'PYTHONPATH':directory})
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
