"""Fetch explicitly pinned public assets; no keys, model downloads or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def prepare(root, pins):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(pins).read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        if Path(name).name!=name: raise ValueError('Invalid asset filename')
        dst=root/name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==info['sha256']: continue
        with urllib.request.urlopen(info['url'],timeout=90) as response:
            raw=response.read(128*1024*1024+1)
        if len(raw)>128*1024*1024 or hashlib.sha256(raw).hexdigest()!=info['sha256']:
            raise ValueError('Pinned source changed/oversized: '+name)
        dst.write_bytes(raw)
    (root/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--assets',default='benchmarks/assets/phase09');parser.add_argument('--pins',default=str(Path(__file__).with_name('pins.json')))
    args=parser.parse_args();prepare(args.assets,args.pins)
