"""Fetch explicitly pinned public assets; no keys, model downloads or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def prepare(root, pins, include_large=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(pins).read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        if info.get('large') and not include_large: continue
        if Path(name).name!=name: raise ValueError('Invalid asset filename')
        dst=root/name
        def checksum(path):
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b''): h.update(chunk)
            return h.hexdigest()
        if dst.exists() and checksum(dst)==info['sha256']: continue
        limit=info.get('size',128*1024*1024)
        temporary=dst.with_suffix(dst.suffix+'.part')
        size=temporary.stat().st_size if temporary.exists() else 0
        if size>=limit: size=0
        request=urllib.request.Request(info['url'],headers={'Range':f'bytes={size}-'} if size else {})
        h=hashlib.sha256()
        with urllib.request.urlopen(request,timeout=90) as response:
            if size and response.status==206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {size}-'):
                    raise ValueError('Invalid range response: '+name)
                with temporary.open('rb') as existing:
                    for chunk in iter(lambda:existing.read(8*1024*1024),b''):h.update(chunk)
            else: size=0
            with temporary.open('ab' if size else 'wb') as handle:
                for chunk in iter(lambda:response.read(8*1024*1024),b''):
                    size+=len(chunk)
                    if size>limit: raise ValueError('Oversized asset: '+name)
                    h.update(chunk);handle.write(chunk)
        if h.hexdigest()!=info['sha256'] or ('size' in info and size!=info['size']):
            raise ValueError('Pinned source changed/oversized: '+name)
        temporary.replace(dst)
        print('Verified',name,size,flush=True)
    prepared={**manifest,'files':{name:info for name,info in manifest['files'].items() if (root/name).is_file()}}
    (root/'sources.json').write_text(json.dumps(prepared,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--assets',default='benchmarks/assets/phase09');parser.add_argument('--pins',default=str(Path(__file__).with_name('pins.json')))
    parser.add_argument('--include-large',action='store_true')
    args=parser.parse_args();prepare(args.assets,args.pins,args.include_large)
