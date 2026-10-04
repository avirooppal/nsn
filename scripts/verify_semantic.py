"""Installed-wheel semantic/offline/restart check using explicit prepared assets."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import tempfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assets=Path(args.assets).resolve()
    def denied(*a,**k):raise AssertionError('Unexpected outbound request')
    socket.create_connection=denied;socket.socket.connect=denied
    import nsn
    from neurosleepnet.embeddings.pooled import LocalPooledEncoder
    assert 'site-packages' in Path(nsn.__file__).resolve().parts
    with tempfile.TemporaryDirectory() as directory:
        encoder=LocalPooledEncoder(local_asset_path=str(assets),offline=True,revision=assets.name)
        with nsn.Runtime(directory,encoder=encoder) as rt:
            eid=rt.append_event(namespace='check',content='Gateway production port is 8080.')
            assert rt.vector_store.count('check','event')==1
        with nsn.Runtime(directory,encoder=encoder) as rt:
            pack=rt.retrieve_pack('Gateway production port',namespace='check')
            assert eid in {item.id for item in pack.items}
            assert not rt.retrieve_pack('Gateway production port',namespace='other').items
    files={str(p.relative_to(assets)):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.rglob('*') if p.is_file()}
    result={'passed':True,'installed_module':nsn.__file__,'network_denied':True,'restart_recall':True,'namespace_isolation':True,'encoder_revision':assets.name,'asset_sha256':files,'packages':{name:importlib.metadata.version(name) for name in ('nsn','torch','sentence-transformers','transformers','numpy')}}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
