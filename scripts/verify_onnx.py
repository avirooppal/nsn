"""Offline installed ONNX wheel smoke, with no answering model or PyTorch."""
import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile


def verify(assets):
    def denied(*args, **kwargs):
        raise AssertionError('Unexpected network request')
    socket.create_connection = denied
    socket.socket.connect = denied
    import nsn
    from neurosleepnet.embeddings.onnx_pooled import ONNXPooledEncoder
    assert 'site-packages' in Path(nsn.__file__).parts
    encoder = ONNXPooledEncoder(assets)
    with tempfile.TemporaryDirectory() as directory:
        with nsn.Runtime(directory, encoder=encoder, defer_indexing=True) as rt:
            eid = rt.append_event(namespace='agent', content='Production gateway port 8080.')
            assert rt.vector_store.count('agent') == 0
        with nsn.Runtime(directory, encoder=encoder, defer_indexing=True) as rt:
            assert rt.process_index_jobs(worker_id='restart') == 1
            assert eid in {item.id for item in rt.retrieve_pack('gateway production port', namespace='agent').items}
            assert not rt.retrieve_pack('gateway production port', namespace='other').items
    assert not any(name in sys.modules for name in ('torch', 'transformers', 'sentence_transformers'))
    return {'passed': True, 'installed_module': nsn.__file__, 'network_denied': True,
            'deferred_restart_recall': True, 'namespace_isolation': True,
            'encoder_fingerprint': encoder.fingerprint, 'heavy_imports': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.assets)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
