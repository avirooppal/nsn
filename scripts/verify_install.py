"""Run from outside the checkout with a freshly installed minimal wheel."""
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import threading


def denied(*args, **kwargs):
    raise AssertionError('Unexpected network access during minimal operation')


socket.socket.connect = denied
socket.create_connection = denied
before_threads = {t.ident for t in threading.enumerate()}
with tempfile.TemporaryDirectory() as directory:
    before = set(Path(directory).iterdir())
    import nsn
    assert {t.ident for t in threading.enumerate()} == before_threads
    assert set(Path(directory).iterdir()) == before
    for module in ('torch', 'numpy', 'faiss', 'spacy', 'sentence_transformers', 'fastapi', 'langchain_core'):
        assert importlib.util.find_spec(module) is None, module+' installed in minimal environment'
    model_file = Path(nsn.__file__).resolve()
    assert 'site-packages' in model_file.parts, model_file
    def fixture(prompt):
        return '8080' if '<retrieved_evidence>' in prompt and '8080' in prompt else 'acknowledged'
    nsn.init(directory)
    model = nsn.wrap(fixture, namespace='check')
    model('Gateway production port is 8080.')
    nsn.close()
    nsn.init(directory)
    model = nsn.wrap(fixture, namespace='check')
    assert model('What is the gateway production port?') == '8080'
    assert not nsn.get_default_runtime().retrieve_pack('gateway port', namespace='other').items
    from importlib.resources import files
    assert files('neurosleepnet.config').joinpath('defaults.yaml').is_file()
    nsn.close()
print(json.dumps({'minimal_wheel':str(model_file),'network_denied':True,'restart_recall':True,'namespace_isolation':True,'package_yaml':True,'import_threads_unchanged':True}))
