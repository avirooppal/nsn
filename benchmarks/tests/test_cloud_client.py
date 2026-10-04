import io
import urllib.error
import pytest
from benchmarks.public_eval.cloud import CloudOllama, NoRedirect


def client(monkeypatch, tag='revision'):
    monkeypatch.setattr(CloudOllama, 'request', lambda self, path, payload=None:
                        {'models': [{'name': 'model', 'digest': tag}]})
    return CloudOllama('model', 'revision', api_key='test-credential')


def test_cloud_identity_never_substitutes(monkeypatch):
    with pytest.raises(RuntimeError, match='revision mismatch'):
        client(monkeypatch, 'changed')


def test_cloud_budgets_and_response_validation(monkeypatch):
    c = client(monkeypatch)
    calls = []
    def request(path, payload=None):
        calls.append(payload)
        return {'model': 'model', 'done': True, 'message': {'content': 'answer'}, 'eval_count': 2}
    c.request = request
    assert c.generate('question')[0] == 'answer'
    assert calls[0]['options']['num_predict'] == 32
    with pytest.raises(ValueError, match='allowance'):
        c.generate('x' * 3000)
    assert len(calls) == 1
    c.request = lambda *a: {'model': 'model', 'done': True, 'message': {'content': 'answer'}, 'eval_count': 33}
    with pytest.raises(RuntimeError, match='token budget'):
        c.generate('question')


def test_cloud_error_redaction_and_redirect_rejection(monkeypatch):
    c = client(monkeypatch)
    # Undo the constructor's test transport, then exercise the real error path.
    monkeypatch.undo()
    def denied(*a, **k):
        raise urllib.error.HTTPError('https://ollama.com/api/chat', 401,
                                     'Bearer test-credential', {}, io.BytesIO(b'test-credential'))
    c._opener.open = denied
    with pytest.raises(RuntimeError) as error:
        c.request('/api/chat', {})
    assert str(error.value) == 'Ollama Cloud HTTP 401'
    assert NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.invalid') is None
