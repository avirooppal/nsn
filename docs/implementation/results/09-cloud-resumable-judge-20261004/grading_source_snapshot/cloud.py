"""Explicit Ollama Cloud evaluation client; never imported by the NSN runtime."""
import ctypes
import ctypes.wintypes
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def evaluation_key():
    key = os.environ.get('OLLAMA_API_KEY')
    if key:
        return key
    if os.name != 'nt':
        raise RuntimeError('Set OLLAMA_API_KEY outside source control')
    path = Path(os.environ['LOCALAPPDATA']) / 'NSN/evaluation/ollama-key.dpapi'
    if not path.is_file():
        raise RuntimeError('No configured evaluation credential')
    class Blob(ctypes.Structure):
        _fields_ = [('size', ctypes.wintypes.DWORD),
                    ('data', ctypes.POINTER(ctypes.c_ubyte))]
    encrypted = path.read_bytes()
    buffer = ctypes.create_string_buffer(encrypted)
    source = Blob(len(encrypted), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    destination = Blob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(destination)
    ):
        raise RuntimeError('Cannot decrypt this user-bound evaluation credential')
    try:
        return ctypes.string_at(destination.data, destination.size).decode('utf-8')
    finally:
        ctypes.windll.kernel32.LocalFree(destination.data)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the bearer credential to a redirected origin.
        return None


class CloudOllama:
    def __init__(self, model, expected_digest, context=2048, output=32,
                 timeout=120, api_key=None):
        self.model = model
        self.host = 'https://ollama.com'
        self.context, self.output, self.timeout = context, output, timeout
        self._key = api_key or evaluation_key()
        self._opener = urllib.request.build_opener(NoRedirect())
        tags = self.request('/api/tags')
        matching = [m for m in tags['models'] if m['name'] == model]
        if len(matching) != 1 or matching[0].get('digest') != expected_digest:
            raise RuntimeError('Cloud model tag/revision mismatch; no substitution')
        self.identity = matching[0]
        self.show = {
            'provider': 'ollama_cloud',
            'identity_limit': 'provider tag revision; full weight checksum and quantization unavailable',
            'server_hardware': 'not disclosed; not local CPU qualification',
            'context_enforcement': 'client conservative allowance; remote option support not assumed',
        }
        self.version = {'provider': 'ollama_cloud', 'endpoint': self.host}

    def request(self, path, payload=None):
        if path not in ('/api/tags', '/api/chat'):
            raise ValueError('Unsupported cloud evaluation endpoint')
        req = urllib.request.Request(
            self.host + path,
            data=None if payload is None else json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + self._key},
        )
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Provider bodies can reflect request data. Never persist them as errors.
            raise RuntimeError('Ollama Cloud HTTP ' + str(exc.code)) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('Ollama Cloud connection failure/timeout') from None

    def generate(self, prompt, output=None, system=None):
        from benchmarks.public_eval.runner import SYSTEM
        system = SYSTEM if system is None else system
        limit = self.output if output is None else output
        if len((system + prompt).encode('utf-8')) + 256 + limit > self.context:
            raise ValueError('Context exceeds conservative byte-token allowance; no silent truncation')
        data = self.request('/api/chat', {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': prompt}],
            'think': False, 'stream': False,
            'options': {'seed': 42, 'temperature': 0,
                        'num_ctx': self.context, 'num_predict': limit},
        })
        content = data.get('message', {}).get('content')
        if data.get('model') != self.model or data.get('done') is not True or not isinstance(content, str) or not content.strip():
            raise RuntimeError('Invalid/empty/incomplete cloud model response')
        if data.get('eval_count', 0) > limit:
            raise RuntimeError('Cloud output exceeded the declared token budget')
        return content.strip(), data
