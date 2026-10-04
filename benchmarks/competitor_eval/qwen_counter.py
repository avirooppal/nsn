"""Verified tokenizer and exact restricted Qwen3 Ollama chat template.

Only the verified two-message, no-tools, think=False shape is supported. This is
not a generic chat tokenizer. Native requests are never shortened to fit.
"""
import hashlib
import json
from pathlib import Path


class QwenChatCounter:
    def __init__(self, assets, client):
        root = Path(assets)
        self.identity = json.loads((root / 'identity.json').read_text(encoding='utf-8'))
        if self.identity['model_digest'] != client.identity['digest']:
            raise ValueError('Counter model digest mismatch')
        if hashlib.sha256(client.show['template'].encode()).hexdigest() != self.identity['template_sha256']:
            raise ValueError('Counter chat template mismatch')
        tokenizer = root / 'tokenizer.json'
        if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != self.identity['tokenizer_sha256']:
            raise ValueError('Counter tokenizer checksum mismatch')
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer))

    @staticmethod
    def render(messages):
        if len(messages) != 2 or [m.get('role') for m in messages] != ['system', 'user'] or any(not isinstance(m.get('content'), str) or m.get('tool_calls') or m.get('thinking') or m.get('images') for m in messages):
            raise ValueError('Exact counter supports only plain system/user, no tools/images/thinking')
        system, user = (m['content'] for m in messages)
        prefix = '<|im_start|>system\n' + system + '<|im_end|>\n' if system else ''
        return prefix + '<|im_start|>user\n' + user + ' /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def count(self, messages):
        return len(self.tokenizer.encode(self.render(messages), add_special_tokens=False).ids)
