"""Optional offline MiniLM inference without the PyTorch/Transformers runtime."""
import functools
import hashlib
import json
from pathlib import Path

from .pooled import AssetNotFoundError, VectorEncoder


class ONNXPooledEncoder(VectorEncoder):
    """Pinned BERT token features, attention-masked mean pooling, L2 normalization.

    Assets are prepared explicitly. This provider never downloads or exports a
    model and does not change the default sentence-transformers encoder.
    """
    def __init__(self, local_asset_path, cache_size=2048):
        self.assets = Path(local_asset_path)
        required = ('model.onnx', 'tokenizer.json', 'config.json',
                    'sentence_bert_config.json', '1_Pooling/config.json')
        if not all((self.assets / name).is_file() for name in required):
            raise AssetNotFoundError('Prepared ONNX MiniLM assets are incomplete')
        config = json.loads((self.assets / 'config.json').read_text(encoding='utf-8'))
        pooling = json.loads((self.assets / '1_Pooling/config.json').read_text(encoding='utf-8'))
        if config.get('model_type') != 'bert' or config.get('hidden_size') != 384 or not pooling.get('pooling_mode_mean_tokens') or any(pooling.get(k) for k in ('pooling_mode_cls_token', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens', 'pooling_mode_weightedmean_tokens', 'pooling_mode_lasttoken')):
            raise ValueError('Only 384-dimensional BERT mean-pooling assets are supported')
        self.max_length = json.loads((self.assets / 'sentence_bert_config.json').read_text(encoding='utf-8'))['max_seq_length']
        identity = hashlib.sha256()
        for name in required:
            identity.update(name.encode('utf-8'))
            with (self.assets / name).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    identity.update(chunk)
        self._revision = identity.hexdigest()
        self._session = self._tokenizer = None
        self._cached = functools.lru_cache(maxsize=cache_size)(lambda text: tuple(self.encode_batch([text])[0]))

    @property
    def model_name(self): return 'all-MiniLM-L6-v2-onnx'

    @property
    def revision(self): return self._revision

    @property
    def dimension(self): return 384

    @property
    def fingerprint(self): return 'onnx_minilm_mean_' + self.revision

    def _load(self):
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Install 'nsn[semantic-onnx]' for ONNX inference") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(self.assets / 'model.onnx'),
                                      sess_options=options, providers=['CPUExecutionProvider'])
        tokenizer = Tokenizer.from_file(str(self.assets / 'tokenizer.json'))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        self._tokenizer, self._session = tokenizer, session

    def encode_batch(self, texts):
        if not texts:
            return []
        self._load()
        # Full public haystacks contain thousands of turns. Bound token-feature
        # allocations independently of caller batch size, preserving input order.
        result = []
        for offset in range(0, len(texts), 64):
            result.extend(self._encode_chunk(texts[offset:offset + 64]))
        return result

    def _encode_chunk(self, texts):
        import numpy as np
        tokens = self._tokenizer.encode_batch(texts)
        inputs = {'input_ids': np.asarray([t.ids for t in tokens], dtype=np.int64),
                  'attention_mask': np.asarray([t.attention_mask for t in tokens], dtype=np.int64),
                  'token_type_ids': np.asarray([t.type_ids for t in tokens], dtype=np.int64)}
        inputs = {i.name: inputs[i.name] for i in self._session.get_inputs()}
        features = self._session.run(None, inputs)[0]
        if features.ndim != 3 or features.shape[-1] != self.dimension:
            raise ValueError('Unexpected ONNX token-feature shape')
        mask = np.asarray([t.attention_mask for t in tokens], dtype=np.float32)[..., None]
        pooled = (features * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.tolist()

    def encode(self, text):
        return list(self._cached(text))
