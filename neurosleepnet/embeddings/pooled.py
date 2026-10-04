"""
Compact Pooled Vector Embedding Providers.
Implements the VectorEncoder protocol, offline asset enforcement,
asset preparation, and lightweight deterministic encoders.
"""
from abc import ABC, abstractmethod
import functools
import hashlib
import math
import os
from typing import Any, List, Optional, Tuple, Union


class AssetNotFoundError(Exception):
    """Raised when required local embedding model assets are missing in offline mode."""
    pass


class VectorEncoder(ABC):
    """Protocol for compact vector encoders."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def revision(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class FastHashEncoder(VectorEncoder):
    """
    Deterministic n-gram hashing dense encoder for tests and minimal non-ML environments.

    WARNING: This is NOT a trained semantic model. It does not learn embeddings or capture
    semantic synonymy/paraphrase beyond exact token and subword n-gram collisions.
    For semantic retrieval, use LocalPooledEncoder or an external embedding model.
    """

    def __init__(self, dimension: int = 128, model_name: str = "fast-hash-v1", revision: str = "v1"):
        self._dim = dimension
        self._model_name = model_name
        self._revision = revision

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"hash_{self._model_name}_{self._revision}_{self._dim}"

    def encode(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim

        vec = [0.0] * self._dim
        words = text.lower().split()
        if not words:
            return [0.0] * self._dim

        # Bag of words and character n-grams hashing
        for w in words:
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            vec[idx] += sign

            # Character tri-grams for subword similarity
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tri = w[i:i+3]
                    h_tri = int(hashlib.md5(tri.encode('utf-8')).hexdigest(), 16)
                    idx_tri = h_tri % self._dim
                    sign_tri = 1.0 if ((h_tri >> 8) & 1) == 1 else -1.0
                    vec[idx_tri] += 0.5 * sign_tri

        # Normalize L2
        sq_sum = sum(x * x for x in vec)
        norm = math.sqrt(sq_sum)
        if norm > 1e-12:
            return [x / norm for x in vec]
        return vec

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


class LocalPooledEncoder(VectorEncoder):
    """
    Local sentence-transformers pooled vector encoder.
    Produces a single normalized float32 vector per text (mean-pooling).
    Enforces offline restrictions and explicit asset preparation.
    Fails explicitly if model weights/assets are missing or corrupt.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        local_asset_path: Optional[str] = None,
        offline: bool = False,
        revision: str = "v1",
        cache_size: int = 2048,
    ):
        self._model_name = model_name
        self._local_asset_path = local_asset_path
        self._offline = offline or (os.environ.get("NSN_OFFLINE", "0").lower() in ("1", "true", "yes"))
        self._revision = revision
        self._dim = 384
        self._model = None

        # Check offline assets before executing
        if self._offline:
            if not self._local_asset_path or not os.path.exists(self._local_asset_path):
                raise AssetNotFoundError(
                    f"Offline mode active: local model assets not found at '{self._local_asset_path}'. "
                    f"Remote downloads are prohibited. Run prepare_assets() or specify a valid local_asset_path."
                )

        self._encode_cached = functools.lru_cache(maxsize=cache_size)(self._encode_raw)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"st_{self._model_name}_{self._revision}_{self._dim}"

    def _load_model(self):
        if self._model is not None:
            return self._model

        # Strictly validate local assets if path is provided
        if self._local_asset_path:
            if not os.path.exists(self._local_asset_path):
                raise AssetNotFoundError(
                    f"Local model asset path does not exist: '{self._local_asset_path}'"
                )
            # Must contain valid model configuration files
            cfg_file = os.path.join(self._local_asset_path, "config.json")
            modules_file = os.path.join(self._local_asset_path, "modules.json")
            if not os.path.exists(cfg_file) and not os.path.exists(modules_file):
                raise AssetNotFoundError(
                    f"Local model asset directory '{self._local_asset_path}' is missing config.json or modules.json. "
                    f"Corrupt or incomplete weights cannot be loaded."
                )
            path_to_load = self._local_asset_path
        else:
            if self._offline:
                raise AssetNotFoundError(
                    f"Cannot load remote model '{self._model_name}' in offline mode without local_asset_path."
                )
            path_to_load = self._model_name

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(path_to_load)
            return self._model
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for LocalPooledEncoder. "
                "Install via: pip install 'nsn[semantic]'"
            ) from exc

    def _encode_raw(self, text: str) -> Tuple[float, ...]:
        model = self._load_model()
        # sentence_embedding produces mean-pooled 1D vector
        out = model.encode(text, output_value="sentence_embedding", normalize_embeddings=True)
        if hasattr(out, "tolist"):
            return tuple(float(x) for x in out.tolist())
        return tuple(float(x) for x in out)

    def encode(self, text: str) -> List[float]:
        return list(self._encode_cached(text))

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        outs = model.encode(texts, output_value="sentence_embedding", normalize_embeddings=True)
        return [list(float(x) for x in row) for row in outs]

    @staticmethod
    def prepare_assets(model_name: str = "all-MiniLM-L6-v2", target_dir: Optional[str] = None) -> str:
        return prepare_assets(model_name=model_name, target_dir=target_dir)


def prepare_assets(model_name: str = "all-MiniLM-L6-v2", target_dir: Optional[str] = None) -> str:
    """
    Explicitly pre-download and save full model assets locally for offline use.
    Saves weights and metadata manifest.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to prepare model assets. "
            "Install via: pip install 'nsn[semantic]'"
        ) from exc

    out_dir = target_dir or os.path.join(os.path.expanduser("~"), ".cache", "nsn", "models", model_name)
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    model.save(out_dir)

    import json
    manifest_path = os.path.join(out_dir, "model_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "dimension": 384,
            "revision": "v1",
            "fingerprint": f"st_{model_name}_v1_384",
            "status": "prepared",
        }, f, indent=2)

    return out_dir

