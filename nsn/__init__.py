"""
NSN — Lightweight, Self-Hosted Memory Layer for SLMs & AI Agents.

Minimal Integration::

    import nsn
    nsn.init("./agent-memory")
    model = nsn.wrap(model)
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional

from nsn.runtime import Runtime
from nsn.wrapper import wrap

__version__ = "0.3.0"

_default_runtime: Optional[Runtime] = None

def __getattr__(name: str) -> Any:
    if name == "Memory":
        from neurosleepnet.sdk.memory import Memory
        return Memory
    if name == "NSN":
        from neurosleepnet.sdk.wrapper import NSN
        return NSN
    raise AttributeError(f"module 'nsn' has no attribute '{name}'")


def init(directory: Any = "", default_namespace: str = "default", **kwargs) -> Any:
    """
    Initialize the NSN runtime with a local storage directory.

    Args:
        directory: Path to storage directory. Defaults to './agent-memory'.
        default_namespace: Default namespace for isolated agent memory.

    Returns:
        Runtime instance.
    """
    global _default_runtime

    # Handle legacy init(model, ...) deprecation gracefully
    if callable(directory) or hasattr(directory, "chat"):
        warnings.warn(
            "Passing a model to nsn.init() is deprecated. Use 'nsn.init(data_dir)' followed by 'nsn.wrap(model)'.",
            DeprecationWarning,
            stacklevel=2,
        )
        if _default_runtime is None:
            _default_runtime = Runtime("./agent-memory", default_namespace=default_namespace)
        return wrap(directory, runtime=_default_runtime, **kwargs)

    data_dir = directory if directory != "" else "./agent-memory"
    abs_dir = os.path.abspath(data_dir)

    if _default_runtime is not None:
        if _default_runtime.data_dir == abs_dir and not _default_runtime._closed:
            return _default_runtime
        elif not kwargs.get("force", False) and not _default_runtime._closed:
            raise ValueError(
                f"NSN runtime is already initialized with directory '{_default_runtime.data_dir}'. "
                f"To change directories, call nsn.close() first or pass force=True."
            )
        else:
            _default_runtime.close()

    runtime_kwargs = {k: v for k, v in kwargs.items() if k != "force"}
    _default_runtime = Runtime(abs_dir, default_namespace=default_namespace, **runtime_kwargs)
    return _default_runtime


def get_default_runtime() -> Optional[Runtime]:
    """Return currently active default runtime if initialized."""
    return _default_runtime


def flush(timeout_seconds: float = 5.0) -> bool:
    """Flush pending jobs and memory writes in the default runtime."""
    if _default_runtime is not None:
        return _default_runtime.flush(timeout_seconds=timeout_seconds)
    return True


def rebuild_vectors(namespace: Optional[str] = None) -> int:
    """Rebuild compact vectors in the default runtime."""
    if _default_runtime is not None:
        return _default_runtime.rebuild_vectors(namespace=namespace)
    return 0


def prepare_assets(target_dir: Optional[str] = None) -> str:
    """Prepare local embedding assets."""
    from neurosleepnet.embeddings.pooled import prepare_assets as _prep
    return _prep(target_dir=target_dir)


def close():
    """Idempotently close the active default runtime."""
    global _default_runtime
    if _default_runtime is not None:
        _default_runtime.close()
        _default_runtime = None


def sleep(**kwargs) -> dict:
    """Explicit bounded consolidation in the initialized runtime."""
    if _default_runtime is None:
        raise RuntimeError("Call nsn.init() before nsn.sleep()")
    return _default_runtime.sleep(**kwargs)


__all__ = [
    "init",
    "wrap",
    "Runtime",
    "get_default_runtime",
    "flush",
    "close",
    "rebuild_vectors",
    "prepare_assets",
    "sleep",
    "Memory",
    "NSN",
    "__version__",
]
