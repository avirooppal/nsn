"""
NSN Model Wrapping Logic.
Attaches runtime memory interception transparently to supported models.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from nsn.adapters import CallableAdapter, AsyncCallableAdapter, OpenAIChatProxy


class _OpenAIModelProxy:
    """Transparent proxy for OpenAI-compatible clients."""

    def __init__(self, model: Any, runtime: Any, namespace: str, **kwargs):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "_nsn_wrapped", True)
        object.__setattr__(self, "chat", OpenAIChatProxy(model.chat, runtime, namespace, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_model"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_model"), name, value)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_model")
        ns = object.__getattribute__(self, "namespace")
        return f"<NSN.WrappedOpenAI model={type(model).__name__} namespace='{ns}'>"


def wrap(
    model: Any,
    namespace: str = "default",
    runtime: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Wrap any supported model with NSN persistent memory.
    """
    if getattr(model, "_nsn_wrapped", False):
        # Prevent double-wrapping
        return model

    if runtime is None:
        from nsn import get_default_runtime
        runtime = get_default_runtime()
        if runtime is None:
            raise RuntimeError(
                "NSN Runtime is not initialized. Please call 'nsn.init(path)' before 'nsn.wrap(model)', "
                "or pass an explicit runtime: 'nsn.wrap(model, runtime=rt)'."
            )

    # Custom adapter override
    custom_adapter = kwargs.get("adapter")
    if custom_adapter is not None:
        wrapped = custom_adapter(model, runtime, namespace, **kwargs)
        setattr(wrapped, "_nsn_wrapped", True)
        setattr(wrapped, "runtime", runtime)
        setattr(wrapped, "namespace", namespace)
        return wrapped

    # OpenAI-compatible client
    if hasattr(model, "chat") and hasattr(getattr(model, "chat", None), "completions"):
        return _OpenAIModelProxy(model, runtime, namespace, **kwargs)

    # Async callable
    if asyncio.iscoroutinefunction(model) or (callable(getattr(model, "ainvoke", None)) and not callable(getattr(model, "invoke", None))):
        wrapped = AsyncCallableAdapter(model, runtime, namespace, **kwargs)
        setattr(wrapped, "_nsn_wrapped", True)
        setattr(wrapped, "runtime", runtime)
        setattr(wrapped, "namespace", namespace)
        return wrapped

    # Generic callable or LangChain invoke
    if callable(model) or hasattr(model, "invoke"):
        wrapped = CallableAdapter(model, runtime, namespace, **kwargs)
        setattr(wrapped, "_nsn_wrapped", True)
        setattr(wrapped, "runtime", runtime)
        setattr(wrapped, "namespace", namespace)
        return wrapped

    raise TypeError(
        f"Unsupported model type '{type(model).__name__}'. NSN can wrap generic callables, "
        f"LangChain models, and OpenAI-compatible clients. For custom clients, pass adapter=..."
    )
