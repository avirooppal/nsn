"""
Supported model adapters for NSN.
Implements the shared call lifecycle:
1. Retrieve prior evidence (before current input is written to storage).
2. Durably capture input event.
3. Invoke model once with delimited evidence pack (without mutating caller's arguments).
4. Record actual completion as assistant event and enqueue enrichment.
5. Preserve return types, exceptions, streaming chunks, and parameters.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("nsn.adapters")

NSN_PER_CALL_OPTIONS = frozenset({
    "input_window_tokens",
    "reserved_output_tokens",
    "tokenizer",
    "namespace",
    "auto_observe_inputs",
    "auto_observe_outputs",
    "as_of_time",
    "session_id",
    "max_memory_tokens",
    "memory_limit",
    "budget_breakdown",
    "chat_counter",
    "adapter",
    "use_graph",
    "graph_limits",
    "use_summaries",
})



def retrieval_options(defaults, per_call):
    keys = ("use_graph", "graph_limits", "use_summaries")
    return {**defaults, **{k: per_call[k] for k in keys if k in per_call}}

def extract_nsn_options(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Separate NSN-only per-call options from legitimate provider parameters.
    Returns:
        (nsn_opts, forward_kwargs)
    """
    nsn_opts: Dict[str, Any] = {}
    forward_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in NSN_PER_CALL_OPTIONS:
            nsn_opts[k] = v
        else:
            forward_kwargs[k] = v
    return nsn_opts, forward_kwargs


def extract_query_text(input_val: Any) -> str:
    """Extract string query from string, dict, or list of messages."""
    if isinstance(input_val, str):
        return input_val
    elif isinstance(input_val, dict):
        for k in ("content", "input", "query", "text", "prompt"):
            if k in input_val and isinstance(input_val[k], str):
                return input_val[k]
        return str(input_val)
    elif isinstance(input_val, list) and input_val:
        for m in reversed(input_val):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        last = input_val[-1]
        return extract_query_text(last)
    return str(input_val) if input_val is not None else ""


class StreamingResponseWrapper:
    """
    Wraps streaming response generator to capture completion text while
    yielding chunks transparently and handling stream cancellation cleanly.
    """

    def __init__(
        self,
        stream_iterator: Iterator[Any],
        on_complete: Callable[[str], None],
        on_cancel: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self._stream = stream_iterator
        self._on_complete = on_complete
        self._on_cancel = on_cancel
        self._on_error = on_error
        self._accumulated_text: List[str] = []
        self._finished = False

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            # Try extracting delta text from OpenAI chunk format
            try:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        self._accumulated_text.append(delta.content)
            except Exception:
                pass
            return chunk
        except StopIteration:
            if not self._finished:
                self._finished = True
                full_text = "".join(self._accumulated_text)
                self._on_complete(full_text)
            raise
        except Exception as e:
            if not self._finished:
                self._finished = True
                if self._on_error:
                    self._on_error(e)
            raise

    def close(self):
        if not self._finished:
            self._finished = True
            partial_text = "".join(self._accumulated_text)
            if self._on_cancel:
                self._on_cancel(partial_text)
        if hasattr(self._stream, "close") and callable(self._stream.close):
            self._stream.close()

    def __del__(self):
        if not self._finished:
            self._finished = True
            partial_text = "".join(self._accumulated_text)
            if self._on_cancel:
                self._on_cancel(partial_text)


class OpenAIChatCompletionsProxy:
    """Proxies client.chat.completions to intercept create() calls."""

    def __init__(self, real_completions: Any, runtime: Any, namespace: str, **kwargs):
        self._real = real_completions
        self._runtime = runtime
        self._namespace = namespace
        self._auto_in = kwargs.get("auto_observe_inputs", True)
        self._auto_out = kwargs.get("auto_observe_outputs", True)
        self._max_tokens = kwargs.get("max_tokens", 512)
        self._tokenizer = kwargs.get("tokenizer", None)
        self._retrieval_options = {k: kwargs[k] for k in ("use_graph", "graph_limits", "use_summaries") if k in kwargs}
        self._input_window_tokens = kwargs.get("input_window_tokens", None)
        self._reserved_output_tokens = kwargs.get("reserved_output_tokens", 0)

    def create(self, *, messages: list, **kwargs) -> Any:
        from neurosleepnet.retrieval.pack import ContextPacker

        # 0. Separate NSN-only per-call options from legitimate provider parameters
        nsn_opts, forward_kwargs = extract_nsn_options(kwargs)

        call_namespace = nsn_opts.get("namespace", self._namespace)
        call_auto_in = nsn_opts.get("auto_observe_inputs", self._auto_in)
        call_auto_out = nsn_opts.get("auto_observe_outputs", self._auto_out)
        call_tokenizer = nsn_opts.get("tokenizer", self._tokenizer)
        call_max_mem = (
            nsn_opts.get("max_memory_tokens")
            or nsn_opts.get("memory_limit")
            or self._max_tokens
        )
        call_input_window = nsn_opts.get("input_window_tokens", self._input_window_tokens)
        as_of_time = nsn_opts.get("as_of_time", None)
        session_id = nsn_opts.get("session_id", None)

        # Extract user query
        user_query = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                user_query = m.get("content", "")
                break

        # Calculate existing messages tokens accounting for message formatting & tool-schemas
        tools = forward_kwargs.get("tools")
        chat_counter = nsn_opts.get("chat_counter")
        existing_tokens, budget_breakdown = ContextPacker.count_chat_tokens(
            messages=messages,
            tools=tools,
            tokenizer=call_tokenizer,
            chat_counter=chat_counter,
        )

        # Calculate reserved output tokens
        reserved_output = (
            nsn_opts.get("reserved_output_tokens")
            or forward_kwargs.get("max_tokens")
            or forward_kwargs.get("max_completion_tokens")
            or self._reserved_output_tokens
            or 0
        )
        budget_breakdown["reserved_output_tokens"] = reserved_output
        budget_breakdown["input_window_tokens"] = call_input_window

        # 1. Retrieve prior evidence BEFORE recording current input (prevents self-recall)
        pack = None
        if user_query:
            pack = self._runtime.retrieve_pack(
                query=user_query,
                namespace=call_namespace,
                max_tokens=call_max_mem,
                tokenizer=call_tokenizer,
                input_window_tokens=call_input_window,
                existing_tokens=existing_tokens,
                reserved_output_tokens=reserved_output,
                as_of_time=as_of_time,
                session_id=session_id,
                budget_breakdown=budget_breakdown,
                **retrieval_options(self._retrieval_options, nsn_opts),
            )

        # 2. Durably capture input event
        if call_auto_in and user_query:
            event_id = self._runtime.append_event(
                namespace=call_namespace,
                role="user",
                content=user_query,
            )
            try:
                self._runtime.fact_manager.extract_conservative_facts({
                    "content": user_query,
                    "role": "user",
                    "namespace": call_namespace,
                    "id": event_id,
                })
            except Exception:
                pass

        # 3. Build enriched messages without mutating caller's original list
        enriched_messages = list(messages)
        if pack and pack.rendered_text:
            sys_msg = {
                "role": "system",
                "content": pack.rendered_text,
            }
            enriched_messages = [sys_msg] + enriched_messages

        # Check if streaming
        is_stream = forward_kwargs.get("stream", False)

        try:
            response = self._real.create(messages=enriched_messages, **forward_kwargs)
        except Exception as exc:
            if call_auto_in and user_query:
                # Record error turn
                self._runtime.append_event(
                    namespace=call_namespace,
                    role="assistant",
                    content=f"Error: {exc}",
                    turn_status="failed",
                )
            raise exc

        # Handle streaming vs non-streaming
        if is_stream:
            def on_complete(full_text: str):
                if call_auto_out and full_text:
                    self._runtime.append_event(
                        namespace=call_namespace,
                        role="assistant",
                        content=full_text,
                        turn_status="completed",
                    )

            def on_cancel(partial_text: str):
                if call_auto_out:
                    self._runtime.append_event(
                        namespace=call_namespace,
                        role="assistant",
                        content=partial_text or "[Stream cancelled by caller]",
                        turn_status="cancelled",
                    )

            def on_error(err: Exception):
                self._runtime.append_event(
                    namespace=call_namespace,
                    role="assistant",
                    content=f"Stream error: {err}",
                    turn_status="failed",
                )

            return StreamingResponseWrapper(
                response,
                on_complete=on_complete,
                on_cancel=on_cancel,
                on_error=on_error,
            )

        # Non-streaming completion
        try:
            if call_auto_out and hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                reply_text = getattr(msg, "content", None)
                if reply_text:
                    self._runtime.append_event(
                        namespace=call_namespace,
                        role="assistant",
                        content=reply_text,
                        turn_status="completed",
                    )
        except Exception:
            pass

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class OpenAIChatProxy:
    """Proxies client.chat to expose .completions."""

    def __init__(self, real_chat: Any, runtime: Any, namespace: str, **kwargs):
        self._real = real_chat
        self.completions = OpenAIChatCompletionsProxy(real_chat.completions, runtime, namespace, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class CallableAdapter:
    """
    Adapter for generic text callables: model(text) -> str or model(dict) -> Any.
    Supports direct __call__, .invoke(), and .ainvoke().
    """

    def __init__(self, model: Callable, runtime: Any, namespace: str, **kwargs):
        self._model = model
        self._runtime = runtime
        self._namespace = namespace
        self._auto_in = kwargs.get("auto_observe_inputs", True)
        self._auto_out = kwargs.get("auto_observe_outputs", True)
        self._max_tokens = kwargs.get("max_tokens", 512)
        self._tokenizer = kwargs.get("tokenizer", None)
        self._retrieval_options = {k: kwargs[k] for k in ("use_graph", "graph_limits", "use_summaries") if k in kwargs}
        self._input_window_tokens = kwargs.get("input_window_tokens", None)
        self._reserved_output_tokens = kwargs.get("reserved_output_tokens", 0)

    def __call__(self, input_val: Any = None, *args, **kwargs) -> Any:
        from neurosleepnet.retrieval.pack import ContextPacker

        # 0. Separate NSN-only per-call options from legitimate provider parameters
        nsn_opts, forward_kwargs = extract_nsn_options(kwargs)

        call_namespace = nsn_opts.get("namespace", self._namespace)
        call_auto_in = nsn_opts.get("auto_observe_inputs", self._auto_in)
        call_auto_out = nsn_opts.get("auto_observe_outputs", self._auto_out)
        call_tokenizer = nsn_opts.get("tokenizer", self._tokenizer)
        call_max_mem = (
            nsn_opts.get("max_memory_tokens")
            or nsn_opts.get("memory_limit")
            or self._max_tokens
        )
        call_input_window = nsn_opts.get("input_window_tokens", self._input_window_tokens)
        as_of_time = nsn_opts.get("as_of_time", None)
        session_id = nsn_opts.get("session_id", None)

        query = extract_query_text(input_val)

        # Calculate existing message/prompt tokens
        if isinstance(input_val, list):
            tools = forward_kwargs.get("tools")
            chat_counter = nsn_opts.get("chat_counter")
            existing_tokens, budget_breakdown = ContextPacker.count_chat_tokens(
                messages=input_val,
                tools=tools,
                tokenizer=call_tokenizer,
                chat_counter=chat_counter,
            )
        else:
            prompt_str = str(input_val or "")
            existing_tokens = ContextPacker.count_tokens(prompt_str, call_tokenizer)
            method = "exact" if call_tokenizer is not None else "estimated"
            budget_breakdown = {
                "total_tokens": existing_tokens,
                "content_tokens": existing_tokens,
                "message_formatting_overhead": 0,
                "tool_schema_tokens": 0,
                "token_estimation_method": method,
            }

        # Calculate reserved output tokens and input window tokens
        reserved_output = (
            nsn_opts.get("reserved_output_tokens")
            or forward_kwargs.get("max_tokens")
            or forward_kwargs.get("max_completion_tokens")
            or self._reserved_output_tokens
            or 0
        )
        budget_breakdown["reserved_output_tokens"] = reserved_output
        budget_breakdown["input_window_tokens"] = call_input_window

        # 1. Retrieve prior evidence BEFORE recording current input
        pack = None
        if query:
            pack = self._runtime.retrieve_pack(
                query=query,
                namespace=call_namespace,
                max_tokens=call_max_mem,
                tokenizer=call_tokenizer,
                input_window_tokens=call_input_window,
                existing_tokens=existing_tokens,
                reserved_output_tokens=reserved_output,
                as_of_time=as_of_time,
                session_id=session_id,
                budget_breakdown=budget_breakdown,
                **retrieval_options(self._retrieval_options, nsn_opts),
            )

        # 2. Durably capture input event
        if call_auto_in and query:
            event_id = self._runtime.append_event(
                namespace=call_namespace,
                role="user",
                content=query,
            )
            try:
                self._runtime.fact_manager.extract_conservative_facts({
                    "content": query,
                    "role": "user",
                    "namespace": call_namespace,
                    "id": event_id,
                })
            except Exception:
                pass

        # 3. Enrich input without mutating caller's original values
        enriched_input = input_val
        if pack and pack.rendered_text:
            if isinstance(input_val, str):
                enriched_input = f"{pack.rendered_text}\n\n{input_val}"
            elif isinstance(input_val, dict):
                enriched_input = dict(input_val)
                key = next((k for k in ("prompt", "input", "query", "text") if k in enriched_input), None)
                if key:
                    enriched_input[key] = f"{pack.rendered_text}\n\n{enriched_input[key]}"
                else:
                    enriched_input["context"] = pack.rendered_text
            elif isinstance(input_val, list):
                enriched_input = [{"role": "system", "content": pack.rendered_text}] + list(input_val)

        # 4. Invoke model once with forward_kwargs (NSN options stripped)
        try:
            if hasattr(self._model, "invoke") and callable(getattr(self._model, "invoke")):
                output = self._model.invoke(enriched_input, *args, **forward_kwargs)
            else:
                output = self._model(enriched_input, *args, **forward_kwargs) if input_val is not None else self._model(*args, **forward_kwargs)
        except Exception as exc:
            if call_auto_in and query:
                self._runtime.append_event(
                    namespace=call_namespace,
                    role="assistant",
                    content=f"Error: {exc}",
                    turn_status="failed",
                )
            raise exc

        # 5. Capture output
        if call_auto_out and output:
            out_str = output if isinstance(output, str) else str(output)
            self._runtime.append_event(
                namespace=call_namespace,
                role="assistant",
                content=out_str,
                turn_status="completed",
            )

        return output

    def invoke(self, input_val: Any = None, *args, **kwargs) -> Any:
        """Explicit invoke method routing through memory interception."""
        return self(input_val, *args, **kwargs)

    async def ainvoke(self, input_val: Any = None, *args, **kwargs) -> Any:
        """Async invoke method routing through memory interception."""
        if hasattr(self._model, "ainvoke") and callable(getattr(self._model, "ainvoke")):
            adapter = AsyncCallableAdapter(
                self._model,
                self._runtime,
                self._namespace,
                auto_observe_inputs=self._auto_in,
                auto_observe_outputs=self._auto_out,
                max_tokens=self._max_tokens,
                tokenizer=self._tokenizer,
                input_window_tokens=self._input_window_tokens,
                reserved_output_tokens=self._reserved_output_tokens,
                **self._retrieval_options,
            )
            return await adapter(input_val, *args, **kwargs)
        return self(input_val, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class AsyncCallableAdapter:
    """
    Adapter for async coroutine callables.
    Supports direct __call__, .ainvoke(), and .invoke().
    """

    def __init__(self, model: Callable, runtime: Any, namespace: str, **kwargs):
        self._model = model
        self._runtime = runtime
        self._namespace = namespace
        self._auto_in = kwargs.get("auto_observe_inputs", True)
        self._auto_out = kwargs.get("auto_observe_outputs", True)
        self._max_tokens = kwargs.get("max_tokens", 512)
        self._tokenizer = kwargs.get("tokenizer", None)
        self._retrieval_options = {k: kwargs[k] for k in ("use_graph", "graph_limits", "use_summaries") if k in kwargs}
        self._input_window_tokens = kwargs.get("input_window_tokens", None)
        self._reserved_output_tokens = kwargs.get("reserved_output_tokens", 0)

    async def __call__(self, input_val: Any = None, *args, **kwargs) -> Any:
        from neurosleepnet.retrieval.pack import ContextPacker

        # 0. Separate NSN-only per-call options from legitimate provider parameters
        nsn_opts, forward_kwargs = extract_nsn_options(kwargs)

        call_namespace = nsn_opts.get("namespace", self._namespace)
        call_auto_in = nsn_opts.get("auto_observe_inputs", self._auto_in)
        call_auto_out = nsn_opts.get("auto_observe_outputs", self._auto_out)
        call_tokenizer = nsn_opts.get("tokenizer", self._tokenizer)
        call_max_mem = (
            nsn_opts.get("max_memory_tokens")
            or nsn_opts.get("memory_limit")
            or self._max_tokens
        )
        call_input_window = nsn_opts.get("input_window_tokens", self._input_window_tokens)
        as_of_time = nsn_opts.get("as_of_time", None)
        session_id = nsn_opts.get("session_id", None)

        query = extract_query_text(input_val)

        # Calculate existing message/prompt tokens
        if isinstance(input_val, list):
            tools = forward_kwargs.get("tools")
            chat_counter = nsn_opts.get("chat_counter")
            existing_tokens, budget_breakdown = ContextPacker.count_chat_tokens(
                messages=input_val,
                tools=tools,
                tokenizer=call_tokenizer,
                chat_counter=chat_counter,
            )
        else:
            prompt_str = str(input_val or "")
            existing_tokens = ContextPacker.count_tokens(prompt_str, call_tokenizer)
            method = "exact" if call_tokenizer is not None else "estimated"
            budget_breakdown = {
                "total_tokens": existing_tokens,
                "content_tokens": existing_tokens,
                "message_formatting_overhead": 0,
                "tool_schema_tokens": 0,
                "token_estimation_method": method,
            }

        # Calculate reserved output tokens and input window tokens
        reserved_output = (
            nsn_opts.get("reserved_output_tokens")
            or forward_kwargs.get("max_tokens")
            or forward_kwargs.get("max_completion_tokens")
            or self._reserved_output_tokens
            or 0
        )
        budget_breakdown["reserved_output_tokens"] = reserved_output
        budget_breakdown["input_window_tokens"] = call_input_window

        pack = None
        if query:
            pack = self._runtime.retrieve_pack(
                query=query,
                namespace=call_namespace,
                max_tokens=call_max_mem,
                tokenizer=call_tokenizer,
                input_window_tokens=call_input_window,
                existing_tokens=existing_tokens,
                reserved_output_tokens=reserved_output,
                as_of_time=as_of_time,
                session_id=session_id,
                budget_breakdown=budget_breakdown,
                **retrieval_options(self._retrieval_options, nsn_opts),
            )

        if call_auto_in and query:
            event_id = self._runtime.append_event(
                namespace=call_namespace,
                role="user",
                content=query,
            )
            try:
                self._runtime.fact_manager.extract_conservative_facts({
                    "content": query,
                    "role": "user",
                    "namespace": call_namespace,
                    "id": event_id,
                })
            except Exception:
                pass

        enriched_input = input_val
        if pack and pack.rendered_text:
            if isinstance(input_val, str):
                enriched_input = f"{pack.rendered_text}\n\n{input_val}"
            elif isinstance(input_val, dict):
                enriched_input = dict(input_val)
                key = next((k for k in ("prompt", "input", "query", "text") if k in enriched_input), None)
                if key:
                    enriched_input[key] = f"{pack.rendered_text}\n\n{enriched_input[key]}"
                else:
                    enriched_input["context"] = pack.rendered_text
            elif isinstance(input_val, list):
                enriched_input = [{"role": "system", "content": pack.rendered_text}] + list(input_val)

        try:
            if hasattr(self._model, "ainvoke") and callable(getattr(self._model, "ainvoke")):
                output = await self._model.ainvoke(enriched_input, *args, **forward_kwargs)
            else:
                output = await self._model(enriched_input, *args, **forward_kwargs) if input_val is not None else await self._model(*args, **forward_kwargs)
        except asyncio.CancelledError:
            if call_auto_in and query:
                self._runtime.append_event(namespace=call_namespace, role="assistant",
                                           content="Model invocation cancelled", turn_status="cancelled")
            raise
        except Exception as exc:
            if call_auto_in and query:
                self._runtime.append_event(
                    namespace=call_namespace,
                    role="assistant",
                    content=f"Error: {exc}",
                    turn_status="failed",
                )
            raise exc

        if call_auto_out and output:
            out_str = output if isinstance(output, str) else str(output)
            self._runtime.append_event(
                namespace=call_namespace,
                role="assistant",
                content=out_str,
                turn_status="completed",
            )

        return output

    async def ainvoke(self, input_val: Any = None, *args, **kwargs) -> Any:
        """Explicit ainvoke method routing through memory interception."""
        return await self(input_val, *args, **kwargs)

    def invoke(self, input_val: Any = None, *args, **kwargs) -> Any:
        """Sync invoke method for async model."""
        return asyncio.run(self(input_val, *args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

