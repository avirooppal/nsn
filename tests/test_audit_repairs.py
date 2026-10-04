"""
Regression tests for Audit Repairs on Phases 00–04:
1. Clean import without requiring or loading optional ML dependencies.
2. Partial / cancelled streams recorded with turn_status='cancelled', NEVER 'completed'.
3. Context budget calculation accounting for existing messages and reserved output.
4. Direct .invoke() and .ainvoke() calls triggering full memory interception.
"""
import asyncio
import os
import subprocess
import sys
import pytest

import nsn


def test_clean_import_without_ml_dependencies():
    """
    Regression 1: Subprocess test verifying that 'import nsn' does not import
    any optional ML libraries (torch, faiss, spacy, sentence_transformers, numpy).
    """
    code = (
        "import sys, nsn\n"
        "ml = [m for m in ('torch', 'faiss', 'spacy', 'sentence_transformers', 'numpy') if m in sys.modules]\n"
        "assert not ml, f'Unwanted ML modules imported: {ml}'\n"
        "print('SUCCESS')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Stderr: {res.stderr}\nStdout: {res.stdout}"
    assert "SUCCESS" in res.stdout


def test_partial_cancelled_stream_stored_as_cancelled_not_completed(tmp_path):
    """
    Regression 2: Partial or cancelled streams must be recorded with
    turn_status='cancelled', never 'completed'.
    """
    mem_dir = str(tmp_path / "cancel_stream_mem")
    nsn.close()
    runtime = nsn.init(mem_dir)

    class MockDelta:
        def __init__(self, content):
            self.content = content

    class MockStreamChoice:
        def __init__(self, content):
            self.delta = MockDelta(content)

    class MockChunk:
        def __init__(self, text):
            self.choices = [MockStreamChoice(text)]

    def mock_stream():
        yield MockChunk("Chunk 1: Hello ")
        yield MockChunk("Chunk 2: Unread world ")
        yield MockChunk("Chunk 3: Never received")

    class MockCompletions:
        def create(self, messages, stream=False, **kwargs):
            return mock_stream()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": MockCompletions()})()})()
    wrapped = nsn.wrap(client, namespace="stream_audit_ns")

    # Start stream and only read the first chunk, then explicitly cancel / close
    stream = wrapped.chat.completions.create(
        messages=[{"role": "user", "content": "Tell me a long story"}],
        stream=True,
    )
    first_chunk = next(stream)
    assert first_chunk.choices[0].delta.content == "Chunk 1: Hello "

    # Cancel stream early
    stream.close()

    # Verify event stored in database
    events = runtime.storage.list_events(namespace="stream_audit_ns")
    asst_events = [e for e in events if e["role"] == "assistant"]
    assert len(asst_events) == 1
    ev = asst_events[0]

    # Crucial assertion: status must be 'cancelled', NOT 'completed'
    assert ev["turn_status"] == "cancelled", f"Expected turn_status='cancelled', got '{ev['turn_status']}'"
    assert "Chunk 1: Hello " in ev["content"]

    nsn.close()


def test_wrapper_context_budget_respects_existing_messages_and_reserved_output(tmp_path):
    """
    Regression 3: Context budgeting must account for existing messages
    and reserved output tokens.
    """
    mem_dir = str(tmp_path / "budget_mem")
    nsn.close()
    runtime = nsn.init(mem_dir)

    # Ingest a prior fact
    runtime.record_fact(
        subject="server",
        predicate="port",
        value=9090,
        namespace="budget_ns",
    )

    received_prompts = []

    class MockCompletions:
        def create(self, messages, **kwargs):
            received_prompts.append([dict(m) for m in messages])
            class MockChoice:
                message = type("Msg", (), {"content": "ok"})()
            return type("Resp", (), {"choices": [MockChoice()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": MockCompletions()})()})()
    wrapped = nsn.wrap(
        client,
        namespace="budget_ns",
        input_window_tokens=500,  # 500 total input window
    )

    # Case A: Existing conversation has 400 tokens + 80 tokens reserved output
    # Total used = 480 / 500 -> available space = 20 tokens (< 25 tokens threshold).
    # Expected: Empty pack injected (impossible budget), no prompt overflow.
    long_msg = "word " * 300  # ~400 tokens
    messages_crowded = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{long_msg} What is the server port?"},
    ]

    wrapped.chat.completions.create(
        messages=messages_crowded,
        max_tokens=80,  # 80 reserved output tokens
    )

    sent_messages_crowded = received_prompts[-1]
    # No memory evidence system message should be inserted because space was exhausted
    system_msgs = [m for m in sent_messages_crowded if m["role"] == "system" and "<retrieved_evidence>" in m["content"]]
    assert len(system_msgs) == 0, "Expected memory pack to be omitted due to exhausted window space"

    # Case B: Window with enough space (1000 window, short messages, 50 reserved output)
    # Available space = ~900 tokens. 20% cap = 180 tokens.
    # Expected: Memory evidence is properly injected within 180 token cap.
    messages_spacious = [
        {"role": "user", "content": "What is the server port?"}
    ]
    wrapped.chat.completions.create(
        messages=messages_spacious,
        input_window_tokens=1000,
        max_tokens=50,
    )
    sent_messages_spacious = received_prompts[-1]
    injected_sys = [m for m in sent_messages_spacious if m["role"] == "system" and "<retrieved_evidence>" in m["content"]]
    assert len(injected_sys) == 1
    assert "9090" in injected_sys[0]["content"]

    nsn.close()


def test_direct_invoke_and_ainvoke_interception(tmp_path):
    """
    Regression 4: Direct .invoke() and .ainvoke() calls must execute
    the full memory interception lifecycle, not bypass it.
    """
    mem_dir = str(tmp_path / "invoke_mem")
    nsn.close()
    runtime = nsn.init(mem_dir)

    # Ingest a prior fact
    runtime.record_fact(
        subject="database",
        predicate="port",
        value=5432,
        namespace="invoke_ns",
    )

    sync_calls = []
    async_calls = []

    class MockLangChainModel:
        def invoke(self, input_val, *args, **kwargs):
            sync_calls.append(input_val)
            return "sync_reply"

        async def ainvoke(self, input_val, *args, **kwargs):
            async_calls.append(input_val)
            return "async_reply"

    raw_model = MockLangChainModel()
    wrapped_model = nsn.wrap(raw_model, namespace="invoke_ns")

    # 1. Test direct .invoke()
    res_sync = wrapped_model.invoke("What is the database port?")
    assert res_sync == "sync_reply"
    assert len(sync_calls) == 1
    # Verify prior memory was injected into input
    assert "<retrieved_evidence>" in sync_calls[0]
    assert "5432" in sync_calls[0]

    # Verify input and assistant turns were recorded in storage
    events = runtime.storage.list_events(namespace="invoke_ns")
    roles = [e["role"] for e in events]
    assert "user" in roles
    assert "assistant" in roles
    assert events[-1]["content"] == "sync_reply"
    assert events[-1]["turn_status"] == "completed"

    # 2. Test direct .ainvoke()
    async def run_async_test():
        res_async = await wrapped_model.ainvoke("What is the database port again?")
        assert res_async == "async_reply"
        assert len(async_calls) == 1
        assert "<retrieved_evidence>" in async_calls[0]
        assert "5432" in async_calls[0]

    asyncio.run(run_async_test())

    nsn.close()


def test_strict_provider_signatures_and_nsn_options_consumption(tmp_path):
    """
    Regression 5: Underlying providers with strict method signatures (no **kwargs catch-all)
    must receive legitimate provider parameters unchanged while NSN-only per-call
    options are consumed cleanly without raising TypeError.
    """
    mem_dir = str(tmp_path / "strict_sig_mem")
    nsn.close()
    runtime = nsn.init(mem_dir)

    runtime.record_fact(
        subject="service",
        predicate="endpoint",
        value="https://api.internal.net",
        namespace="strict_ns",
    )

    # 1. Strict OpenAI-compatible chat completions provider (NO **kwargs)
    class StrictChatCompletions:
        def __init__(self):
            self.received_kwargs = {}

        def create(
            self,
            *,
            messages: list,
            model: str = "gpt-4o",
            temperature: float = 1.0,
            max_tokens: int = 100,
            tools: list = None,
            stream: bool = False,
        ):
            self.received_kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                "stream": stream,
            }
            class MockChoice:
                message = type("Msg", (), {"content": "Strict completions response"})()
            return type("Resp", (), {"choices": [MockChoice()]})()

    strict_comp = StrictChatCompletions()
    strict_client = type("Client", (), {"chat": type("Chat", (), {"completions": strict_comp})()})()
    wrapped_client = nsn.wrap(strict_client, namespace="strict_ns")

    tool_def = [{"type": "function", "function": {"name": "fetch_endpoint"}}]

    # Call with mix of provider parameters (model, temperature, max_tokens, tools)
    # and NSN-only per-call options (input_window_tokens, reserved_output_tokens, max_memory_tokens)
    resp = wrapped_client.chat.completions.create(
        messages=[{"role": "user", "content": "What is the service endpoint?"}],
        model="slm-chat-custom",
        temperature=0.4,
        max_tokens=220,
        tools=tool_def,
        input_window_tokens=4096,
        reserved_output_tokens=220,
        max_memory_tokens=150,
    )
    assert resp.choices[0].message.content == "Strict completions response"

    # Verify provider parameters were preserved unchanged
    received = strict_comp.received_kwargs
    assert received["model"] == "slm-chat-custom"
    assert received["temperature"] == 0.4
    assert received["max_tokens"] == 220
    assert received["tools"] == tool_def
    assert received["stream"] is False

    # Verify NSN-only options were stripped and NOT forwarded to strict provider
    assert "input_window_tokens" not in received
    assert "reserved_output_tokens" not in received
    assert "max_memory_tokens" not in received
    assert "namespace" not in received

    # Verify memory was retrieved and injected into messages
    sent_msgs = received["messages"]
    sys_msgs = [m for m in sent_msgs if m.get("role") == "system" and "<retrieved_evidence>" in m.get("content", "")]
    assert len(sys_msgs) == 1
    assert "https://api.internal.net" in sys_msgs[0]["content"]

    # 2. Strict generic callable model (NO **kwargs)
    class StrictCallableModel:
        def __init__(self):
            self.call_args = {}

        def __call__(self, prompt: str, temperature: float = 1.0, max_tokens: int = 100):
            self.call_args = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return "Strict callable response"

    strict_callable = StrictCallableModel()
    wrapped_callable = nsn.wrap(strict_callable, namespace="strict_ns")

    res_call = wrapped_callable(
        "What is the service endpoint?",
        temperature=0.2,
        max_tokens=150,
        input_window_tokens=2048,
        namespace="strict_ns",
        max_memory_tokens=100,
    )
    assert res_call == "Strict callable response"
    assert strict_callable.call_args["temperature"] == 0.2
    assert strict_callable.call_args["max_tokens"] == 150
    assert "input_window_tokens" not in strict_callable.call_args
    assert "max_memory_tokens" not in strict_callable.call_args
    assert "https://api.internal.net" in strict_callable.call_args["prompt"]

    # 3. Strict LangChain invoke model (NO **kwargs)
    class StrictLangChainModel:
        def __init__(self):
            self.invoke_args = {}

        def invoke(self, input: str, temperature: float = 1.0):
            self.invoke_args = {
                "input": input,
                "temperature": temperature,
            }
            return "Strict invoke response"

    strict_lc = StrictLangChainModel()
    wrapped_lc = nsn.wrap(strict_lc, namespace="strict_ns")

    res_invoke = wrapped_lc.invoke(
        "What is the service endpoint?",
        temperature=0.1,
        input_window_tokens=2048,
        reserved_output_tokens=100,
    )
    assert res_invoke == "Strict invoke response"
    assert strict_lc.invoke_args["temperature"] == 0.1
    assert "input_window_tokens" not in strict_lc.invoke_args
    assert "https://api.internal.net" in strict_lc.invoke_args["input"]

    nsn.close()


def test_context_budget_message_formatting_and_tool_schema_overhead(tmp_path):
    """
    Regression 6: Context budgeting must account for message formatting overhead
    and tool-schema tokens. Estimation method must be honestly labeled
    ('estimated' when message/tool formatting is estimated, 'exact' when using actual chat template/counter).
    """
    mem_dir = str(tmp_path / "budget_overhead_mem")
    nsn.close()
    runtime = nsn.init(mem_dir)

    runtime.record_fact(
        subject="database",
        predicate="host",
        value="db.internal.local",
        namespace="budget_ns",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute an analytical SQL query against the internal read replica database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The SQL statement to execute."},
                        "timeout_seconds": {"type": "integer", "description": "Execution timeout limit."},
                    },
                    "required": ["sql"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a database administrative assistant."},
        {"role": "user", "content": "What is the database host?"},
    ]

    # 1. Formatting overhead estimated without tokenizer
    total_est, breakdown_est = runtime.packer.count_chat_tokens(messages=messages, tools=tools, tokenizer=None)
    assert breakdown_est["token_estimation_method"] == "estimated"
    assert breakdown_est["message_formatting_overhead"] > 0
    assert breakdown_est["tool_schema_tokens"] > 30
    assert total_est == (
        breakdown_est["content_tokens"]
        + breakdown_est["message_formatting_overhead"]
        + breakdown_est["tool_schema_tokens"]
    )

    # 2. Raw tokenizer still estimates formatting overhead -> label is 'estimated'
    def mock_raw_tokenizer(text: str) -> int:
        return len(text.split())

    total_raw, breakdown_raw = runtime.packer.count_chat_tokens(
        messages=messages,
        tools=tools,
        tokenizer=mock_raw_tokenizer,
    )
    assert breakdown_raw["token_estimation_method"] == "estimated"

    # 3. Provider-specific chat counter -> label is 'exact'
    def mock_provider_chat_counter(msgs, tls):
        return 42

    total_exact, breakdown_exact = runtime.packer.count_chat_tokens(
        messages=messages,
        tools=tools,
        chat_counter=mock_provider_chat_counter,
    )
    assert breakdown_exact["token_estimation_method"] == "exact"
    assert total_exact == 42

    # 4. Retrieve pack through runtime and inspect honest labeling & budget breakdown in diagnostic_explain
    pack_est = runtime.retrieve_pack(
        query="database host",
        namespace="budget_ns",
        max_tokens=200,
        input_window_tokens=500,
        existing_tokens=total_est,
        reserved_output_tokens=50,
        tokenizer=None,
        budget_breakdown=breakdown_est,
    )
    assert pack_est.token_estimation_method == "estimated"
    assert pack_est.diagnostic_explain["token_estimation_method"] == "estimated"
    assert pack_est.diagnostic_explain["budget_breakdown"]["tool_schema_tokens"] == breakdown_est["tool_schema_tokens"]

    pack_exact = runtime.retrieve_pack(
        query="database host",
        namespace="budget_ns",
        max_tokens=200,
        input_window_tokens=500,
        existing_tokens=total_exact,
        reserved_output_tokens=50,
        budget_breakdown=breakdown_exact,
    )
    assert pack_exact.token_estimation_method == "exact"
    assert pack_exact.diagnostic_explain["token_estimation_method"] == "exact"

    # 5. Large tool schema consuming available budget leads to impossible token budget
    huge_tools = [
        {
            "type": "function",
            "function": {
                "name": f"fn_{i}",
                "description": "Very long tool description " * 15,
                "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
            },
        }
        for i in range(10)
    ]
    total_huge, breakdown_huge = runtime.packer.count_chat_tokens(messages=messages, tools=huge_tools, tokenizer=None)
    pack_exhausted = runtime.retrieve_pack(
        query="database host",
        namespace="budget_ns",
        max_tokens=200,
        input_window_tokens=350,
        existing_tokens=total_huge,
        reserved_output_tokens=50,
        budget_breakdown=breakdown_huge,
    )
    assert pack_exhausted.rendered_text == ""
    assert pack_exhausted.diagnostic_explain["reason"] == "impossible_token_budget"
    assert pack_exhausted.diagnostic_explain["budget_breakdown"]["tool_schema_tokens"] > 200

    nsn.close()


def test_chat_token_count_labels_estimated_vs_exact(tmp_path):
    """
    Regression 7: Label chat token counts 'estimated' whenever message or tool
    formatting overhead is estimated. Label counts 'exact' only when using
    the target model's actual chat template or provider-specific counter.
    """
    from neurosleepnet.retrieval.pack import ContextPacker

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Hello, world!"},
    ]
    tools = [{"type": "function", "function": {"name": "test_fn", "description": "test"}}]

    # Case A: No tokenizer -> 'estimated'
    _, b_none = ContextPacker.count_chat_tokens(messages, tools=tools, tokenizer=None)
    assert b_none["token_estimation_method"] == "estimated"

    # Case B: Plain text tokenizer without apply_chat_template -> 'estimated' (formatting overhead is estimated)
    class PlainTokenizer:
        def encode(self, text):
            return text.split()

    _, b_plain = ContextPacker.count_chat_tokens(messages, tools=tools, tokenizer=PlainTokenizer())
    assert b_plain["token_estimation_method"] == "estimated"

    # Case C: Target model tokenizer with actual apply_chat_template -> 'exact'
    class ModelTokenizerWithTemplate:
        def encode(self, text):
            return text.split()

        def apply_chat_template(self, msgs, tools=None, tokenize=False):
            # Model's actual chat template
            formatted = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
            if tools:
                formatted += f"<|tools|>{tools}<|end_tools|>"
            return formatted

    tot_tpl, b_tpl = ContextPacker.count_chat_tokens(messages, tools=tools, tokenizer=ModelTokenizerWithTemplate())
    assert b_tpl["token_estimation_method"] == "exact"
    assert tot_tpl > 0

    # Case D: Provider-specific counter callback -> 'exact'
    tot_cb, b_cb = ContextPacker.count_chat_tokens(
        messages,
        tools=tools,
        chat_counter=lambda m, t: 55,
    )
    assert b_cb["token_estimation_method"] == "exact"
    assert tot_cb == 55


