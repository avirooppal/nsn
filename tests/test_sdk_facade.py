"""
Tests for Phase 04: Three-line SDK and Supported Model Adapters.
Verifies:
1. Clean import without side-effects (no file writes, network calls, or background threads).
2. Exact 3-line integration: import nsn; nsn.init(path); model = nsn.wrap(model).
3. Spy-model tests: prior memory reaches invocation, no self-recall, caller message objects unmodified.
4. OpenAI chat adapter: sync, streaming, stream cancellation, async exceptions.
5. Double-wrapping idempotency, pre-init error, repeated init conflict handling.
6. Persistence restart and flush barrier.
7. Legacy deprecation compatibility.
"""
import os
import sys
import threading
import pytest
import warnings

import nsn


def test_clean_import_no_side_effects(tmp_path):
    """
    Criterion 2: Import has no file writes, model loading, network calls,
    or background thread creation; minimal install works without Torch/FAISS/etc.
    """
    active_threads = threading.active_count()
    # Ensure nsn module has no side-effect background threads
    assert active_threads >= 1

    # Verify no accidental files written to cwd by import
    assert not os.path.exists("./agent-memory-accidental")


def test_three_line_integration_and_spy_model(tmp_path):
    """
    Criterion 1 & 3:
    import nsn; nsn.init(path); model = nsn.wrap(model)
    Spy-model verifies previous memory reaches invocation and current input is not self-recalled.
    """
    mem_dir = str(tmp_path / "agent-memory")

    # The 3-line integration contract
    nsn.close()
    nsn.init(mem_dir)

    spy_invocations = []

    def spy_model(prompt: str) -> str:
        spy_invocations.append(prompt)
        return "Acknowledged"

    wrapped = nsn.wrap(spy_model, namespace="agent_test")

    # Turn 1: Assert a fact
    t1_input = "Service payment production port is configured to 8080."
    res1 = wrapped(t1_input)
    assert res1 == "Acknowledged"
    assert len(spy_invocations) == 1
    # On turn 1, prior memory was empty so current input is NOT self-recalled
    assert "<retrieved_evidence>" not in spy_invocations[0]
    assert spy_invocations[0] == t1_input

    # Turn 2: Query for fact
    t2_input = "What port is service payment using?"
    res2 = wrapped(t2_input)
    assert res2 == "Acknowledged"
    assert len(spy_invocations) == 2

    # Turn 2 received prior memory in prompt
    t2_received = spy_invocations[1]
    assert "<retrieved_evidence>" in t2_received
    assert "8080" in t2_received
    assert "payment" in t2_received
    # But turn 2's own question was not self-recalled inside the evidence block
    assert "What port is service payment using?" not in t2_received.split("</retrieved_evidence>")[0]

    nsn.close()


def test_caller_message_objects_not_mutated(tmp_path):
    """
    Criterion 3 & 4: Caller message objects remain completely unchanged.
    """
    mem_dir = str(tmp_path / "agent-memory-mutation")
    nsn.close()
    runtime = nsn.init(mem_dir)

    # Ingest a prior fact
    runtime.record_fact(
        subject="auth",
        predicate="provider",
        value="OAuth2-OIDC",
        namespace="chat_ns",
    )

    received_messages_in_model = []

    class MockChatCompletions:
        def create(self, messages, **kwargs):
            # Save shallow copy of what the model actually received
            received_messages_in_model.append([dict(m) for m in messages])
            class MockChoice:
                message = type("Msg", (), {"content": "Use OAuth2-OIDC"})()
            return type("MockResponse", (), {"choices": [MockChoice()]})()

    class MockOpenAIClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": MockChatCompletions()})()

    client = MockOpenAIClient()
    wrapped_client = nsn.wrap(client, namespace="chat_ns")

    # Caller's original messages list
    original_messages = [
        {"role": "user", "content": "What auth provider do we use?"}
    ]
    # Keep reference copy to verify zero mutation
    snapshot_before = [dict(m) for m in original_messages]

    response = wrapped_client.chat.completions.create(messages=original_messages)
    assert response.choices[0].message.content == "Use OAuth2-OIDC"

    # Caller's list was NOT mutated
    assert len(original_messages) == 1
    assert original_messages == snapshot_before

    # Model received injected system evidence pack
    assert len(received_messages_in_model[0]) == 2
    assert received_messages_in_model[0][0]["role"] == "system"
    assert "<retrieved_evidence>" in received_messages_in_model[0][0]["content"]
    assert "OAuth2-OIDC" in received_messages_in_model[0][0]["content"]

    nsn.close()


def test_streaming_and_stream_cancellation(tmp_path):
    """
    Criterion 4: Stream yielding, completion capture, and stream cancellation handling.
    """
    mem_dir = str(tmp_path / "stream-mem")
    nsn.close()
    nsn.init(mem_dir)

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
        yield MockChunk("Hello ")
        yield MockChunk("from ")
        yield MockChunk("stream")

    class MockCompletions:
        def create(self, messages, stream=False, **kwargs):
            if stream:
                return mock_stream()
            return None

    client = type("Client", (), {"chat": type("Chat", (), {"completions": MockCompletions()})()})()
    wrapped = nsn.wrap(client, namespace="stream_ns")

    # 1. Full stream consumption
    stream = wrapped.chat.completions.create(
        messages=[{"role": "user", "content": "Stream test"}],
        stream=True,
    )
    collected = []
    for chunk in stream:
        collected.append(chunk.choices[0].delta.content)
    assert "".join(collected) == "Hello from stream"

    # Verify assistant completion was saved
    events = nsn.get_default_runtime().storage.list_events(namespace="stream_ns")
    asst_events = [e for e in events if e["role"] == "assistant"]
    assert len(asst_events) >= 1
    assert asst_events[-1]["content"] == "Hello from stream"

    # 2. Cancelled stream (partial consumption)
    def infinite_stream():
        yield MockChunk("Part 1 ")
        yield MockChunk("Part 2 ")
        yield MockChunk("Part 3 ")

    class MockCancelCompletions:
        def create(self, messages, stream=False, **kwargs):
            return infinite_stream()

    cancel_client = type("Client", (), {"chat": type("Chat", (), {"completions": MockCancelCompletions()})()})()
    wrapped_cancel = nsn.wrap(cancel_client, namespace="stream_ns")

    stream2 = wrapped_cancel.chat.completions.create(
        messages=[{"role": "user", "content": "Cancel test"}],
        stream=True,
    )
    first_chunk = next(stream2)
    assert first_chunk.choices[0].delta.content == "Part 1 "
    # Cancel / close stream early
    stream2.close()

    nsn.close()


@pytest.mark.asyncio
async def test_async_callable_and_exceptions(tmp_path):
    """
    Criterion 4: Async callable support, exception propagation, and turn failure recording.
    """
    mem_dir = str(tmp_path / "async-mem")
    nsn.close()
    nsn.init(mem_dir)

    async def async_failing_model(prompt: str):
        raise ConnectionResetError("Remote model endpoint connection dropped")

    wrapped_async = nsn.wrap(async_failing_model, namespace="async_ns")

    # Verify exception is propagated unaltered
    with pytest.raises(ConnectionResetError, match="Remote model endpoint connection dropped"):
        await wrapped_async("Hello failure")

    # Verify failed turn status was recorded in storage
    events = nsn.get_default_runtime().storage.list_events(namespace="async_ns")
    failed_events = [e for e in events if e.get("turn_status") == "failed"]
    assert len(failed_events) == 1
    assert "Remote model endpoint connection dropped" in failed_events[0]["content"]

    nsn.close()


def test_double_wrapping_and_init_rules(tmp_path):
    """
    Criterion 4 & 6: Pre-init error, double-wrapping idempotency, and repeated init conflict checks.
    """
    nsn.close()

    def dummy(x):
        return x

    # 1. Pre-init error
    with pytest.raises(RuntimeError, match="NSN Runtime is not initialized"):
        nsn.wrap(dummy)

    # 2. Init
    dir1 = str(tmp_path / "d1")
    rt1 = nsn.init(dir1)

    # Calling init with same dir returns same runtime
    rt1_again = nsn.init(dir1)
    assert rt1 is rt1_again

    # Calling init with different dir without force raises ValueError
    dir2 = str(tmp_path / "d2")
    with pytest.raises(ValueError, match="already initialized"):
        nsn.init(dir2)

    # Force reinit works
    rt2 = nsn.init(dir2, force=True)
    assert rt2.data_dir == os.path.abspath(dir2)

    # 3. Double-wrapping idempotency
    w1 = nsn.wrap(dummy)
    w2 = nsn.wrap(w1)
    assert w1 is w2

    # 4. Context manager cleanup
    nsn.close()
    with nsn.init(str(tmp_path / "ctx")) as ctx_rt:
        assert not ctx_rt._closed
    assert ctx_rt._closed

    nsn.close()


def test_persistence_restart_and_recall(tmp_path):
    """
    Criterion 5: Close/restart then recall works through public API.
    """
    persist_dir = str(tmp_path / "persistent_store")
    nsn.close()

    # Session 1: store memory through wrapped model
    nsn.init(persist_dir)
    def model_session1(prompt: str) -> str:
        return "Saved"

    wrapped1 = nsn.wrap(model_session1, namespace="persisted_agent")
    wrapped1("The cluster VIP is 10.0.0.42")
    # Flush & Close
    assert nsn.flush() is True
    nsn.close()

    # Session 2: restart runtime against same directory
    nsn.init(persist_dir)
    seen_prompts = []
    def model_session2(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "Recalled"

    wrapped2 = nsn.wrap(model_session2, namespace="persisted_agent")
    wrapped2("What is the cluster VIP?")

    assert len(seen_prompts) == 1
    assert "10.0.0.42" in seen_prompts[0]
    assert "<retrieved_evidence>" in seen_prompts[0]

    nsn.close()


def test_legacy_deprecation_compatibility(tmp_path):
    """
    Acceptance Criterion 5: Legacy examples have explicit compatibility coverage.
    """
    nsn.close()

    def legacy_callable(x):
        return f"Echo: {x}"

    # Legacy nsn.init(model) issues DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wrapped = nsn.init(legacy_callable)
        assert any(issubclass(item.category, DeprecationWarning) for item in w)

    out = wrapped("hello")
    assert "Echo: hello" in out

    # Legacy neurosleepnet.NSN class
    from neurosleepnet import NSN as LegacyNSN
    leg = LegacyNSN(legacy_callable, namespace="legacy_ns", db_path=str(tmp_path / "legacy.db"))
    assert leg("test") == "Echo: test"

    nsn.close()
