"""
Tests for Phase 02: Facts, Corrections, History, Provenance.
Verifies temporal validity, conflict groups, set-valued attributes,
historical queries (as-of), question rejection, and event deletion invalidation.
"""
import pytest
from datetime import datetime, timezone, timedelta
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.facts import FactManager, Fact


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test_facts.db")
    st = SQLiteAdapter(db_path)
    yield st
    st.close()


@pytest.fixture
def fact_manager(storage):
    return FactManager(storage)


def test_fact_update_and_historical_query(fact_manager, storage):
    """
    Criterion 1: Ingesting 'port 8080' then 'port 9090' leaves both raw events intact
    and yields current port 9090 while retaining port 8080 as historical truth for queries
    asking about the earlier time.
    """
    ns = "test_ns"
    t1 = "2026-01-01T10:00:00Z"
    t2 = "2026-01-02T10:00:00Z"

    # Append events
    e1_id = storage.append_event(
        namespace=ns,
        role="user",
        content="Service auth production port is 8080",
        event_time=t1,
    )
    e2_id = storage.append_event(
        namespace=ns,
        role="user",
        content="Service auth production port is updated to 9090",
        event_time=t2,
    )

    # Verify both raw events exist
    assert storage.get_event(e1_id, ns) is not None
    assert storage.get_event(e2_id, ns) is not None

    # Record facts
    f1 = fact_manager.record_fact(
        subject="auth",
        predicate="production_port",
        value=8080,
        namespace=ns,
        value_type="number",
        valid_from=t1,
        source_authority=0.8,
        supporting_event_ids=[e1_id],
    )
    assert f1.status in ("confirmed", "asserted")

    f2 = fact_manager.record_fact(
        subject="auth",
        predicate="production_port",
        value=9090,
        namespace=ns,
        value_type="number",
        valid_from=t2,
        source_authority=0.8,
        supporting_event_ids=[e2_id],
    )

    # Current fact should be 9090
    current = fact_manager.get_current_fact("auth", "production_port", ns)
    assert current is not None
    assert current.value == 9090
    assert current.status in ("confirmed", "asserted")

    # Historical query as-of t1 + 1 hour -> should be 8080
    historical = fact_manager.get_fact_as_of("auth", "production_port", "2026-01-01T11:00:00Z", ns)
    assert historical is not None
    assert historical.value == 8080

    # Historical query as-of t2 + 1 hour -> should be 9090
    historical_t2 = fact_manager.get_fact_as_of("auth", "production_port", "2026-01-02T11:00:00Z", ns)
    assert historical_t2 is not None
    assert historical_t2.value == 9090


def test_out_of_order_event_ingestion(fact_manager):
    """
    Criterion 2: Ingesting Day 3 after Day 4 does not overwrite Day 4 as current.
    """
    ns = "test_ns"
    t_day4 = "2026-01-04T12:00:00Z"
    t_day3 = "2026-01-03T12:00:00Z"

    # Ingest Day 4 first
    f_day4 = fact_manager.record_fact(
        subject="api",
        predicate="database_host",
        value="db-replica-2.internal",
        namespace=ns,
        valid_from=t_day4,
        source_authority=0.8,
    )
    assert f_day4.value == "db-replica-2.internal"
    assert f_day4.status in ("confirmed", "asserted")

    # Now ingest Day 3 (out of order arrival)
    f_day3 = fact_manager.record_fact(
        subject="api",
        predicate="database_host",
        value="db-replica-1.internal",
        namespace=ns,
        valid_from=t_day3,
        source_authority=0.8,
    )

    # Day 3 should immediately be marked superseded because Day 4 has a newer valid_from
    assert f_day3.status == "superseded"

    # Current fact must remain Day 4
    current = fact_manager.get_current_fact("api", "database_host", ns)
    assert current is not None
    assert current.value == "db-replica-2.internal"


def test_set_valued_attribute_preservation(fact_manager):
    """
    Criterion 3: Adding a new allowed user preserves existing users instead of overwriting the field.
    """
    ns = "test_ns"

    # Add user "alice"
    f1 = fact_manager.record_fact(
        subject="admin_role",
        predicate="allowed_users",
        value="alice",
        namespace=ns,
        cardinality="set",
    )
    assert f1.cardinality == "set"
    assert "alice" in f1.value

    # Add user "bob"
    f2 = fact_manager.record_fact(
        subject="admin_role",
        predicate="allowed_users",
        value="bob",
        namespace=ns,
        cardinality="set",
    )
    assert "alice" in f2.value
    assert "bob" in f2.value

    # Add "alice" again (idempotent / no duplicate)
    f3 = fact_manager.record_fact(
        subject="admin_role",
        predicate="allowed_users",
        value="alice",
        namespace=ns,
        cardinality="set",
    )
    assert f3.value == ["alice", "bob"]

    # Verify active facts
    current = fact_manager.get_current_fact("admin_role", "allowed_users", ns)
    assert current is not None
    assert set(current.value) == {"alice", "bob"}


def test_questions_not_recorded_as_facts(fact_manager):
    """
    Criterion 4: 'What port is service X using?' is not recorded as a confirmed fact that port is unknown/None.
    """
    ns = "test_ns"

    q1 = {
        "content": "What port is service X using?",
        "role": "user",
        "namespace": ns,
    }
    extracted1 = fact_manager.extract_conservative_facts(q1)
    assert len(extracted1) == 0

    q2 = {
        "content": "How do we configure the auth service?",
        "role": "user",
        "namespace": ns,
    }
    assert len(fact_manager.extract_conservative_facts(q2)) == 0

    # Unconfirmed assistant output does not become high-authority confirmed fact
    asst_msg = {
        "content": "The system cluster is ready.",
        "role": "assistant",
        "source": "assistant",
        "namespace": ns,
    }
    extracted_asst = fact_manager.extract_conservative_facts(asst_msg)
    if extracted_asst:
        for f in extracted_asst:
            # Must NOT be marked confirmed
            assert f.status != "confirmed"
            assert f.source_authority < 0.5


def test_contradictions_and_event_deletion_invalidation(fact_manager, storage):
    """
    Criterion 5: Ingesting contradictory statements from sources of equal authority marks
    the fact disputed, assigns a conflict group, and preserves both versions;
    deleting the supporting event invalidates derived facts without corrupting other facts.
    """
    ns = "test_ns"
    t = "2026-01-05T12:00:00Z"

    e1 = storage.append_event(
        namespace=ns,
        role="user",
        content="Service billing endpoint is https://billing-a.internal",
        event_time=t,
    )
    e2 = storage.append_event(
        namespace=ns,
        role="user",
        content="Service billing endpoint is https://billing-b.internal",
        event_time=t,
    )

    # Ingest equal authority at same valid_from
    f1 = fact_manager.record_fact(
        subject="billing",
        predicate="endpoint",
        value="https://billing-a.internal",
        namespace=ns,
        valid_from=t,
        source_authority=0.8,
        supporting_event_ids=[e1],
    )
    f2 = fact_manager.record_fact(
        subject="billing",
        predicate="endpoint",
        value="https://billing-b.internal",
        namespace=ns,
        valid_from=t,
        source_authority=0.8,
        supporting_event_ids=[e2],
    )

    # Check disputed status and conflict group
    f1_refreshed = fact_manager.get_fact(f1.id, ns)
    f2_refreshed = fact_manager.get_fact(f2.id, ns)

    assert f1_refreshed.status == "disputed"
    assert f2_refreshed.status == "disputed"
    assert f1_refreshed.conflict_group is not None
    assert f1_refreshed.conflict_group == f2_refreshed.conflict_group

    # Independent fact to verify isolation during deletion
    other_event = storage.append_event(
        namespace=ns,
        role="user",
        content="Independent fact",
    )
    f_other = fact_manager.record_fact(
        subject="other",
        predicate="key",
        value="value",
        namespace=ns,
        supporting_event_ids=[other_event],
    )

    # Delete supporting event e1
    deleted = storage.delete_event(e1, ns)
    assert deleted is True

    # f1 should now be marked deleted because its only supporting event was deleted
    f1_after = fact_manager.get_fact(f1.id, ns)
    assert f1_after.status == "deleted"

    # f2 and other fact must remain intact
    f2_after = fact_manager.get_fact(f2.id, ns)
    assert f2_after.status != "deleted"

    f_other_after = fact_manager.get_fact(f_other.id, ns)
    assert f_other_after.status != "deleted"
    assert f_other_after.value == "value"
