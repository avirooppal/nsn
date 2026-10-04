import sqlite3

import pytest
from nsn.runtime import Runtime


def relation(rt, subject, predicate, obj, namespace="default", **kwargs):
    event = rt.append_event(content=f"{subject} {predicate} {obj}.", namespace=namespace)
    rid = rt.record_relation(subject=subject, predicate=predicate, object=obj,
                             event_id=event, namespace=namespace, **kwargs)
    return rid, event


def test_scoped_entities_aliases_support_and_ambiguity(tmp_path):
    rt = Runtime(str(tmp_path), graph=True)
    _, ea = relation(rt, "Alice", "works_for", "Acme", namespace="a")
    relation(rt, "Alice", "works_for", "Beta", namespace="b")
    g = rt.relationships
    assert g.resolve("alice", "a") != g.resolve("alice", "b")
    g.add_alias("Alice", "Al", "a")
    assert g.resolve("al", "b") is None
    paths = g.traverse("Al", "a", max_tokens=1000).paths
    assert paths and paths[0]["edges"][0]["object"] == "acme"
    with pytest.raises(ValueError, match="Support event"):
        g.record("Alice", "knows", "Eve", ea, namespace="b")
    with pytest.raises(ValueError, match="span"):
        g.record("Alice", "knows", "Eve", ea, namespace="a", span_end=1000)
    g.add_alias("Another Alice", "Al", "a")
    assert g.resolve("Al", "a") is None
    result = g.traverse("Al", "a")
    assert result.paths == [] and result.diagnostics["ambiguous_aliases"] == ["al"]
    rt.close()


def test_temporal_complete_paths_and_deleted_support(tmp_path):
    rt = Runtime(str(tmp_path), graph=True)
    relation(rt, "Alice", "works_for", "Acme", valid_from="2025-01-01T00:00:00Z")
    _, old_source = relation(rt, "Acme", "located_in", "Paris", valid_from="2025-01-01T00:00:00Z")
    _, new_source = relation(rt, "Acme", "located_in", "Rome", valid_from="2026-01-01T00:00:00Z", replace=True)
    g = rt.relationships
    old = g.traverse("Alice", as_of_time="2025-06-01T00:00:00Z", max_tokens=1500).paths[0]
    current = g.traverse("Alice", max_tokens=1500).paths[0]
    assert len(old["edges"]) == 2 and old["edges"][-1]["object"] == "paris"
    assert len(current["edges"]) == 2 and current["edges"][-1]["object"] == "rome"
    assert old_source in old["support_refs"] and new_source not in old["support_refs"]
    rt.storage.delete_event(new_source, "default")
    assert all(len(p["edges"]) == 1 for p in g.traverse("Alice", max_tokens=1500).paths)
    rt.close()


def test_associations_and_uncertain_extraction_are_not_proof(tmp_path):
    rt = Runtime(str(tmp_path), graph=True)
    relation(rt, "Alice", "RELATED_TO", "Acme")
    assert rt.relationships.traverse("Alice", max_tokens=1000).paths == []
    for text in ("She works for Acme.", "Alice and Bob talked.", "Does Alice work for Acme?", "Alice does not work for Acme.", "Alice works for it.", "Alice never works for Acme."):
        eid = rt.append_event(content=text)
        assert rt.extract_relations(eid) == []
    eid = rt.append_event(content="Alice works for Acme.")
    assert len(rt.extract_relations(eid)) == 1
    conn = sqlite3.connect(rt.db_path)
    assert conn.execute("SELECT COUNT(*) FROM unresolved_relationships").fetchone()[0] == 6
    conn.close()
    assert rt.relationships.traverse("Alice", max_tokens=1000).paths
    rt.close()


def test_cycles_dense_neighborhoods_and_explicit_bounds(tmp_path):
    rt = Runtime(str(tmp_path), graph=True)
    relation(rt, "A", "links", "B")
    relation(rt, "B", "links", "A")
    for i in range(12):
        relation(rt, "A", "links", f"N{i}")
    result = rt.relationships.traverse("A", max_nodes=4, max_edges=5, max_depth=2,
                                       max_candidates=3, max_tokens=1500)
    assert result.diagnostics["nodes"] <= 4 and result.diagnostics["edges"] <= 5
    assert len(result.paths) <= 3
    assert all(len(p["edges"]) <= 2 for p in result.paths)
    assert result.diagnostics["truncated"]
    for option in ("time_budget_ms", "max_tokens", "max_depth", "max_edges", "max_nodes", "max_candidates"):
        assert rt.relationships.traverse("A", **{option: 0}).paths == []
    assert rt.relationships.traverse("A", max_tokens=1).paths == []
    rt.close()


def test_pack_contains_whole_supported_path_or_excludes_it(tmp_path):
    rt = Runtime(str(tmp_path), graph=True)
    _, e1 = relation(rt, "Alice", "works_for", "Acme")
    _, e2 = relation(rt, "Acme", "located_in", "Paris")
    pack = rt.retrieve_pack("Where is Alice's employer based?", max_tokens=512)
    paths = [p for p in pack.items if p.kind == "relation_path"]
    assert paths and {e1, e2} <= set(paths[0].support_refs)
    assert e1 in paths[0].content and e2 in paths[0].content
    assert "paris" in paths[0].content and pack.token_count <= 512
    tiny = rt.retrieve_pack("Alice", max_tokens=25)
    assert not any(p.kind == "relation_path" for p in tiny.items)
    assert rt.retrieve_pack("Alice", use_graph=False).diagnostic_explain["graph"] is None
    assert not any(p.kind == "relation_path" for p in rt.retrieve_pack("Alice", session_id="other").items)
    rt.close()
