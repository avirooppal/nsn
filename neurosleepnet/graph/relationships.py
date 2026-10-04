"""Optional, source-backed SQLite relationship paths. No inferred logical rules."""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import sqlite3
import time
import uuid


def key(text):
    return " ".join(str(text).casefold().split())


def timestamp(value=None):
    dt = datetime.now(timezone.utc) if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Relationship times require a timezone")
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class PathResult:
    paths: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class RelationshipMemory:
    def __init__(self, storage):
        self.storage = storage
        self.db_path = storage.db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _entity(conn, namespace, name):
        name = key(name)
        if not namespace or not name:
            raise ValueError("Namespace and entity name are required")
        conn.execute("INSERT OR IGNORE INTO relationship_entities VALUES (?,?,?)", (namespace, str(uuid.uuid4()), name))
        eid = conn.execute("SELECT id FROM relationship_entities WHERE namespace=? AND name=?", (namespace, name)).fetchone()[0]
        conn.execute("INSERT OR IGNORE INTO relationship_aliases VALUES (?,?,?)", (namespace, name, eid))
        return eid

    def add_alias(self, name, alias, namespace="default"):
        conn = self._connect()
        try:
            with conn:
                eid = self._entity(conn, namespace, name)
                if not key(alias):
                    raise ValueError("Alias cannot be empty")
                conn.execute("INSERT OR IGNORE INTO relationship_aliases VALUES (?,?,?)", (namespace, key(alias), eid))
        finally:
            conn.close()
        return eid

    def resolve(self, alias, namespace="default"):
        conn = self._connect()
        try:
            rows = conn.execute("SELECT entity_id FROM relationship_aliases WHERE namespace=? AND alias=?", (namespace, key(alias))).fetchall()
            return rows[0][0] if len(rows) == 1 else None
        finally:
            conn.close()

    def record(self, subject, predicate, object, event_id, namespace="default",
               span_start=0, span_end=None, kind="explicit", confidence=1.0,
               valid_from=None, valid_to=None, replace=False, _expected_content=None):
        """Record a caller-asserted relation supported by an exact event span.

        replace=True closes earlier versions of the same subject/predicate.
        It requires increasing valid_from; out-of-order replacement is rejected.
        """
        if kind not in ("explicit", "association") or not 0 <= confidence <= 1:
            raise ValueError("Invalid relationship kind or confidence")
        predicate = key(predicate)
        if not predicate:
            raise ValueError("Predicate cannot be empty")
        if predicate.upper() in ("RELATED_TO", "ASSOCIATED_WITH", "CO_OCCURS"):
            kind = "association"
        vf, vt = timestamp(valid_from) if valid_from else None, timestamp(valid_to) if valid_to else None
        conn = self._connect()
        rid = str(uuid.uuid4())
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                event = conn.execute("SELECT content, turn_status, event_time, observed_at FROM events WHERE id=? AND namespace=?", (event_id, namespace)).fetchone()
                if not event or event["turn_status"] in ("cancelled", "deleted"):
                    raise ValueError("Support event is missing, cancelled, or outside namespace")
                if _expected_content is not None and event["content"] != _expected_content:
                    raise ValueError("source_changed_during_extraction")
                vf = vf or timestamp(event["event_time"] or event["observed_at"])
                if vt and vt <= vf:
                    raise ValueError("valid_to must be after valid_from")
                end = len(event["content"]) if span_end is None else span_end
                if not 0 <= span_start < end <= len(event["content"]):
                    raise ValueError("Support span must be nonempty and within the event")
                sid = self._entity(conn, namespace, subject)
                oid = self._entity(conn, namespace, object)
                existing = conn.execute("""SELECT id FROM relationships WHERE namespace=? AND subject_id=?
                    AND predicate=? AND object_id=? AND kind=? AND valid_from=? AND valid_to IS ?""",
                    (namespace, sid, predicate, oid, kind, vf, vt)).fetchone()
                if existing:
                    rid = existing[0]
                    conn.execute("INSERT OR IGNORE INTO relationship_support VALUES (?,?,?,?,?)", (namespace, rid, event_id, span_start, end))
                    return rid
                if replace:
                    prior = conn.execute("SELECT valid_from FROM relationships WHERE namespace=? AND subject_id=? AND predicate=? AND valid_to IS NULL", (namespace, sid, predicate)).fetchall()
                    if any(r[0] >= vf for r in prior):
                        raise ValueError("Replacement must be later than existing versions")
                    conn.execute("UPDATE relationships SET valid_to=? WHERE namespace=? AND subject_id=? AND predicate=? AND valid_to IS NULL", (vf, namespace, sid, predicate))
                conn.execute("INSERT INTO relationships VALUES (?,?,?,?,?,?,?,?,?)", (namespace, rid, sid, predicate, oid, kind, confidence, vf, vt))
                conn.execute("INSERT INTO relationship_support VALUES (?,?,?,?,?)", (namespace, rid, event_id, span_start, end))
        finally:
            conn.close()
        return rid

    def extract(self, event_id, namespace="default"):
        """Conservative grammar only. Unrecognized text is explicitly unresolved."""
        event = self.storage.get_event(event_id, namespace)
        if not event:
            raise ValueError("Support event is outside namespace or missing")
        text = event["content"]
        # No pronoun resolution, negation, questions, or guessed co-occurrence.
        match = re.fullmatch(r"([\w][\w -]*?) (works for|is located in) ([\w][\w -]*?)\.?", text.strip())
        names = [match[1], match[3]] if match else []
        ambiguous = (not match or bool(re.search(r"\b(not|never|maybe|possibly|allegedly|if|would|could|may)\b", text, re.I))
                     or any(key(n) in ("he", "she", "they", "it", "this", "that") or re.search(r"\b(and|or)\b", key(n)) for n in names))
        conn = self._connect()
        try:
            for i, name in enumerate(names):
                rows = conn.execute("""SELECT e.name FROM relationship_aliases a JOIN relationship_entities e
                    ON e.namespace=a.namespace AND e.id=a.entity_id WHERE a.namespace=? AND a.alias=?""", (namespace, key(name))).fetchall()
                if len(rows) > 1:
                    ambiguous = True
                elif rows:
                    names[i] = rows[0][0]
        finally:
            conn.close()
        if not ambiguous:
            return [self.record(names[0], {"works for": "works_for", "is located in": "located_in"}[match[2]], names[1],
                                event_id, namespace, confidence=0.95,
                                valid_from=event.get("event_time") or event["observed_at"], _expected_content=text)]
        conn = self._connect()
        try:
            with conn:
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + ":" + event_id + ":" + text))
                conn.execute("INSERT OR IGNORE INTO unresolved_relationships VALUES (?,?,?,?,?)", (uid, namespace, event_id, text, "unsupported_grammar_or_ambiguous_reference"))
        finally:
            conn.close()
        return []

    def traverse(self, query, namespace="default", as_of_time=None, max_depth=3,
                 max_nodes=32, max_edges=64, max_candidates=8, time_budget_ms=25,
                 max_tokens=256, count_tokens=None, session_id=None):
        """Directed explicit paths, bounded work and evidence size, at one DB snapshot."""
        limits = dict(max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges,
                      max_candidates=max_candidates, time_budget_ms=time_budget_ms, max_tokens=max_tokens)
        if any(v < 0 for v in limits.values()):
            raise ValueError("Traversal bounds cannot be negative")
        count = count_tokens or (lambda s: len(s.encode("utf-8")))
        start = time.perf_counter()
        deadline = start + time_budget_ms / 1000
        diag = {"limits": limits, "nodes": 0, "edges": 0, "tokens": 0, "truncated": [], "ambiguous_aliases": []}
        result = PathResult(diagnostics=diag)
        if not all(limits.values()):
            diag["truncated"].append("zero_budget")
            return result
        now = timestamp(as_of_time)
        with self.storage._runtime_connection() as conn:
            previous_factory = conn.row_factory
            conn.row_factory = sqlite3.Row
            conn.set_progress_handler(lambda: int(time.perf_counter() >= deadline), 100)
            try:
                conn.execute("BEGIN")
                aliases = conn.execute("""SELECT alias, MIN(entity_id) AS entity_id, COUNT(*) AS n
                    FROM relationship_aliases WHERE namespace=? AND instr(?, ' ' || alias || ' ')>0
                    GROUP BY alias ORDER BY length(alias) DESC, alias LIMIT ?""",
                    (namespace, " " + key(re.sub(r"[^\w -]", " ", query)) + " ", max_nodes + 1)).fetchall()
                seeds = []
                for row in aliases[:max_nodes]:
                    if row["n"] != 1:
                        diag["ambiguous_aliases"].append(row["alias"])
                    elif row["entity_id"] not in seeds:
                        seeds.append(row["entity_id"])
                if len(aliases) > max_nodes:
                    diag["truncated"].append("max_nodes")
                nodes = set(seeds)
                queue = deque((sid, [], {sid}) for sid in seeds)
                while queue:
                    if time.perf_counter() >= deadline:
                        diag["truncated"].append("time_budget_ms")
                        break
                    sid, path, visited = queue.popleft()
                    if len(path) >= max_depth:
                        diag["truncated"].append("max_depth")
                        continue
                    remaining = max_edges - diag["edges"]
                    if remaining <= 0:
                        diag["truncated"].append("max_edges")
                        break
                    rows = conn.execute("""SELECT r.*, s.name AS subject, o.name AS object
                        FROM relationships r
                        JOIN relationship_entities s ON s.namespace=r.namespace AND s.id=r.subject_id
                        JOIN relationship_entities o ON o.namespace=r.namespace AND o.id=r.object_id
                        WHERE r.namespace=? AND r.subject_id=? AND r.kind='explicit'
                          AND r.valid_from<=? AND (r.valid_to IS NULL OR r.valid_to>?)
                        ORDER BY r.id LIMIT ?""", (namespace, sid, now, now, remaining + 1)).fetchall()
                    if len(rows) > remaining:
                        diag["truncated"].append("max_edges")
                    for row in rows[:remaining]:
                        diag["edges"] += 1
                        oid = row["object_id"]
                        if oid in visited:
                            continue
                        if oid not in nodes and len(nodes) >= max_nodes:
                            diag["truncated"].append("max_nodes")
                            continue
                        supports = conn.execute("""SELECT e.id, e.content, e.source_identity, e.observed_at,
                            s.span_start, s.span_end FROM relationship_support s JOIN events e ON e.id=s.event_id AND e.namespace=s.namespace
                            WHERE s.namespace=? AND s.relation_id=? AND e.turn_status NOT IN ('cancelled','deleted')
                              AND (? IS NULL OR e.session_id=?)
                            ORDER BY e.id LIMIT 1""", (namespace, row["id"], session_id, session_id)).fetchall()
                        if not supports:
                            continue
                        support = dict(supports[0])
                        support["span"] = support.pop("content")[support["span_start"]:support["span_end"]]
                        edge = {k: row[k] for k in ("id", "subject", "predicate", "object", "confidence", "valid_from", "valid_to")}
                        edge["support"] = support
                        new_path = path + [edge]
                        nodes.add(oid)
                        queue.append((oid, new_path, visited | {oid}))
                        # Prefer longer paths when packing, but every candidate is self-contained.
                        rendered = self.render(new_path)
                        cost = count(rendered)
                        if cost > max_tokens:
                            diag["truncated"].append("max_tokens")
                            continue
                        result.paths.append({"edges": new_path, "content": rendered, "support_refs": list(dict.fromkeys(e["support"]["id"] for e in new_path)), "tokens": cost})
                        # Bound candidate accumulation as well as traversal expansion.
                        if len(result.paths) >= max_candidates:
                            diag["truncated"].append("max_candidates")
                            queue.clear()
                            break
                # A total token bound, prioritizing the longest complete chains.
                selected = []
                for p in sorted(result.paths, key=lambda p: (-len(p["edges"]), p["content"])):
                    if diag["tokens"] + p["tokens"] <= max_tokens:
                        selected.append(p)
                        diag["tokens"] += p["tokens"]
                    else:
                        diag["truncated"].append("max_tokens")
                result.paths = selected
                diag["nodes"] = len(nodes)
            except sqlite3.OperationalError as exc:
                if "interrupt" not in str(exc).lower():
                    raise
                result.paths = []
                diag["truncated"].append("time_budget_ms")
            finally:
                conn.set_progress_handler(None, 0)
                conn.rollback()
                conn.row_factory = previous_factory
                diag["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
                diag["truncated"] = sorted(set(diag["truncated"]))

        return result

    @staticmethod
    def render(edges):
        lines = ["[RELATION PATH: explicit source assertions; no inferred conclusion]"]
        for e in edges:
            s = e["support"]
            lines.append(f"{e['subject']} --{e['predicate']}--> {e['object']} | confidence={e['confidence']} | valid=[{e['valid_from']},{e['valid_to'] or 'open'})")
            lines.append(f"Source event {s['id']} ({s['source_identity']}, {s['observed_at']}) span[{s['span_start']}:{s['span_end']}]: {s['span']}")
        return "\n".join(lines)
