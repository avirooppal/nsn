"""
Compact Vector Storage for SLMs and AI Agents.
Provides lightweight, binary-serialized pooled vector storage and exact cosine search
directly in SQLite without requiring external ANN dependencies or heavy JSON overhead.
"""
from dataclasses import dataclass, asdict
import datetime
import math
import os
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None


class IncompatibleIndexError(Exception):
    """Raised when existing vector index metadata does not match encoder configuration."""
    pass


@dataclass
class VectorManifest:
    index_name: str
    model_name: str
    model_revision: str
    dimension: int
    preprocessing_fingerprint: str
    index_revision: int
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class CompactVectorStore:
    """
    Stores normalized pooled float32 vectors compactly in SQLite.
    Performs exact dot-product cosine similarity search with partition scoping,
    tombstone deletion, and pending-tail visibility.
    """

    def __init__(
        self,
        db_path: str,
        dimension: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
        revision: str = "v1",
        fingerprint: str = "default_pooled",
        index_name: str = "default_compact_index",
        model_revision: Optional[str] = None,
        preprocessing_fingerprint: Optional[str] = None,
    ):
        self.db_path = db_path
        self.dimension = int(dimension)
        self.model_name = str(model_name)
        self.revision = str(model_revision if model_revision is not None else revision)
        self.fingerprint = str(preprocessing_fingerprint if preprocessing_fingerprint is not None else fingerprint)
        self.index_name = str(index_name)

        # In-memory pending tail for immediate visibility across callers
        self._tail_vectors: Dict[str, Tuple[str, str, bytes]] = {}
        # In-memory matrix cache for warm searches
        self._matrix_cache: Dict[str, Tuple[List[str], List[str], Any]] = {}
        self._cache_revision = None

        self._initialize()

    @property
    def model_revision(self) -> str:
        return self.revision

    @property
    def preprocessing_fingerprint(self) -> str:
        return self.fingerprint

    @classmethod
    def from_encoder(cls, db_path: str, encoder: Any, index_name: str = "default_compact_index") -> "CompactVectorStore":
        dim = getattr(encoder, "dimension", 384)
        name = getattr(encoder, "model_name", "unknown_model")
        rev = getattr(encoder, "revision", getattr(encoder, "model_revision", "v1"))
        fp = getattr(encoder, "fingerprint", getattr(encoder, "preprocessing_fingerprint", "default"))
        return cls(
            db_path=db_path,
            dimension=dim,
            model_name=name,
            revision=rev,
            fingerprint=fp,
            index_name=index_name,
        )

    def _invalidate_cache(self, namespace: Optional[str] = None):
        if namespace:
            self._matrix_cache.pop(namespace, None)
            self._matrix_cache.pop((namespace, True), None)
        else:
            self._matrix_cache.clear()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _initialize(self):
        # Apply migrations if needed
        from neurosleepnet.storage.migrations import apply_migrations
        apply_migrations(self.db_path)

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT index_name, model_name, model_revision, dimension, preprocessing_fingerprint, index_revision, updated_at "
                "FROM vector_manifest WHERE index_name = ?",
                (self.index_name,)
            )
            row = cursor.fetchone()
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if row:
                stored_model = row[1]
                stored_rev = row[2]
                stored_dim = row[3]
                stored_fp = row[4]

                # Strict compatibility check across all four declared protocol attributes
                if stored_model != self.model_name:
                    raise IncompatibleIndexError(
                        f"Model identity mismatch for index '{self.index_name}': "
                        f"persisted='{stored_model}', requested='{self.model_name}'"
                    )
                if stored_dim != self.dimension:
                    raise IncompatibleIndexError(
                        f"Vector dimension mismatch for index '{self.index_name}': "
                        f"persisted={stored_dim}, requested={self.dimension}"
                    )
                if stored_rev != self.revision:
                    raise IncompatibleIndexError(
                        f"Model revision mismatch for index '{self.index_name}': "
                        f"persisted='{stored_rev}', requested='{self.revision}'"
                    )
                if stored_fp != self.fingerprint:
                    raise IncompatibleIndexError(
                        f"Preprocessing fingerprint mismatch for index '{self.index_name}': "
                        f"persisted='{stored_fp}', requested='{self.fingerprint}'"
                    )
            else:
                cursor.execute(
                    "INSERT INTO vector_manifest (index_name, model_name, model_revision, dimension, preprocessing_fingerprint, index_revision, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?)",
                    (self.index_name, self.model_name, self.revision, self.dimension, self.fingerprint, now)
                )
                conn.commit()
        finally:
            conn.close()

    def _normalize(self, vec: Union[List[float], Any]) -> List[float]:
        """Normalize vector to unit length L2."""
        if np is not None and isinstance(vec, np.ndarray):
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                return (vec / norm).astype(np.float32).tolist()
            return [0.0] * self.dimension

        sq_sum = sum(x * x for x in vec)
        norm = math.sqrt(sq_sum)
        if norm > 1e-12:
            return [float(x / norm) for x in vec]
        return [0.0] * self.dimension

    def _pack_vector(self, vec: List[float]) -> bytes:
        """Pack normalized float list into compact raw float32 bytes."""
        if len(vec) != self.dimension:
            raise ValueError(f"Vector length {len(vec)} does not match dimension {self.dimension}")
        if np is not None:
            return np.array(vec, dtype=np.float32).tobytes()
        return struct.pack(f"{self.dimension}f", *vec)

    def add(
        self,
        item_id: str,
        vector: Union[List[float], Any],
        namespace: str = "default",
        target_type: str = "fact",
        persist: bool = True,
        validate_target: bool = True,
    ) -> bool:
        """
        Add a compact vector for a fact or event.
        If validate_target=True, atomically validates current target state in SQLite
        alongside persistence, preventing deletion or supersession during encoding
        from restoring vectors.
        """
        norm_vec = self._normalize(vector)
        raw_blob = self._pack_vector(norm_vec)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if persist:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                with conn:
                    # SELECT alone does not begin a Python sqlite3 transaction.
                    # Reserve the writer before validating the authoritative target.
                    conn.execute("BEGIN IMMEDIATE")
                    if validate_target:
                        if target_type == "fact":
                            cursor.execute(
                                "SELECT status FROM facts WHERE id = ? AND namespace = ?",
                                (item_id, namespace)
                            )
                            row = cursor.fetchone()
                            if not row or row[0] in ("deleted", "superseded"):
                                cursor.execute(
                                    "UPDATE compact_vectors SET tombstoned = 1 WHERE id = ? AND namespace = ?",
                                    (item_id, namespace)
                                )
                                self._tail_vectors.pop(item_id, None)
                                self._invalidate_cache(namespace)
                                return False
                        elif target_type in ("event", "summary"):
                            cursor.execute(
                                "SELECT turn_status FROM events WHERE id = ? AND namespace = ?",
                                (item_id, namespace)
                            )
                            row = cursor.fetchone()
                            if not row or row[0] in ("cancelled", "deleted"):
                                cursor.execute(
                                    "UPDATE compact_vectors SET tombstoned = 1 WHERE id = ? AND namespace = ?",
                                    (item_id, namespace)
                                )
                                self._tail_vectors.pop(item_id, None)
                                self._invalidate_cache(namespace)
                                return False

                    cursor.execute(
                        "INSERT OR REPLACE INTO compact_vectors (id, namespace, target_type, vector, created_at, tombstoned) "
                        "VALUES (?, ?, ?, ?, ?, 0)",
                        (item_id, namespace, target_type, raw_blob, now)
                    )
            finally:
                conn.close()

        # Update in-memory tail and invalidate cache
        self._tail_vectors[item_id] = (namespace, target_type, raw_blob)
        self._invalidate_cache(namespace)
        return True

    def add_batch(self, items: List[Tuple[str, Union[List[float], Any], str, str]]):
        """
        Batch add items: (item_id, vector, namespace, target_type)
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records = []
        namespaces = set()
        for item_id, vec, ns, t_type in items:
            norm_vec = self._normalize(vec)
            blob = self._pack_vector(norm_vec)
            self._tail_vectors[item_id] = (ns, t_type, blob)
            records.append((item_id, ns, t_type, blob, now))
            namespaces.add(ns)

        for ns in namespaces:
            self._invalidate_cache(ns)

        if records:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    "INSERT OR REPLACE INTO compact_vectors (id, namespace, target_type, vector, created_at, tombstoned) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    records
                )
                conn.commit()
            finally:
                conn.close()

    def tombstone(self, item_id: str, namespace: Optional[str] = None):
        """Mark an item as tombstoned (soft-deleted)."""
        if item_id in self._tail_vectors:
            del self._tail_vectors[item_id]
        self._invalidate_cache(namespace)

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if namespace:
                cursor.execute(
                    "UPDATE compact_vectors SET tombstoned = 1 WHERE id = ? AND namespace = ?",
                    (item_id, namespace)
                )
            else:
                cursor.execute(
                    "UPDATE compact_vectors SET tombstoned = 1 WHERE id = ?",
                    (item_id,)
                )
            conn.commit()
        finally:
            conn.close()

    def delete(self, item_id: str, namespace: Optional[str] = None):
        """Delete vector physically or via tombstone."""
        self.tombstone(item_id, namespace)

    def search(
        self,
        query_vector: Union[List[float], Any],
        namespace: str = "default",
        target_type: Optional[str] = None,
        limit: int = 10,
        exclude_summaries: bool = False,
        eligible_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform exact dot-product cosine similarity search scoped to namespace.
        """
        norm_q = self._normalize(query_vector)
        if limit <= 0:
            return []
        # A different runtime may have changed or invalidated vectors.
        conn = self._get_connection()
        try:
            revision = conn.execute("SELECT revision FROM vector_changes WHERE id=1").fetchone()[0]
        finally:
            conn.close()
        if revision != self._cache_revision:
            self._invalidate_cache()
            self._cache_revision = revision

        if np is not None:
            # Check in-memory matrix cache first for fast warm search
            cache_key = (namespace, True) if exclude_summaries else namespace
            cached = self._matrix_cache.get(cache_key) if target_type is None else None
            if cached is not None:
                ids, types, matrix = cached
            else:
                conn = self._get_connection()
                cursor = conn.cursor()
                try:
                    if target_type:
                        cursor.execute(
                            "SELECT id, target_type, vector FROM compact_vectors "
                            "WHERE namespace = ? AND target_type = ? AND tombstoned = 0",
                            (namespace, target_type)
                        )
                    else:
                        cursor.execute(
                            "SELECT id, target_type, vector FROM compact_vectors "
                            "WHERE namespace = ? AND tombstoned = 0",
                            (namespace,)
                        )
                    rows = cursor.fetchall()
                    if exclude_summaries:
                        rows = [r for r in rows if r[1] != "summary"]
                finally:
                    conn.close()

                if not rows:
                    return []

                ids = [r[0] for r in rows]
                types = [r[1] for r in rows]
                all_blobs = b"".join(r[2] for r in rows)
                matrix = np.frombuffer(all_blobs, dtype=np.float32).reshape(len(rows), self.dimension)
                if target_type is None:
                    self._matrix_cache[cache_key] = (ids, types, matrix)

            if len(ids) == 0:
                return []

            q_arr = np.array(norm_q, dtype=np.float32)
            sims = np.dot(matrix, q_arr)

            # Filter before top-k, without narrowing the cached namespace matrix.
            indices = None
            if eligible_ids is not None:
                indices = np.asarray([i for i, item_id in enumerate(ids) if item_id in eligible_ids], dtype=np.intp)
                if not len(indices):
                    return []
                sims = sims[indices]

            k = min(limit, len(sims))
            if k == len(sims):
                top_indices = np.argsort(-sims)
            else:
                top_indices = np.argpartition(-sims, k)[:k]
                top_indices = top_indices[np.argsort(-sims[top_indices])]

            results = []
            for idx in top_indices:
                score = float(sims[idx])
                if indices is not None:
                    idx = indices[idx]
                results.append({
                    "id": ids[idx],
                    "target_type": types[idx],
                    "score": score,
                })
            return results
        else:
            # Pure Python fallback: query active rows from SQLite directly
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                if target_type:
                    cursor.execute(
                        "SELECT id, target_type, vector FROM compact_vectors "
                        "WHERE namespace = ? AND target_type = ? AND tombstoned = 0",
                        (namespace, target_type)
                    )
                else:
                    cursor.execute(
                        "SELECT id, target_type, vector FROM compact_vectors "
                        "WHERE namespace = ? AND tombstoned = 0",
                        (namespace,)
                    )
                rows = cursor.fetchall()
                if exclude_summaries:
                    rows = [r for r in rows if r[1] != "summary"]
            finally:
                conn.close()

            if not rows:
                return []

            unpack_fmt = f"{self.dimension}f"
            results = []
            for item_id, t_type, blob in rows:
                if eligible_ids is not None and item_id not in eligible_ids:
                    continue
                doc_vec = struct.unpack(unpack_fmt, blob)
                score = sum(q * d for q, d in zip(norm_q, doc_vec))
                results.append({
                    "id": item_id,
                    "target_type": t_type,
                    "score": float(score),
                })
            # Deterministic sorting: score DESC, id DESC
            results.sort(key=lambda x: (x["score"], x["id"]), reverse=True)
            return results[:limit]

    def count(self, namespace: Optional[str] = None, target_type: Optional[str] = None) -> int:
        """Count active (non-tombstoned) vectors."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            query = "SELECT COUNT(*) FROM compact_vectors WHERE tombstoned = 0"
            params = []
            if namespace:
                query += " AND namespace = ?"
                params.append(namespace)
            if target_type:
                query += " AND target_type = ?"
                params.append(target_type)
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def get_manifest(self) -> Optional[VectorManifest]:
        """Fetch index manifest."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT index_name, model_name, model_revision, dimension, preprocessing_fingerprint, index_revision, updated_at "
                "FROM vector_manifest WHERE index_name = ?",
                (self.index_name,)
            )
            row = cursor.fetchone()
            if row:
                return VectorManifest(
                    index_name=row[0],
                    model_name=row[1],
                    model_revision=row[2],
                    dimension=row[3],
                    preprocessing_fingerprint=row[4],
                    index_revision=row[5],
                    updated_at=row[6],
                )
            return None
        finally:
            conn.close()

    def rebuild(self, db_path: str, encoder: Any, namespace: Optional[str] = None) -> int:
        """
        Rebuild compact vectors for active facts and events from SQLite database.
        Clears existing vectors in namespace, re-encodes, and updates index revision.
        """
        rebuild_started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        items_to_add = []
        try:
            # Rebuild active facts
            if namespace:
                cursor.execute(
                    "SELECT id, subject, predicate, value, unit, namespace FROM facts "
                    "WHERE namespace = ? AND status IN ('asserted', 'confirmed', 'disputed')",
                    (namespace,)
                )
            else:
                cursor.execute(
                    "SELECT id, subject, predicate, value, unit, namespace FROM facts "
                    "WHERE status IN ('asserted', 'confirmed', 'disputed')"
                )
            fact_rows = cursor.fetchall()
            for r in fact_rows:
                fid, subj, pred, val, unit, ns = r[0], r[1], r[2], r[3], r[4], r[5]
                desc = f"{subj} {pred} is {val}"
                if unit:
                    desc += f" {unit}"
                vec = encoder.encode(desc)
                items_to_add.append((fid, vec, ns, "fact"))

            # Rebuild events
            if namespace:
                cursor.execute(
                    "SELECT id, content, namespace, role FROM events WHERE namespace = ?",
                    (namespace,)
                )
            else:
                cursor.execute("SELECT id, content, namespace, role FROM events")
            event_rows = cursor.fetchall()
            for r in event_rows:
                eid, content, ns = r[0], r[1], r[2]
                if content and content.strip():
                    vec = encoder.encode(content)
                    items_to_add.append((eid, vec, ns, "summary" if r[3] == "summary" else "event"))

            # Encode outside the writer transaction, then validate and replace
            # the index in one commit. A failed encoding leaves the old index intact.
            conn.execute("BEGIN IMMEDIATE")
            records = []
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for item_id, vec, ns, kind in items_to_add:
                table, column = ("facts", "status") if kind == "fact" else ("events", "turn_status")
                row = conn.execute(
                    f"SELECT {column} FROM {table} WHERE id=? AND namespace=?", (item_id, ns)
                ).fetchone()
                if row and row[0] not in ("deleted", "superseded", "cancelled"):
                    records.append((item_id, ns, kind, self._pack_vector(self._normalize(vec)), now))

            # Clear existing vectors for namespace
            if namespace:
                cursor.execute("DELETE FROM compact_vectors WHERE namespace = ? AND created_at <= ?", (namespace, rebuild_started))
            else:
                cursor.execute("DELETE FROM compact_vectors WHERE created_at <= ?", (rebuild_started,))

            # Increment index revision
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor.execute(
                "UPDATE vector_manifest SET index_revision = index_revision + 1, updated_at = ? WHERE index_name = ?",
                (now, self.index_name)
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO compact_vectors (id,namespace,target_type,vector,created_at,tombstoned) VALUES (?,?,?,?,?,0)",
                records,
            )
            conn.commit()
        finally:
            conn.close()

        # Clear in-memory tail
        if namespace:
            self._tail_vectors = {k: v for k, v in self._tail_vectors.items() if v[0] != namespace}
        else:
            self._tail_vectors.clear()

        self._invalidate_cache(namespace)
        return len(records)

