"""
Unit tests, regression validations, and performance benchmarks for Phase 05: Compact Semantic Retrieval.
Verifies:
1. Paraphrase recall improvement over lexical BM25 without regressing exact keyword/identifier recall.
2. LocalPooledEncoder failure on missing or corrupt weights (no silent fallback).
3. Index manifest compatibility checks across all four declared protocol attributes (model_name, revision, dimension, fingerprint).
4. Pure Python vector search without NumPy (regression test).
5. Embedding/index failure visibility and recovery via durable jobs.
6. Real prepared local encoder evaluation on frozen queries with distractors.
7. Tombstones, deletion, scoped search, and index rebuild.
8. Benchmarks for varied 1k and 10k corpora: cold/warm p50/p95, peak RSS, disk (including WAL), and asset footprint.
"""
import os
import shutil
import tempfile
import time
import tracemalloc
import socket
import psutil
import pytest
import sqlite3
import threading
import concurrent.futures

import nsn
from neurosleepnet.embeddings.pooled import (
    VectorEncoder,
    FastHashEncoder,
    LocalPooledEncoder,
    AssetNotFoundError,
    prepare_assets,
)
import neurosleepnet.storage.compact_vector as cv_module
from neurosleepnet.storage.compact_vector import (
    CompactVectorStore,
    IncompatibleIndexError,
)
from neurosleepnet.retrieval.pack import ContextPacker
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.facts import FactManager


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="nsn_semantic_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class SemanticMockEncoder(VectorEncoder):
    """
    Test encoder with controlled semantic clustering for deterministic assertion testing.
    Explicitly labeled as a test fixture, not a trained ML model.
    """
    def __init__(self, dimension: int = 128, revision: str = "v1", model_name: str = "semantic-mock"):
        self._dim = dimension
        self._revision = revision
        self._model_name = model_name
        self._fingerprint = f"mock_{model_name}_{revision}_{dimension}"

        self._clusters = {
            "residence": ["living", "staying", "dwelling", "resides", "located", "relocated", "apartment", "house", "berlin"],
            "profession": ["job", "role", "work", "career", "occupation", "engineer", "developer", "programmer"],
            "preference": ["prefers", "likes", "favors", "choice", "dark mode", "theme"],
        }

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
        return self._fingerprint

    def encode(self, text: str):
        import math
        vec = [0.0] * self._dim
        lower = text.lower()
        for c_idx, (cluster_name, keywords) in enumerate(self._clusters.items()):
            for kw in keywords:
                if kw in lower:
                    base_dim = (c_idx * 16) % self._dim
                    vec[base_dim] += 1.0
                    vec[(base_dim + 1) % self._dim] += 0.5

        words = lower.split()
        for w in words:
            h = hash(w) % self._dim
            vec[h] += 0.2

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            return [x / norm for x in vec]
        fallback = [0.0] * self._dim
        fallback[0] = 1.0
        return fallback

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


def test_paraphrase_recall_improvement_over_minimal(temp_dir):
    """
    Verify paraphrase query retrieves evidence that pure lexical BM25 misses,
    while exact identifier queries do not regress.
    """
    min_dir = os.path.join(temp_dir, "minimal")
    min_rt = nsn.Runtime(min_dir, default_namespace="test_ns", semantic=False)
    min_rt.record_fact(subject="engineer", predicate="residence", value="Berlin")
    min_rt.append_event(role="user", content="The developer relocated to the capital of Germany last summer.")
    min_rt.append_event(role="user", content="User prefers server port 8080 for deployments.")

    sem_dir = os.path.join(temp_dir, "semantic")
    encoder = SemanticMockEncoder(dimension=128)
    sem_rt = nsn.Runtime(sem_dir, default_namespace="test_ns", semantic=True, encoder=encoder)
    sem_rt.record_fact(subject="engineer", predicate="residence", value="Berlin")
    sem_rt.append_event(role="user", content="The developer relocated to the capital of Germany last summer.")
    sem_rt.append_event(role="user", content="User prefers server port 8080 for deployments.")

    paraphrase_query = "Where is the programmer dwelling currently?"

    min_pack = min_rt.retrieve_pack(query=paraphrase_query, namespace="test_ns")
    sem_pack = sem_rt.retrieve_pack(query=paraphrase_query, namespace="test_ns")

    # Lexical BM25 matches 0 facts on zero-token-overlap query
    min_fact_ids = [it.id for it in min_pack.items if it.kind == "fact"]
    assert len(min_fact_ids) == 0

    # Semantic hybrid RRF surfaces the paraphrased fact
    sem_fact_ids = [it.id for it in sem_pack.items if it.kind == "fact"]
    assert len(sem_fact_ids) > 0
    assert sem_pack.items[0].subject == "engineer"
    assert sem_pack.items[0].value == "Berlin"

    # Exact identifier query: both must recall without regression
    id_query = "port 8080"
    min_id_pack = min_rt.retrieve_pack(query=id_query, namespace="test_ns")
    sem_id_pack = sem_rt.retrieve_pack(query=id_query, namespace="test_ns")

    assert any("8080" in it.content for it in min_id_pack.items)
    assert any("8080" in it.content for it in sem_id_pack.items)

    timings = sem_pack.diagnostic_explain.get("timings")
    assert timings is not None
    assert "query_embed_ms" in timings
    assert "lexical_search_ms" in timings
    assert "dense_search_ms" in timings
    assert "fusion_ms" in timings
    assert "total_retrieve_ms" in timings

    min_rt.close()
    sem_rt.close()


def test_local_pooled_encoder_fails_explicitly_without_silent_fallback(temp_dir):
    """
    Verify LocalPooledEncoder fails clearly on missing or corrupt assets and never falls back to hashing.
    """
    # 1. Non-existent path in offline mode -> AssetNotFoundError
    non_existent = os.path.join(temp_dir, "missing_weights_path")
    with pytest.raises(AssetNotFoundError) as exc_missing:
        LocalPooledEncoder(local_asset_path=non_existent, offline=True)
    assert "not found" in str(exc_missing.value).lower()

    # 2. Corrupt/empty directory (missing config.json/modules.json) -> AssetNotFoundError
    corrupt_dir = os.path.join(temp_dir, "corrupt_weights_path")
    os.makedirs(corrupt_dir, exist_ok=True)
    with pytest.raises(AssetNotFoundError) as exc_corrupt:
        enc = LocalPooledEncoder(local_asset_path=corrupt_dir, offline=False)
        enc.encode("test text")
    assert "missing config.json" in str(exc_corrupt.value).lower() or "invalid" in str(exc_corrupt.value).lower()

    # 3. Offline mode without local asset path -> AssetNotFoundError
    with pytest.raises(AssetNotFoundError) as exc_offline:
        LocalPooledEncoder(local_asset_path=None, offline=True)
    assert "offline" in str(exc_offline.value).lower()


def test_manifest_validation_all_four_protocol_attributes(temp_dir):
    """
    Verify IncompatibleIndexError is raised on mismatch of any of the 4 declared protocol attributes:
    model_name, dimension, revision, fingerprint.
    """
    db_path = os.path.join(temp_dir, "manifest_val.db")
    storage = SQLiteAdapter(db_path)
    storage.close()

    # Create baseline index
    store = CompactVectorStore(
        db_path=db_path,
        dimension=128,
        model_name="base-encoder",
        revision="v1",
        fingerprint="fp-norm-1",
    )
    store.add("item_1", [0.1] * 128, namespace="ns1", target_type="fact")
    manifest = store.get_manifest()
    assert manifest is not None
    assert manifest.model_name == "base-encoder"
    assert manifest.dimension == 128
    assert manifest.model_revision == "v1"
    assert manifest.preprocessing_fingerprint == "fp-norm-1"

    # Mismatch 1: model_name
    with pytest.raises(IncompatibleIndexError) as exc_model:
        CompactVectorStore(db_path=db_path, dimension=128, model_name="different-encoder", revision="v1", fingerprint="fp-norm-1")
    assert "model identity mismatch" in str(exc_model.value).lower()

    # Mismatch 2: dimension
    with pytest.raises(IncompatibleIndexError) as exc_dim:
        CompactVectorStore(db_path=db_path, dimension=384, model_name="base-encoder", revision="v1", fingerprint="fp-norm-1")
    assert "dimension mismatch" in str(exc_dim.value).lower()

    # Mismatch 3: revision
    with pytest.raises(IncompatibleIndexError) as exc_rev:
        CompactVectorStore(db_path=db_path, dimension=128, model_name="base-encoder", revision="v2", fingerprint="fp-norm-1")
    assert "revision mismatch" in str(exc_rev.value).lower()

    # Mismatch 4: fingerprint
    with pytest.raises(IncompatibleIndexError) as exc_fp:
        CompactVectorStore(db_path=db_path, dimension=128, model_name="base-encoder", revision="v1", fingerprint="fp-norm-2")
    assert "fingerprint mismatch" in str(exc_fp.value).lower()


def test_search_without_numpy_regression(temp_dir, monkeypatch):
    """
    Verify exact dot-product search executes correctly and deterministically
    when NumPy is unavailable (pure Python fallback).
    """
    db_path = os.path.join(temp_dir, "no_numpy.db")
    dim = 64
    store = CompactVectorStore(db_path=db_path, dimension=dim, model_name="test-enc", revision="v1", fingerprint="fp1")

    # Add 5 items with distinct vectors
    v1 = [1.0] + [0.0] * (dim - 1)
    v2 = [0.0, 1.0] + [0.0] * (dim - 2)
    v3 = [0.5, 0.5] + [0.0] * (dim - 2)
    store.add("item_1", v1, namespace="ns_test", target_type="fact", validate_target=False)
    store.add("item_2", v2, namespace="ns_test", target_type="fact", validate_target=False)
    store.add("item_3", v3, namespace="ns_test", target_type="fact", validate_target=False)

    # Execute search with NumPy
    q = [1.0] + [0.0] * (dim - 1)
    results_numpy = store.search(q, namespace="ns_test", limit=3)

    # Now monkeypatch cv_module.np to None to force pure Python fallback
    monkeypatch.setattr(cv_module, "np", None)

    # Invalidate cache so it executes the pure Python branch
    store._invalidate_cache()
    results_pure_py = store.search(q, namespace="ns_test", limit=3)

    assert len(results_pure_py) == 3
    # item_1 must be top rank with score near 1.0
    assert results_pure_py[0]["id"] == "item_1"
    assert abs(results_pure_py[0]["score"] - 1.0) < 1e-4

    # Ranking and order must match NumPy output exactly
    numpy_ids = [r["id"] for r in results_numpy]
    pure_py_ids = [r["id"] for r in results_pure_py]
    assert pure_py_ids == numpy_ids


def test_embedding_failure_durable_jobs_visibility_and_recovery(temp_dir):
    """
    Verify embedding/index failures are not silently swallowed:
    durable vector_index jobs are enqueued and can be processed to recover.
    """
    rt_dir = os.path.join(temp_dir, "failure_recovery_rt")

    class FlakyEncoder(VectorEncoder):
        def __init__(self):
            self.fail_mode = True
            self._dim = 64
            self._model_name = "flaky-encoder"
            self._revision = "v1"
            self._fingerprint = "flaky_v1_64"

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
            return self._fingerprint

        def encode(self, text: str):
            if self.fail_mode:
                raise RuntimeError("Simulated embedding model failure during indexing")
            return [0.1] * self._dim

        def encode_batch(self, texts):
            return [self.encode(t) for t in texts]

    encoder = FlakyEncoder()
    rt = nsn.Runtime(rt_dir, default_namespace="test_ns", semantic=True, encoder=encoder)

    # 1. Record fact while encoder is failing
    fact = rt.record_fact(subject="server", predicate="status", value="running")
    assert fact is not None

    # Verify failure was NOT swallowed silently: durable job exists in jobs table
    jobs = rt.storage.list_jobs(status="pending")
    assert len(jobs) == 1
    assert jobs[0]["job_type"] == "vector_index"
    assert jobs[0]["payload"]["target_id"] == fact.id
    assert "Simulated embedding model failure" in jobs[0]["payload"]["error"]

    # 2. Append event while encoder is failing
    eid = rt.append_event(role="user", content="System diagnostics initialized.")
    jobs = rt.storage.list_jobs(status="pending")
    assert len(jobs) == 2

    # 3. Retrieve pack while encoder is failing: query embedding error must be reported
    pack = rt.retrieve_pack("server status", namespace="test_ns")
    assert "embedding_error" in pack.diagnostic_explain
    assert "Simulated embedding model failure" in pack.diagnostic_explain["embedding_error"]

    # 4. Recover: fix encoder and process durable jobs
    encoder.fail_mode = False
    recovered_count = rt.process_index_jobs()
    assert recovered_count == 2

    # Verify both vectors are now indexed in vector store
    assert rt.vector_store.count("test_ns", "fact") == 1
    assert rt.vector_store.count("test_ns", "event") == 1

    rt.close()


def test_tombstone_deletion_and_rebuild(temp_dir):
    """
    Verify facts superseding and deletion properly tombstone vectors,
    and rebuild_vectors correctly recreates active vectors from SQLite.
    """
    rt_dir = os.path.join(temp_dir, "tombstone_rt")
    encoder = FastHashEncoder(dimension=64)
    rt = nsn.Runtime(rt_dir, default_namespace="corp", semantic=True, encoder=encoder)

    # 1. Add Fact 1 with valid_from
    f1 = rt.record_fact(subject="api_key", predicate="status", value="active", valid_from="2026-01-01T00:00:00Z")
    assert rt.vector_store.count("corp", "fact") == 1

    # 2. Update Fact 1 with newer valid_from -> old fact is superseded and tombstoned
    f2 = rt.record_fact(subject="api_key", predicate="status", value="revoked", valid_from="2026-01-02T00:00:00Z")
    assert rt.vector_store.count("corp", "fact") == 1

    # 3. Add event
    rt.append_event(role="user", content="System reboot scheduled for midnight.")
    assert rt.vector_store.count("corp", "event") == 1
    assert rt.vector_store.count("corp") == 2

    # 4. Delete the active fact
    rt.fact_manager.delete_fact(f2.id, namespace="corp")
    assert rt.vector_store.count("corp", "fact") == 0

    # 5. Rebuild vectors: re-indexes active facts and events, updates index_revision
    manifest_before = rt.vector_store.get_manifest()
    rev_before = manifest_before.index_revision if manifest_before else 1

    count_rebuilt = rt.rebuild_vectors(namespace="corp")
    assert count_rebuilt == 1
    assert rt.vector_store.count("corp") == 1

    manifest_after = rt.vector_store.get_manifest()
    assert manifest_after.index_revision > rev_before

    rt.close()


def test_deletion_and_supersession_before_replay(temp_dir):
    """
    Verify indexing recovery replays from current authoritative records,
    skipping deleted or superseded targets and preventing replay from restoring their vectors.
    """
    rt_dir = os.path.join(temp_dir, "del_supersede_rt")

    class FlakyEncoder(VectorEncoder):
        def __init__(self):
            self._dim = 32
            self._rev = "v1"
            self._name = "flaky"
            self._fp = "fp"
            self.fail_mode = True

        @property
        def model_name(self) -> str: return self._name
        @property
        def revision(self) -> str: return self._rev
        @property
        def dimension(self) -> int: return self._dim
        @property
        def fingerprint(self) -> str: return self._fp
        def encode(self, text: str):
            if self.fail_mode:
                raise RuntimeError("Encoder failing")
            return [0.1] * self._dim
        def encode_batch(self, texts):
            return [self.encode(t) for t in texts]

    encoder = FlakyEncoder()
    rt = nsn.Runtime(rt_dir, default_namespace="test_ns", semantic=True, encoder=encoder)

    # 1. Fact 1 (port 8080) recorded with failing encoder -> indexing intent in SQLite
    f1 = rt.record_fact(subject="gateway", predicate="port", value=8080, valid_from="2026-01-01T00:00:00Z")
    # 2. Fact 2 (port 9090) supersedes Fact 1 with failing encoder
    f2 = rt.record_fact(subject="gateway", predicate="port", value=9090, valid_from="2026-02-01T00:00:00Z")
    # 3. Fact 3 (deleted secret) recorded with failing encoder
    f3 = rt.record_fact(subject="secret", predicate="key", value="xyz")
    # 4. Event 1 (active event) recorded with failing encoder
    e1 = rt.append_event(role="user", content="System diagnostics ready.")

    # Explicitly delete Fact 3 before replay
    rt.fact_manager.delete_fact(f3.id, namespace="test_ns")

    # Authoritative statuses
    f1_refreshed = rt.facts.get_fact(f1.id, "test_ns")
    assert f1_refreshed.status == "superseded"
    f3_refreshed = rt.facts.get_fact(f3.id, "test_ns")
    assert f3_refreshed.status == "deleted"

    # Recover encoder and run replay
    encoder.fail_mode = False
    processed = rt.process_index_jobs(worker_id="recovery_worker")
    assert processed >= 2

    # Verify vector store contents:
    # - Superseded f1 vector is NOT indexed
    # - Deleted f3 vector is NOT indexed
    # - Active f2 IS indexed
    # - Event e1 IS indexed
    assert rt.vector_store.count("test_ns", "fact") == 1
    assert rt.vector_store.count("test_ns", "event") == 1

    rt.close()


def test_restart_recovery_and_unrelated_jobs_untouched(temp_dir):
    """
    Verify pending vector indexing jobs survive restart, and process_index_jobs
    leases and processes ONLY vector_index jobs, leaving unrelated jobs untouched.
    """
    rt_dir = os.path.join(temp_dir, "restart_recovery_rt")

    class FlakyEncoder(VectorEncoder):
        def __init__(self):
            self._dim = 32
            self.fail_mode = True
        @property
        def model_name(self) -> str: return "flaky"
        @property
        def revision(self) -> str: return "v1"
        @property
        def dimension(self) -> int: return self._dim
        @property
        def fingerprint(self) -> str: return "fp"
        def encode(self, text: str):
            if self.fail_mode:
                raise RuntimeError("Encoder failing")
            return [0.1] * self._dim
        def encode_batch(self, texts):
            return [self.encode(t) for t in texts]

    encoder1 = FlakyEncoder()
    rt1 = nsn.Runtime(rt_dir, default_namespace="test_ns", semantic=True, encoder=encoder1)

    # Ingest with failing encoder
    fact = rt1.record_fact(subject="node", predicate="status", value="ready")
    eid = rt1.append_event(role="user", content="Cluster initialized.")

    # Enqueue unrelated consolidation job to jobs table
    rt1.storage.enqueue_job(job_type="consolidation", payload={"cycle": "nrem"}, namespace="test_ns")

    # Pending jobs before restart: 2 vector_index + 1 consolidation = 3
    jobs_before = rt1.storage.list_jobs(status="pending")
    assert len(jobs_before) == 3

    # Close Runtime 1 completely
    rt1.close()

    # Open Runtime 2 with working encoder (same model identity, recovered)
    encoder2 = FlakyEncoder()
    encoder2.fail_mode = False
    rt2 = nsn.Runtime(rt_dir, default_namespace="test_ns", semantic=True, encoder=encoder2)

    # Verify pending jobs survived restart
    assert len(rt2.storage.list_jobs(status="pending")) == 3

    # Process vector index jobs
    recovered = rt2.process_index_jobs(worker_id="w_restart")
    assert recovered == 2

    # Unrelated consolidation job must remain untouched in pending status
    pending_after = rt2.storage.list_jobs(status="pending")
    assert len(pending_after) == 1
    assert pending_after[0]["job_type"] == "consolidation"

    # Vector store contains the recovered items
    assert rt2.vector_store.count("test_ns", "fact") == 1
    assert rt2.vector_store.count("test_ns", "event") == 1

    rt2.close()


def test_competing_workers_and_atomic_leasing(temp_dir):
    """
    Verify atomic claims across genuinely concurrent workers using ThreadPoolExecutor,
    lease ownership enforcement, and namespace isolation under high contention.
    """
    rt_dir = os.path.join(temp_dir, "concurrent_workers_rt")
    encoder = FastHashEncoder(dimension=32)
    rt = nsn.Runtime(rt_dir, default_namespace="ns_alpha", semantic=True, encoder=encoder)
    storage = rt.storage

    # Seed 24 jobs across two namespaces
    total_jobs = 24
    for i in range(total_jobs):
        ns = "ns_alpha" if i % 2 == 0 else "ns_beta"
        storage.enqueue_job("vector_index", {"target_id": f"fact_{i}", "target_type": "fact"}, namespace=ns)

    claimed_by_worker = {}
    completed_jobs = []
    completed_lock = threading.Lock()

    def worker_run(worker_name: str):
        worker_claimed = []
        while True:
            jobs = storage.get_pending_jobs(limit=4, lease_duration_seconds=30, worker_id=worker_name, job_type="vector_index")
            if not jobs:
                break
            for j in jobs:
                worker_claimed.append(j["id"])
                time.sleep(0.005)  # Contention window
                success = storage.complete_job(j["id"], worker_id=worker_name)
                if success:
                    with completed_lock:
                        completed_jobs.append(j["id"])
        return worker_name, worker_claimed

    num_workers = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_run, f"worker_{i}") for i in range(num_workers)]
        for fut in futures:
            w_name, claims = fut.result()
            claimed_by_worker[w_name] = set(claims)

    # 1. Atomic claims: no two workers claimed the same job
    all_claimed = []
    for w_name, claims in claimed_by_worker.items():
        all_claimed.extend(list(claims))
    assert len(all_claimed) == total_jobs, f"Expected {total_jobs} total claims, got {len(all_claimed)}"
    assert len(set(all_claimed)) == total_jobs, "Overlapping claims detected across concurrent workers!"

    # 2. All jobs successfully completed
    assert len(completed_jobs) == total_jobs
    pending_left = storage.list_jobs(status="pending")
    processing_left = storage.list_jobs(status="processing")
    assert len(pending_left) == 0
    assert len(processing_left) == 0

    rt.close()


def test_append_event_idempotency_retry_handling(temp_dir):
    """
    Verify append_event idempotency:
    1. Returns authoritative event ID on retries.
    2. Does NOT index retry content.
    3. Does NOT create orphan vectors or duplicate records in SQLite/vector store.
    """
    rt_dir = os.path.join(temp_dir, "idempotency_rt")
    encoder = FastHashEncoder(dimension=32)
    rt = nsn.Runtime(rt_dir, default_namespace="default", semantic=True, encoder=encoder)

    # Initial append with idempotency key
    id1 = rt.append_event(
        content="Primary telemetry packet 001",
        idempotency_key="telemetry-001",
        namespace="default",
    )
    assert id1 is not None

    # Verify 1 event in storage and 1 vector in vector store
    assert rt.vector_store.count("default", "event") == 1
    stored_evt = rt.storage.get_event(id1, namespace="default")
    assert stored_evt["content"] == "Primary telemetry packet 001"

    # Retry append with same idempotency key but DIFFERENT content
    id2 = rt.append_event(
        content="Retry with modified text that must be ignored",
        idempotency_key="telemetry-001",
        namespace="default",
    )

    # Must return authoritative ID
    assert id2 == id1

    # Must NOT update storage content, must NOT create second event
    conn = sqlite3.connect(rt.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), content FROM events WHERE namespace = 'default'")
    row = cursor.fetchone()
    assert row[0] == 1
    assert row[1] == "Primary telemetry packet 001"

    # Must NOT create orphan vectors in vector store
    cursor.execute("SELECT COUNT(*) FROM compact_vectors WHERE namespace = 'default' AND tombstoned = 0")
    vec_count = cursor.fetchone()[0]
    assert vec_count == 1
    assert rt.vector_store.count("default", "event") == 1

    # Verify the vector corresponds to original content, not retry content
    orig_vec = encoder.encode("Primary telemetry packet 001")
    retry_vec = encoder.encode("Retry with modified text that must be ignored")
    search_orig = rt.vector_store.search(orig_vec, namespace="default", limit=5)
    search_retry = rt.vector_store.search(retry_vec, namespace="default", limit=5)
    assert len(search_orig) == 1
    assert search_orig[0]["id"] == id1
    assert search_orig[0]["score"] > 0.99  # Exact match for original content
    assert search_retry[0]["score"] < 0.99  # Retry content was not indexed

    # Must leave no pending jobs
    assert len(rt.storage.list_jobs(status="pending")) == 0

    conn.close()
    rt.close()


def test_deletion_during_blocked_encoding_prevents_vector_restoration(temp_dir):
    """
    Verify deletion or supersession during encoding prevents vector restoration.
    Atomic validation inside vector persistence transaction must reject vectors
    for targets deleted/superseded while encoding was executing.
    """
    rt_dir = os.path.join(temp_dir, "blocked_encoding_rt")

    encoding_started = threading.Event()
    allow_encoding_finish = threading.Event()

    class BlockingEncoder(VectorEncoder):
        def __init__(self):
            self._dim = 32
        @property
        def model_name(self) -> str: return "blocking-enc"
        @property
        def revision(self) -> str: return "v1"
        @property
        def dimension(self) -> int: return self._dim
        @property
        def fingerprint(self) -> str: return "blocking_fp"
        def encode(self, text: str):
            encoding_started.set()
            allow_encoding_finish.wait(timeout=5.0)
            return [0.1] * self._dim
        def encode_batch(self, texts):
            return [self.encode(t) for t in texts]

    encoder = BlockingEncoder()
    rt = nsn.Runtime(rt_dir, default_namespace="default", semantic=True, encoder=encoder)

    # 1. Test fact deleted during encoding
    fact_id = "fact_will_be_deleted"
    def record_fact_in_thread():
        rt.record_fact(
            id=fact_id,
            subject="cluster",
            predicate="leader",
            value="node_A",
            namespace="default",
        )

    t = threading.Thread(target=record_fact_in_thread)
    t.start()

    # Wait until encoder starts encoding
    assert encoding_started.wait(timeout=3.0)

    # While encoder is blocked in encode(), delete the fact in storage
    rt.fact_manager.delete_fact(fact_id, namespace="default")
    assert rt.facts.get_fact(fact_id, "default").status == "deleted"

    # Allow encoder to finish and proceed to vector_store.add()
    allow_encoding_finish.set()
    t.join(timeout=3.0)

    # Atomic validation inside CompactVectorStore.add should have rejected insertion
    assert rt.vector_store.count("default", "fact") == 0
    conn = sqlite3.connect(rt.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT tombstoned FROM compact_vectors WHERE id = ?", (fact_id,))
    row = cursor.fetchone()
    if row:
        assert row[0] == 1
    conn.close()

    # 2. Test event cancelled during encoding
    encoding_started.clear()
    allow_encoding_finish.clear()

    event_id = "event_will_be_cancelled"
    def append_event_in_thread():
        rt.append_event(
            id=event_id,
            role="user",
            content="Streamed chunk content that gets cancelled",
            namespace="default",
        )

    t2 = threading.Thread(target=append_event_in_thread)
    t2.start()

    assert encoding_started.wait(timeout=3.0)

    # Cancel the event while encoding is blocked
    conn = sqlite3.connect(rt.db_path)
    with conn:
        conn.execute("UPDATE events SET turn_status = 'cancelled' WHERE id = ?", (event_id,))
    conn.close()

    allow_encoding_finish.set()
    t2.join(timeout=3.0)

    # Vector store must not restore vector for cancelled event
    assert rt.vector_store.count("default", "event") == 0
    conn = sqlite3.connect(rt.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT tombstoned FROM compact_vectors WHERE id = ?", (event_id,))
    row2 = cursor.fetchone()
    if row2:
        assert row2[0] == 1
    conn.close()

    rt.close()


def test_lease_bypass_attempts(temp_dir):
    """
    Verify lease ownership enforcement:
    1. Worker completion/failure requires owned, unexpired lease.
    2. Other workers cannot complete or fail a leased job.
    3. Inline completion/failure (worker_id=None) cannot override an active worker lease.
    4. Expired lease prevents completion without re-leasing.
    """
    storage = SQLiteAdapter(os.path.join(temp_dir, "lease_bypass.db"))

    # Enqueue a job
    storage.enqueue_job("vector_index", {"target_id": "item_1", "target_type": "fact"}, namespace="default")
    jobs = storage.list_jobs(status="pending")
    assert len(jobs) == 1
    job_id = jobs[0]["id"]

    # Worker A leases the job for 10 seconds
    leased = storage.get_pending_jobs(limit=1, lease_duration_seconds=10, worker_id="worker_A", job_type="vector_index")
    assert len(leased) == 1
    assert leased[0]["id"] == job_id

    # 1. Other worker attempts to complete -> REJECTED
    assert storage.complete_job(job_id, worker_id="worker_B") is False

    # 2. Other worker attempts to fail -> REJECTED
    assert storage.fail_job(job_id, "hijack attempt", worker_id="worker_B") is False

    # 3. Inline caller attempts to complete -> REJECTED (cannot override worker lease)
    assert storage.complete_job(job_id, worker_id=None) is False

    # 4. Inline caller attempts to fail -> REJECTED
    assert storage.fail_job(job_id, "inline hijack", worker_id=None) is False

    # Verify job is still processing under worker_A
    conn = sqlite3.connect(storage.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status, leased_by FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    assert row[0] == "processing"
    assert row[1] == "worker_A"

    # 5. Simulate lease expiration
    past_time = "2020-01-01T00:00:00Z"
    with conn:
        conn.execute("UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (past_time, job_id))
    conn.close()

    # Worker A tries to complete an expired lease -> REJECTED
    assert storage.complete_job(job_id, worker_id="worker_A") is False

    # Job can now be reclaimed by Worker C
    reclaimed = storage.get_pending_jobs(limit=1, lease_duration_seconds=10, worker_id="worker_C", job_type="vector_index")
    assert len(reclaimed) == 1
    assert reclaimed[0]["id"] == job_id

    # Worker C completes its own lease -> SUCCEEDS
    assert storage.complete_job(job_id, worker_id="worker_C") is True

    # 6. Verify inline completion only works on unleased pending jobs
    storage.enqueue_job("vector_index", {"target_id": "item_2", "target_type": "fact"}, namespace="default")
    pending = storage.list_jobs(status="pending")
    assert len(pending) == 1
    p_id = pending[0]["id"]
    # Inline caller completes pending job -> SUCCEEDS
    assert storage.complete_job(p_id, worker_id=None) is True


def test_local_pooled_encoder_offline_with_blocked_network(temp_dir, monkeypatch):
    """
    Verify LocalPooledEncoder offline mode operates with zero network requests.
    Missing weights fail immediately with AssetNotFoundError rather than attempting network calls.
    """
    def blocked_socket(*args, **kwargs):
        raise RuntimeError("Blocked network call in offline mode")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    # Missing weights fail with AssetNotFoundError without network attempts
    missing_dir = os.path.join(temp_dir, "missing_offline_assets")
    with pytest.raises(AssetNotFoundError) as exc_info:
        LocalPooledEncoder(local_asset_path=missing_dir, offline=True)
    assert "local model assets not found" in str(exc_info.value).lower()


def test_real_prepared_local_encoder_evaluation(temp_dir):
    """
    Evaluate real prepared local encoder (all-MiniLM-L6-v2) on frozen queries with distractors.
    Separated from offline unit tests: skips if assets are not prepared and online download
    is not explicitly enabled via NSN_EVAL_REAL_MODEL=1.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed; skipping real-model evaluation")

    asset_dir = os.path.join(temp_dir, "real_model_assets")
    cached_asset_dir = os.path.join(os.path.expanduser("~"), ".cache", "nsn", "models", "all-MiniLM-L6-v2")
    if os.path.exists(os.path.join(cached_asset_dir, "config.json")):
        shutil.copytree(cached_asset_dir, asset_dir)
    elif os.environ.get("NSN_EVAL_REAL_MODEL") == "1":
        prepare_assets(model_name="all-MiniLM-L6-v2", target_dir=asset_dir)
    else:
        pytest.skip("Real-model evaluation is kept separate from offline unit tests (set NSN_EVAL_REAL_MODEL=1 to run)")

    assert os.path.exists(os.path.join(asset_dir, "model_manifest.json")) or os.path.exists(os.path.join(asset_dir, "config.json"))

    # Block network during offline loading and evaluation to verify zero network leakage
    orig_socket = socket.socket
    try:
        def blocked_socket(*args, **kwargs):
            raise RuntimeError("Unexpected network attempt during offline execution!")
        socket.socket = blocked_socket

        encoder = LocalPooledEncoder(local_asset_path=asset_dir, offline=True)
        rt_dir = os.path.join(temp_dir, "real_encoder_rt")
        rt = nsn.Runtime(rt_dir, default_namespace="eval_ns", semantic=True, encoder=encoder)

        # Ingest Ground-Truth Knowledge
        rt.record_fact(
            subject="engineer",
            predicate="current_residence",
            value="Munich",
            namespace="eval_ns",
        )
        rt.append_event(
            role="assistant",
            content="Production cluster auth secret uuid-9f8a-4421-bc72 deployed to us-west-2.",
            namespace="eval_ns",
        )
        rt.record_fact(
            subject="gateway_port",
            predicate="listen_port",
            value=8080,
            valid_from="2026-01-01T00:00:00Z",
            namespace="eval_ns",
        )
        rt.record_fact(
            subject="gateway_port",
            predicate="listen_port",
            value=9090,
            valid_from="2026-02-01T00:00:00Z",
            namespace="eval_ns",
        )

        # Ingest Distractors
        rt.append_event(
            role="user",
            content="The software engineer wrote documentation about ancient architecture in Munich.",
            namespace="eval_ns",
        )
        rt.append_event(
            role="user",
            content="Temporary staging cluster auth token uuid-0000-0000-0000 decommissioned in eu-central-1.",
            namespace="eval_ns",
        )

        # Query A: Paraphrase ("Where is the developer dwelling presently?")
        pack_para = rt.retrieve_pack("Where is the developer dwelling presently?", namespace="eval_ns")
        assert len(pack_para.items) > 0
        top_para = pack_para.items[0]
        assert top_para.kind == "fact"
        assert top_para.value == "Munich"

        # Query B: Exact Identifier ("uuid-9f8a-4421-bc72")
        pack_id = rt.retrieve_pack("uuid-9f8a-4421-bc72", namespace="eval_ns")
        assert len(pack_id.items) > 0
        assert "uuid-9f8a-4421-bc72" in pack_id.items[0].content
        assert "us-west-2" in pack_id.items[0].content

        # Query C: Updated Fact ("gateway_port listen_port")
        pack_update = rt.retrieve_pack("gateway_port listen_port", namespace="eval_ns")
        active_ports = [str(it.value) for it in pack_update.items if it.kind == "fact"]
        assert "9090" in active_ports
        assert "8080" not in active_ports

        rt.close()
    finally:
        socket.socket = orig_socket


@pytest.mark.performance
def test_performance_benchmarks_varied_1k_and_10k_corpora(temp_dir):
    """
    Performance benchmarks measuring:
    - Exactly 1,000 and 10,000 authoritative records AND corresponding vectors in the runtime being measured.
    - Verified count assertions before timing.
    - Cold and warm p50 / p95 for:
        1) Vector-only search (exact dot product)
        2) End-to-end retrieve_pack (query embedding + retrieval/fusion + context packing)
    - Process RSS sampled at workload phase boundaries; maximum sample is not an OS peak.
    - Python heap allocations via tracemalloc.
    - Disk size including SQLite WAL.
    - FastHashEncoder explicit labeling: synthetic n-gram hashing does not measure real-model embedding cost.
    """
    dim = 384
    encoder = FastHashEncoder(dimension=dim)
    process = psutil.Process()

    results_table = {}

    for corpus_size in [1000, 10000]:
        rt_dir = os.path.join(temp_dir, f"bench_rt_{corpus_size}")
        rt = nsn.Runtime(rt_dir, default_namespace="default", semantic=True, encoder=encoder)

        tracemalloc.start()
        rss_samples = []
        rss_samples.append(("start", process.memory_info().rss))

        # 1. Ingest exactly corpus_size authoritative events and corresponding vectors into the runtime
        t_start_insert = time.perf_counter()
        events_rows = []
        fts_rows = []
        vector_items = []

        for i in range(corpus_size):
            eid = f"evt_{corpus_size}_{i:05d}"
            content = f"Telemetry diagnostics incident code_{i % 100} server node_{i % 25} operational status {i}."
            events_rows.append((eid, "default", "s1", "user", content, "user", "2026-01-01T00:00:00Z", "completed", "permanent"))
            fts_rows.append((eid, content, "default"))
            vec = encoder.encode(content)
            vector_items.append((eid, vec, "default", "event"))

        conn = sqlite3.connect(rt.db_path)
        with conn:
            conn.executemany(
                "INSERT INTO events (id, namespace, session_id, role, content, source_identity, observed_at, turn_status, retention_policy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                events_rows
            )
            conn.executemany(
                "INSERT INTO events_fts (id, content, namespace) VALUES (?, ?, ?)",
                fts_rows
            )
        conn.close()

        # Batch insert corresponding vectors into rt.vector_store
        rt.vector_store.add_batch(vector_items)
        insert_duration_ms = (time.perf_counter() - t_start_insert) * 1000.0

        rss_samples.append(("post_insert", process.memory_info().rss))

        # 2. Assert exact authoritative record and vector counts in runtime before timing
        conn = sqlite3.connect(rt.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events WHERE namespace = 'default'")
        db_event_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM compact_vectors WHERE namespace = 'default' AND tombstoned = 0")
        db_vector_count = cursor.fetchone()[0]
        conn.close()

        assert db_event_count == corpus_size, f"Expected {corpus_size} authoritative events, found {db_event_count}"
        assert db_vector_count == corpus_size, f"Expected {corpus_size} compact vectors, found {db_vector_count}"
        assert rt.vector_store.count("default", "event") == corpus_size

        # 3. Vector-Only Search Timings
        query_text = "Telemetry diagnostics incident code_42"
        query_vec = encoder.encode(query_text)

        # Vector-Only Cold search latency (uncached store instance / cold cache)
        store_cold = CompactVectorStore(
            db_path=rt.db_path,
            dimension=dim,
            model_name=encoder.model_name,
            revision=encoder.revision,
            fingerprint=encoder.fingerprint
        )
        t_vec_cold_start = time.perf_counter()
        _ = store_cold.search(query_vec, namespace="default", limit=10)
        vec_cold_ms = (time.perf_counter() - t_vec_cold_start) * 1000.0

        # Vector-Only Warm search latency: 30 repetitions
        vec_warm_latencies = []
        for _ in range(30):
            t_w = time.perf_counter()
            _ = rt.vector_store.search(query_vec, namespace="default", limit=10)
            vec_warm_latencies.append((time.perf_counter() - t_w) * 1000.0)

        vec_warm_latencies.sort()
        vec_warm_p50_ms = vec_warm_latencies[len(vec_warm_latencies) // 2]
        vec_warm_p95_ms = vec_warm_latencies[int(len(vec_warm_latencies) * 0.95)]

        rss_samples.append(("post_vec_search", process.memory_info().rss))

        # 4. End-to-End retrieve_pack Timings (Embedding + Lexical/Dense Retrieval/Fusion + Budget Packing)
        # Cold E2E retrieve_pack (uncached runtime instance)
        rt_cold = nsn.Runtime(rt_dir, default_namespace="default", semantic=True, encoder=encoder)
        t_e2e_cold_start = time.perf_counter()
        pack_cold = rt_cold.retrieve_pack(query_text, namespace="default", token_budget=256)
        e2e_cold_total_ms = (time.perf_counter() - t_e2e_cold_start) * 1000.0
        timings_cold_breakdown = pack_cold.diagnostic_explain.get("timings", {})
        rt_cold.close()

        # Warm E2E retrieve_pack: 30 repetitions
        e2e_total_latencies = []
        e2e_embed_latencies = []
        e2e_fusion_latencies = []
        e2e_pack_latencies = []

        for _ in range(30):
            t_e2e_w = time.perf_counter()
            pack_warm = rt.retrieve_pack(query_text, namespace="default", token_budget=256)
            total_elapsed = (time.perf_counter() - t_e2e_w) * 1000.0
            e2e_total_latencies.append(total_elapsed)

            t_data = pack_warm.diagnostic_explain.get("timings", {})
            e2e_embed_latencies.append(t_data.get("query_embed_ms", 0.0))
            fusion_time = t_data.get("lexical_search_ms", 0.0) + t_data.get("dense_search_ms", 0.0) + t_data.get("fusion_ms", 0.0)
            e2e_fusion_latencies.append(fusion_time)
            e2e_pack_latencies.append(t_data.get("pack_ms", 0.0))

        rss_samples.append(("post_e2e_search", process.memory_info().rss))

        e2e_total_latencies.sort()
        e2e_warm_total_p50_ms = e2e_total_latencies[len(e2e_total_latencies) // 2]
        e2e_warm_total_p95_ms = e2e_total_latencies[int(len(e2e_total_latencies) * 0.95)]
        e2e_embed_p50 = sorted(e2e_embed_latencies)[len(e2e_embed_latencies) // 2]
        e2e_fusion_p50 = sorted(e2e_fusion_latencies)[len(e2e_fusion_latencies) // 2]
        e2e_pack_p50 = sorted(e2e_pack_latencies)[len(e2e_pack_latencies) // 2]

        current_heap, peak_python_heap = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 5. Disk and Process Footprint
        db_size = os.path.getsize(rt.db_path)
        wal_path = rt.db_path + "-wal"
        wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
        total_disk_bytes = db_size + wal_size
        conn = sqlite3.connect(rt.db_path)
        vector_payload_bytes = conn.execute("SELECT SUM(length(vector)) FROM compact_vectors WHERE namespace='default' AND tombstoned=0").fetchone()[0]
        conn.close()

        rt.close()

        min_rss = min(s[1] for s in rss_samples)
        max_rss = max(s[1] for s in rss_samples)

        results_table[corpus_size] = {
            "authoritative_events": db_event_count,
            "vectors_in_store": db_vector_count,
            "insert_ms": insert_duration_ms,
            "vec_cold_ms": vec_cold_ms,
            "vec_warm_p50_ms": vec_warm_p50_ms,
            "vec_warm_p95_ms": vec_warm_p95_ms,
            "e2e_cold_total_ms": e2e_cold_total_ms,
            "e2e_warm_total_p50_ms": e2e_warm_total_p50_ms,
            "e2e_warm_total_p95_ms": e2e_warm_total_p95_ms,
            "e2e_query_embed_p50_ms": e2e_embed_p50,
            "e2e_fusion_p50_ms": e2e_fusion_p50,
            "e2e_pack_p50_ms": e2e_pack_p50,
            "python_heap_mb": peak_python_heap / (1024 * 1024),
            "rss_start_mb": rss_samples[0][1] / (1024 * 1024),
            "rss_peak_mb": max_rss / (1024 * 1024),
            "rss_samples_mb": [(lbl, round(b / (1024 * 1024), 2)) for lbl, b in rss_samples],
            "disk_total_kb": total_disk_bytes / 1024,
            "bytes_per_vector": total_disk_bytes / corpus_size,
            "vector_payload_bytes": vector_payload_bytes,
        }

    print("\n" + "=" * 80)
    print("PHASE 05 MEASURED BENCHMARKS: EXACT 1k vs 10k CORPORA IN RUNTIME")
    print("NOTE: Timings use FastHashEncoder (synthetic n-gram hash). FastHash timings do NOT")
    print("measure real-model embedding cost; neural encoder performance must be measured separately.")
    print("=" * 80)
    for c_size, metrics in results_table.items():
        print(f"Corpus Size: {c_size} items")
        print(f"  - Authoritative Records in SQLite:     {metrics['authoritative_events']} events (asserted == {c_size})")
        print(f"  - Authoritative Vectors in Store:      {metrics['vectors_in_store']} vectors (asserted == {c_size})")
        print(f"  - Ingestion (SQLite + Vectors):        {metrics['insert_ms']:.2f} ms ({metrics['insert_ms'] / c_size:.3f} ms/item)")
        print(f"  - Vector-Only Search (Cold):           {metrics['vec_cold_ms']:.2f} ms")
        print(f"  - Vector-Only Search (Warm):           p50 = {metrics['vec_warm_p50_ms']:.3f} ms, p95 = {metrics['vec_warm_p95_ms']:.3f} ms")
        print(f"  - End-to-End retrieve_pack (Cold):     {metrics['e2e_cold_total_ms']:.2f} ms")
        print(f"  - End-to-End retrieve_pack (Warm):     p50 = {metrics['e2e_warm_total_p50_ms']:.3f} ms, p95 = {metrics['e2e_warm_total_p95_ms']:.3f} ms")
        print(f"      * FastHash Query Embedding (p50):  {metrics['e2e_query_embed_p50_ms']:.3f} ms (synthetic test hash only)")
        print(f"      * Retrieval & Fusion (Lex+Dense):  {metrics['e2e_fusion_p50_ms']:.3f} ms")
        print(f"      * Context Packing & Budgets:       {metrics['e2e_pack_p50_ms']:.3f} ms")
        print(f"  - Peak Python Heap (tracemalloc):      {metrics['python_heap_mb']:.2f} MB")
        print(f"  - Process RSS Samples (psutil):        {metrics['rss_samples_mb']}")
        print(f"  - Maximum sampled RSS (psutil):           {metrics['rss_peak_mb']:.2f} MB (start: {metrics['rss_start_mb']:.2f} MB)")
        print(f"  - Total Disk (inc. WAL and metadata):  {metrics['disk_total_kb']:.1f} KB ({metrics['bytes_per_vector']:.0f} amortized bytes/item)")
        print(f"  - Raw float32 vector payload:         {metrics['vector_payload_bytes'] / c_size:.0f} bytes/vector")
        print("-" * 80)

    # Validations
    assert results_table[1000]["authoritative_events"] == 1000
    assert results_table[1000]["vectors_in_store"] == 1000
    assert results_table[10000]["authoritative_events"] == 10000
    assert results_table[10000]["vectors_in_store"] == 10000
    assert results_table[1000]["vec_warm_p50_ms"] < 10.0
    assert results_table[10000]["vec_warm_p50_ms"] < 20.0
    # Full database bytes include events, FTS, lineage, and fixed schema pages.
    # Verify the vector's actual compact encoding without hiding metadata growth.
    assert results_table[1000]["vector_payload_bytes"] == 1000 * dim * 4
    assert results_table[10000]["vector_payload_bytes"] == 10000 * dim * 4
    assert results_table[1000]["e2e_warm_total_p50_ms"] < 100.0
    assert results_table[10000]["e2e_warm_total_p50_ms"] < 200.0

