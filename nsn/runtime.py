"""
NSN Runtime: Central controller for self-hosted memory layer.
Manages scoped SQLite storage, fact manager, context packer, and adapter lifecycles.
"""
import os
import uuid
import json
import sqlite3
import warnings
from datetime import datetime, timezone
from typing import Any, Optional

from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.facts import FactManager
from neurosleepnet.retrieval.pack import ContextPacker, EvidencePack


class Runtime:
    """
    NSN memory runtime instance bound to a local directory.
    """

    def __init__(
        self,
        data_dir: str = "",
        default_namespace: str = "default",
        semantic: bool = False,
        encoder: Optional[Any] = None,
        local_asset_path: Optional[str] = None,
        offline: bool = False,
        graph: bool = False,
        defer_indexing: bool = False,
    ):
        if not data_dir:
            data_dir = "./agent-memory"

        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "memory.db")
        self.default_namespace = default_namespace
        self.semantic = semantic or (encoder is not None)
        self.offline = offline
        self.defer_indexing = defer_indexing

        self.storage = SQLiteAdapter(self.db_path, keep_wal_open=True)

        self.encoder = None
        self.vector_store = None

        if self.semantic:
            if encoder is not None:
                self.encoder = encoder
            else:
                from neurosleepnet.embeddings.pooled import LocalPooledEncoder
                self.encoder = LocalPooledEncoder(local_asset_path=local_asset_path, offline=offline)

            from neurosleepnet.storage.compact_vector import CompactVectorStore
            self.vector_store = CompactVectorStore.from_encoder(
                db_path=self.db_path,
                encoder=self.encoder,
            )

        self.fact_manager = FactManager(
            self.storage,
            vector_store=self.vector_store,
            encoder=self.encoder,
            defer_indexing=defer_indexing,
        )
        self.facts = self.fact_manager
        self.relationships = None
        if graph:
            from neurosleepnet.graph.relationships import RelationshipMemory
            self.relationships = RelationshipMemory(self.storage)
        self.packer = ContextPacker(
            self.storage,
            self.fact_manager,
            vector_store=self.vector_store,
            encoder=self.encoder,
            relationships=self.relationships,
        )
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Runtime is closed")

    def wrap(self, model: Any, namespace: Optional[str] = None, **kwargs) -> Any:
        """
        Wrap a model with this runtime.
        """
        if self._closed:
            raise RuntimeError("Cannot wrap model with a closed NSN Runtime.")
        from nsn.wrapper import wrap
        ns = namespace or self.default_namespace
        return wrap(model, namespace=ns, runtime=self, **kwargs)

    def retrieve_pack(
        self,
        query: str,
        namespace: Optional[str] = None,
        max_tokens: int = 512,
        input_window_tokens: Optional[int] = None,
        existing_tokens: int = 0,
        reserved_output_tokens: int = 0,
        as_of_time: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        session_id: Optional[str] = None,
        budget_breakdown: Optional[dict] = None,
        **kwargs,
    ) -> EvidencePack:
        self._require_open()
        ns = namespace or self.default_namespace
        if "token_budget" in kwargs:
            max_tokens = kwargs.pop("token_budget")
        return self.packer.retrieve_pack(
            query=query,
            namespace=ns,
            max_tokens=max_tokens,
            input_window_tokens=input_window_tokens,
            existing_tokens=existing_tokens,
            reserved_output_tokens=reserved_output_tokens,
            as_of_time=as_of_time,
            tokenizer=tokenizer,
            session_id=session_id,
            budget_breakdown=budget_breakdown,
            **kwargs,
        )

    def append_event(self, namespace: Optional[str] = None, **kwargs) -> str:
        self._require_open()
        ns = namespace or self.default_namespace
        content = kwargs.get("content", "")
        event_id = kwargs.get("id") or str(uuid.uuid4())
        kwargs["id"] = event_id

        indexing_job_id = None
        outbox_jobs = kwargs.pop("outbox_jobs", None) or []
        if self.vector_store is not None and content and content.strip():
            indexing_job_id = str(uuid.uuid4())
            outbox_jobs.append({
                "id": indexing_job_id,
                "job_type": "vector_index",
                "payload": {
                    "target_id": event_id,
                    "target_type": "event",
                },
            })

        actual_id = self.storage.append_event(namespace=ns, outbox_jobs=outbox_jobs, **kwargs)

        is_inserted = getattr(actual_id, "inserted", True)
        if not is_inserted or str(actual_id) != str(event_id):
            return str(actual_id)

        if not self.defer_indexing and self.vector_store is not None and self.encoder is not None and content and content.strip():
            try:
                vec = self.encoder.encode(content)
                self.vector_store.add(str(actual_id), vec, namespace=ns, target_type="event", validate_target=True)
                if indexing_job_id:
                    self.storage.complete_job(indexing_job_id, worker_id=None)
            except Exception as e:
                if indexing_job_id:
                    self.storage.fail_job(indexing_job_id, str(e), worker_id=None)

        return str(actual_id)

    def process_index_jobs(self, worker_id: str = "default", limit: int = 50) -> int:
        """
        Process and recover any queued vector indexing durable jobs.
        Leases only vector_index jobs and replays from authoritative records,
        skipping deleted, superseded, or missing targets.
        """
        self._require_open()
        if self.vector_store is None or self.encoder is None:
            return 0
        jobs = self.storage.get_pending_jobs(
            limit=limit,
            lease_duration_seconds=30,
            worker_id=worker_id,
            job_type="vector_index",
        )
        processed = 0
        for job in jobs:
            payload = job.get("payload", {})
            target_id = payload.get("target_id")
            target_type = payload.get("target_type")
            namespace = job.get("namespace", self.default_namespace)

            try:
                if target_type == "fact":
                    fact_obj = self.facts.get_fact(target_id, namespace=namespace)
                    if fact_obj is None or fact_obj.status in ("deleted", "superseded"):
                        # Target is missing, deleted, or superseded: do not restore vector
                        try:
                            self.vector_store.tombstone(target_id, namespace=namespace)
                        except Exception:
                            pass
                        self.storage.complete_job(job["id"], worker_id=worker_id)
                        processed += 1
                        continue

                    desc = f"{fact_obj.subject} {fact_obj.predicate} is {fact_obj.value}"
                    if fact_obj.unit:
                        desc += f" {fact_obj.unit}"
                    vec = self.encoder.encode(desc)
                    self.vector_store.add(target_id, vec, namespace=namespace, target_type="fact")
                    self.storage.complete_job(job["id"], worker_id=worker_id)
                    processed += 1

                elif target_type in ("event", "summary"):
                    evt = self.storage.get_event(target_id, namespace=namespace)
                    if evt is None or evt.get("turn_status") in ("cancelled", "deleted") or evt.get("status") in ("deleted", "superseded"):
                        # Target is missing or cancelled: do not restore vector
                        try:
                            self.vector_store.tombstone(target_id, namespace=namespace)
                        except Exception:
                            pass
                        self.storage.complete_job(job["id"], worker_id=worker_id)
                        processed += 1
                        continue

                    evt_content = evt.get("content", "")
                    if evt_content and evt_content.strip():
                        vec = self.encoder.encode(evt_content)
                        self.vector_store.add(target_id, vec, namespace=namespace, target_type=target_type)
                    self.storage.complete_job(job["id"], worker_id=worker_id)
                    processed += 1
                else:
                    # Unrecognized target type: complete job to prevent poison loop
                    self.storage.complete_job(job["id"], worker_id=worker_id)
                    processed += 1

            except Exception as exc:
                self.storage.fail_job(job["id"], str(exc), worker_id=worker_id)

        return processed

    def rebuild_vectors(self, namespace: Optional[str] = None) -> int:
        """
        Rebuild compact vectors for active facts and events from SQLite.
        """
        self._require_open()
        if self.vector_store is None or self.encoder is None:
            return 0
        return self.vector_store.rebuild(
            db_path=self.db_path,
            encoder=self.encoder,
            namespace=namespace or self.default_namespace,
        )

    @classmethod
    def prepare_assets(cls, target_dir: Optional[str] = None) -> str:
        """Explicitly prepare local embedding assets."""
        from neurosleepnet.embeddings.pooled import prepare_assets
        return prepare_assets(target_dir=target_dir)

    def record_fact(self, namespace: Optional[str] = None, **kwargs) -> Any:
        self._require_open()
        ns = namespace or self.default_namespace
        return self.fact_manager.record_fact(namespace=ns, **kwargs)

    def record_relation(self, namespace: Optional[str] = None, **kwargs) -> str:
        self._require_open()
        if self.relationships is None:
            raise RuntimeError("Relationship retrieval requires graph=True")
        return self.relationships.record(namespace=namespace or self.default_namespace, **kwargs)

    def extract_relations(self, event_id: str, namespace: Optional[str] = None) -> list:
        self._require_open()
        if self.relationships is None:
            raise RuntimeError("Relationship retrieval requires graph=True")
        return self.relationships.extract(event_id, namespace or self.default_namespace)

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        """
        Verify all queued memory transactions and jobs are flushed.
        """
        if self._closed:
            return True
        return self.storage.flush(timeout_seconds=timeout_seconds)

    def sleep(self, namespace: Optional[str] = None, **kwargs) -> dict:
        """Explicit bounded consolidation; no daemon or automatic model calls."""
        if self._closed:
            raise RuntimeError("Runtime is closed")
        from neurosleepnet.sleep.consolidation import Consolidator
        return Consolidator(self).sleep(namespace=namespace, **kwargs)

    def pin(self, source_id: str, source_type: str = "fact", namespace: Optional[str] = None, pinned: bool = True):
        self._require_open()
        from neurosleepnet.sleep.consolidation import Consolidator
        return Consolidator(self).pin(source_id, source_type, namespace, pinned)

    @property
    def advanced(self):
        from nsn.advanced import AdvancedMemory
        return AdvancedMemory(self)

    def timeline(self, **kwargs):
        return self.advanced.timeline(**kwargs)

    def explain(self, query, **kwargs):
        return self.retrieve_pack(query, **kwargs).to_dict()

    def stats(self, namespace=None):
        return self.advanced.stats(namespace)

    def export_memory(self, namespace=None):
        return self.advanced.export_memory(namespace)

    def import_memory(self, data, namespace=None):
        return self.advanced.import_memory(data, namespace)

    def delete_namespace(self, namespace=None):
        return self.advanced.delete_namespace(namespace)

    def record_procedure(self, **kwargs):
        return self.advanced.record_procedure(**kwargs)

    def record_outcome(self, **kwargs):
        return self.advanced.record_outcome(**kwargs)

    def retrieve_procedures(self, query, **kwargs):
        return self.advanced.retrieve_procedures(query, **kwargs)

    def close(self):
        """
        Idempotently close the runtime.
        """
        if not self._closed:
            self._closed = True
            try:
                self.storage.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return f"<NSN.Runtime dir='{self.data_dir}' default_namespace='{self.default_namespace}' [{status}]>"
