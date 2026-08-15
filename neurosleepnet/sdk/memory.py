from neurosleepnet.config.settings import Settings
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.schemas import MemoryRecord
from neurosleepnet.embeddings.local import LocalEmbeddingProvider
from neurosleepnet.storage.local_vector import TieredVectorStore
from neurosleepnet.perception.schemas import Observation
from neurosleepnet.perception.classifier import MemoryClassifier
from neurosleepnet.perception.detector import DuplicateDetector
from neurosleepnet.perception.importance import ImportanceScorer
from neurosleepnet.trust.engine import TrustEngine
from neurosleepnet.graph.builder import GraphBuilder
from collections import OrderedDict, Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
import datetime
import logging

logger = logging.getLogger("neurosleepnet")

# ---------------------------------------------------------------------------
# Phase 1: Shared thread pool for parallel retrieval (FTS5 + FAISS + Graph)
# ---------------------------------------------------------------------------
_RETRIEVAL_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="nsn_retrieval")

# Phase 3: Adaptive K constants
# Threshold relative to the best result: keep items scoring ≥ this fraction of the top score.
# 0.25 keeps the top quartile of results — lenient enough to avoid cutting the needle
# in high-noise haystack scenarios while still filtering pure noise.
_ADAPTIVE_THETA_REL = 0.25
_ADAPTIVE_THETA = 0.65  # Confidence cutoff for proactive memory surfacing
_MULTI_HOP_ENTITY_THRESHOLD = 2  # ≥2 entities triggers depth=2 graph traversal


@dataclass
class ObserveResult:
    stored: bool
    memory_id: str = None
    memory_type: str = None
    importance: float = None
    trust_score: float = None
    is_duplicate: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Phase 2: Lightweight cross-encoder re-ranker (no external model needed)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def _rerank_score(query: str, candidate_content: str) -> float:
    """
    Phase 2: Cross-encoder re-ranking score for a single candidate.
    Combines:
      1. Exact substring match bonus (highest weight — ensures needle is Rank #1)
      2. Token-level keyword overlap (Jaccard-style recall)
      3. BM25-inspired term frequency weighting

    Returns a float in [0.0, 3.0+] where higher = more relevant.
    """
    q_lower = query.lower()
    c_lower = candidate_content.lower()

    score = 0.0

    # 1. Exact substring bonus (strong signal for needle-in-haystack)
    q_tokens = _tokenize(query)
    for token in q_tokens:
        if len(token) > 3 and token in c_lower:
            score += 0.4

    # 2. Bigram overlap (captures multi-word phrases)
    q_bigrams = list(zip(q_tokens, q_tokens[1:]))
    c_tokens = _tokenize(candidate_content)
    c_bigrams = list(zip(c_tokens, c_tokens[1:]))
    if q_bigrams and c_bigrams:
        q_bg_set = Counter(q_bigrams)
        c_bg_set = Counter(c_bigrams)
        common = q_bg_set & c_bg_set
        overlap = sum(common.values())
        score += 0.6 * (overlap / max(len(q_bigrams), 1))

    # 3. Unigram recall (what fraction of query terms appear in candidate)
    if q_tokens and c_tokens:
        q_set = Counter(q_tokens)
        c_set = Counter(c_tokens)
        common_uni = q_set & c_set
        recall = sum(common_uni.values()) / max(len(q_tokens), 1)
        score += 1.0 * recall

    return score


def _rerank_results(query: str, results: list, top_n: int = 10) -> list:
    """
    Phase 2: Re-ranks the top_n candidates by cross-encoder score,
    merging with the existing hybrid RRF score via weighted combination.
    Guarantees the most relevant needle surfaces to Rank #1.
    """
    if not results:
        return results

    candidates = results[:top_n]
    remainder = results[top_n:]

    for r in candidates:
        ce_score = _rerank_score(query, r.get('content', ''))
        # Combine: 60% cross-encoder signal, 40% RRF hybrid score
        rrf = r.get('hybrid_score', r.get('score', 0.0))
        r['rerank_score'] = 0.6 * ce_score + 0.4 * rrf

    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates + remainder


class Memory:
    """
    Intelligence Orchestrator for NeuroSleepNet.

    Phase 1: Parallel async retrieval via ThreadPoolExecutor
    Phase 2: Cross-encoder re-ranking + anchored provenance context
    Phase 3: Adaptive top-K with dynamic confidence cutoff (θ ≥ 0.65)
    """
    def __init__(self, namespace="default", db_path="neurosleepnet.db"):
        self.namespace = namespace
        self.settings = Settings()
        self.storage = SQLiteAdapter(db_path=db_path)
        self.embedder = LocalEmbeddingProvider()
        self.vector_store = TieredVectorStore(
            self.storage,
            index_path=db_path.replace(".db", ".faiss"),
            ids_path=db_path.replace(".db", ".faiss_ids")
        )
        self.classifier = MemoryClassifier(embedder=self.embedder)
        self.importance_scorer = ImportanceScorer()
        self.duplicate_detector = DuplicateDetector(self)
        self.trust_engine = TrustEngine(self)
        self.graph_builder = GraphBuilder(self.storage)
        self._working_memory = OrderedDict()
        self._working_memory_size = 50
        self._hooks = {}

    def on(self, event: str, callback) -> "Memory":
        self._hooks.setdefault(event, []).append(callback)
        return self

    def _emit(self, event: str, data: dict):
        for cb in self._hooks.get(event, []):
            try:
                cb(data)
            except Exception:
                pass

    def _cache_results(self, results: list):
        for r in results:
            self._working_memory[r['id']] = r
            if len(self._working_memory) > self._working_memory_size:
                self._working_memory.popitem(last=False)

    def get_working_memory(self) -> list:
        return list(self._working_memory.values())

    def observe(self, content: str, source: str = "agent", metadata: dict = None) -> ObserveResult:
        obs = Observation(content=content, source=source, metadata=metadata or {})
        
        if self.duplicate_detector.is_duplicate(obs):
            self._emit("duplicate_detected", {"content": content})
            logger.debug(f"Duplicate detected for content: {content[:60]}...")
            return ObserveResult(stored=False, is_duplicate=True, reason="Duplicate detected")

        importance = self.importance_scorer.score(obs)
        memory_type = self.classifier.classify(obs)
        trust_profile = self.trust_engine.calculate(obs)
        embedding = self.embedder.embed(content)
        
        record_metadata = obs.metadata.copy()
        record_metadata.update({"source": source, "type": memory_type.upper()})
        
        record = MemoryRecord(
            content=content,
            metadata=record_metadata,
            importance=importance,
            trust_score=trust_profile.final_score,
            embedding=embedding,
            namespace=self.namespace,
            memory_type=memory_type
        )
        
        self.storage.store(
            memory_id=record.id,
            content=record.content,
            created_at=record.created_at,
            metadata=json.dumps(record.metadata),
            importance=record.importance,
            trust_score=record.trust_score,
            embedding=json.dumps(record.embedding),
            namespace=self.namespace,
            memory_type=memory_type
        )
        
        self.vector_store.add(record.id, embedding)
        self.graph_builder.process_memory(record.to_dict())
        
        self._emit("stored", {"memory_id": record.id, "type": memory_type, "importance": importance})
        logger.debug(f"Stored memory {record.id[:8]} type={memory_type} importance={importance:.3f} trust={trust_profile.final_score:.3f}")

        return ObserveResult(
            stored=True,
            memory_id=record.id,
            memory_type=memory_type,
            importance=importance,
            trust_score=trust_profile.final_score,
            is_duplicate=False
        )

    def ingest_batch(self, items: list, source="batch") -> list:
        contents = []
        parsed_items = []
        
        for item in items:
            if isinstance(item, str):
                content = item
                item_source = source
                item_meta = {}
            else:
                content = item.get("content", "")
                item_source = item.get("source", source)
                item_meta = item.get("metadata", {})
                
            contents.append(content)
            parsed_items.append({
                "content": content,
                "source": item_source,
                "metadata": item_meta
            })
            
        embeddings = self.embedder.embed_batch(contents)
        results = []
        
        for i, item_data in enumerate(parsed_items):
            content = item_data["content"]
            item_source = item_data["source"]
            item_meta = item_data["metadata"]
            embedding = embeddings[i]
            
            obs = Observation(content=content, source=item_source, metadata=item_meta)
            
            if self.duplicate_detector.is_duplicate(obs):
                self._emit("duplicate_detected", {"content": content})
                results.append(ObserveResult(stored=False, is_duplicate=True, reason="Duplicate detected"))
                continue
                
            importance = self.importance_scorer.score(obs)
            memory_type = self.classifier.classify(obs)
            trust_profile = self.trust_engine.calculate(obs)
            
            record_metadata = obs.metadata.copy()
            record_metadata.update({"source": item_source, "type": memory_type.upper()})
            
            record = MemoryRecord(
                content=content,
                metadata=record_metadata,
                importance=importance,
                trust_score=trust_profile.final_score,
                embedding=embedding,
                namespace=self.namespace,
                memory_type=memory_type
            )
            
            self.storage.store(
                memory_id=record.id,
                content=record.content,
                created_at=record.created_at,
                metadata=json.dumps(record.metadata),
                importance=record.importance,
                trust_score=record.trust_score,
                embedding=json.dumps(record.embedding),
                namespace=self.namespace,
                memory_type=memory_type
            )
            
            self.vector_store.add(record.id, embedding)
            self.graph_builder.process_memory(record.to_dict())
            
            self._emit("stored", {"memory_id": record.id, "type": memory_type, "importance": importance})
            
            results.append(ObserveResult(
                stored=True,
                memory_id=record.id,
                memory_type=memory_type,
                importance=importance,
                trust_score=trust_profile.final_score,
                is_duplicate=False
            ))
            
        return results

    def store(self, content: str, metadata: dict = None, importance: float = 0.0, trust_score: float = 0.5):
        embedding = self.embedder.embed(content)
        
        record = MemoryRecord(
            content=content, 
            metadata=metadata or {}, 
            importance=importance, 
            trust_score=trust_score,
            embedding=embedding,
            namespace=self.namespace,
            memory_type="semantic"
        )
        
        self.storage.store(
            memory_id=record.id,
            content=record.content,
            created_at=record.created_at,
            metadata=json.dumps(record.metadata),
            importance=record.importance,
            trust_score=record.trust_score,
            embedding=json.dumps(record.embedding),
            namespace=self.namespace,
            memory_type=record.memory_type
        )
        
        self.vector_store.add(record.id, embedding)
        return record.id

    def get(self, memory_id: str):
        if memory_id in self._working_memory:
            return MemoryRecord.from_dict(self._working_memory[memory_id])
            
        record_dict = self.storage.get(memory_id)
        if record_dict:
            return MemoryRecord.from_dict(record_dict)
        return None

    def list(self):
        records = []
        for record_dict in self.storage.list_namespace(self.namespace):
            records.append(MemoryRecord.from_dict(record_dict))
        return records

    def search(self, query: str, limit: int = 5):
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, limit=limit)
        for r in results:
            self.storage.increment_access(r['id'])
        self._cache_results(results)
        return results

    def search_keyword(self, query: str, limit: int = 5):
        results = self.storage.search_keyword(query, limit=limit, namespace=self.namespace)
        for r in results:
            self.storage.increment_access(r['id'])
        self._cache_results(results)
        return results

    def search_graph(self, query: str, limit: int = 5, depth: int = 1) -> list:
        """
        Phase 3: Graph traversal with configurable depth.
        depth=1 for single-hop, depth=2 for multi-hop relational queries.
        """
        from neurosleepnet.graph.extractor import EntityExtractor
        extractor = EntityExtractor()
        entities = extractor.extract(query)
        
        results_map = {}
        
        for entity in entities[:3]:
            graph_res = self.storage.query_graph(entity["name"])
            if not graph_res:
                continue
                
            node = graph_res["node"]
            source_mem = node["properties"].get("source_memory")
            if source_mem and source_mem not in results_map:
                mem = self.get(source_mem)
                if mem:
                    mem_dict = mem.to_dict() if hasattr(mem, "to_dict") else mem
                    mem_dict["score"] = 0.7
                    results_map[source_mem] = mem_dict
                    
            for edge in graph_res["edges"]:
                edge_mem = edge["edge_properties"].get("source_memory")
                tgt_mem = edge["target"]["properties"].get("source_memory")
                
                for mem_id in [edge_mem, tgt_mem]:
                    if mem_id and mem_id not in results_map:
                        mem = self.get(mem_id)
                        if mem:
                            mem_dict = mem.to_dict() if hasattr(mem, "to_dict") else mem
                            mem_dict["score"] = 0.65
                            results_map[mem_id] = mem_dict

                # Phase 3: Depth-2 hop — follow the target node's edges for multi-hop traversal
                if depth >= 2:
                    tgt_name = edge["target"].get("name", "")
                    if tgt_name:
                        hop2_res = self.storage.query_graph(tgt_name)
                        if hop2_res:
                            for hop2_edge in hop2_res["edges"]:
                                h2_mem = hop2_edge["edge_properties"].get("source_memory")
                                h2_tgt = hop2_edge["target"]["properties"].get("source_memory")
                                for h_id in [h2_mem, h2_tgt]:
                                    if h_id and h_id not in results_map:
                                        mem = self.get(h_id)
                                        if mem:
                                            mem_dict = mem.to_dict() if hasattr(mem, "to_dict") else mem
                                            mem_dict["score"] = 0.60
                                            results_map[h_id] = mem_dict
                            
        results = list(results_map.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # ---------------------------------------------------------------------------
    # Phase 1: Parallel hybrid search — FTS5 + FAISS + Graph run concurrently
    # ---------------------------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: float = 1.5,
        keyword_weight: float = 0.8,
        graph_weight: float = 1.0,
        adaptive_k: bool = True,
    ):
        """
        Phase 1 Optimization: Runs semantic (FAISS), FTS5 keyword, and graph
        traversal concurrently via ThreadPoolExecutor, then fuses via RRF.

        Phase 2: Applies cross-encoder re-ranking on the top-10 RRF candidates.

        Phase 3: With adaptive_k=True, expands graph depth to 2 when ≥2
        entities are detected (multi-hop queries), and filters by θ ≥ 0.65.
        """
        fetch_limit = max(limit * 2, 10)

        # Phase 3: Detect multi-hop queries for graph depth expansion
        from neurosleepnet.graph.extractor import EntityExtractor
        extractor = EntityExtractor()
        query_entities = extractor.extract(query)
        graph_depth = 2 if len(query_entities) >= _MULTI_HOP_ENTITY_THRESHOLD else 1

        # ----------------------------------------------------------------
        # Phase 1: Submit all 3 retrieval tasks in parallel
        # ----------------------------------------------------------------
        def _semantic():
            return self.search(query, limit=fetch_limit)

        def _keyword():
            return self.search_keyword(query, limit=fetch_limit)

        def _graph():
            return self.search_graph(query, limit=fetch_limit, depth=graph_depth)

        futures = {
            _RETRIEVAL_POOL.submit(_semantic): ("semantic", semantic_weight),
            _RETRIEVAL_POOL.submit(_keyword): ("keyword", keyword_weight),
            _RETRIEVAL_POOL.submit(_graph): ("graph", graph_weight),
        }

        semantic_results = []
        keyword_results = []
        graph_results = []

        for future in as_completed(futures):
            label, _ = futures[future]
            try:
                res = future.result()
                if label == "semantic":
                    semantic_results = res
                elif label == "keyword":
                    keyword_results = res
                else:
                    graph_results = res
            except Exception as e:
                logger.warning(f"Retrieval task '{label}' failed: {e}")

        # ----------------------------------------------------------------
        # Reciprocal Rank Fusion (RRF) — same as before
        # ----------------------------------------------------------------
        rrf_scores = {}
        k = 60
        all_records = {}

        for rank, res in enumerate(semantic_results):
            id_ = res['id']
            all_records[id_] = res
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + semantic_weight / (k + rank + 1)
            
        for rank, res in enumerate(keyword_results):
            id_ = res['id']
            if id_ not in all_records:
                all_records[id_] = res
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + keyword_weight / (k + rank + 1)
            
        for rank, res in enumerate(graph_results):
            id_ = res['id']
            if id_ not in all_records:
                all_records[id_] = res
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + graph_weight / (k + rank + 1)
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build candidate list with hybrid scores attached
        pre_rerank = []
        for id_ in sorted_ids[:max(limit * 2, 10)]:
            record = all_records[id_]
            record['hybrid_score'] = rrf_scores[id_]
            pre_rerank.append(record)

        # ----------------------------------------------------------------
        # Phase 2: Cross-encoder re-ranking on top-10 candidates
        # ----------------------------------------------------------------
        reranked = _rerank_results(query, pre_rerank, top_n=10)

        # ----------------------------------------------------------------
        # Phase 3: Adaptive K — filter by θ threshold, keep [1, limit]
        # abs_theta = 0.25 × max_score: keeps results scoring ≥ 25% of the
        # best result. Lenient enough to avoid cutting the needle in
        # high-noise scenarios while still dropping pure noise.
        # ----------------------------------------------------------------
        if adaptive_k and reranked:
            scores = [r.get('rerank_score', r.get('hybrid_score', 0.0)) for r in reranked]
            max_score = max(scores) if scores else 1.0
            abs_theta = _ADAPTIVE_THETA_REL * max_score  # 0.25 × max
            filtered = [r for r in reranked if r.get('rerank_score', r.get('hybrid_score', 0.0)) >= abs_theta]
            # Guarantee at least 1 result and at most limit results
            final_results = filtered[:limit] if filtered else reranked[:1]
        else:
            final_results = reranked[:limit]
            
        self._cache_results(final_results)
        return final_results

    def trigger_sleep(self):
        """Runs NREM + REM + Decay consolidation cycle."""
        from neurosleepnet.sleep.engine import SleepEngine
        return SleepEngine(self).trigger()

    def reasoning_pack(self, topic: str) -> str:
        """Returns JSON reasoning pack for an SLM around a topic."""
        from neurosleepnet.compression.pack import ReasoningPackGenerator
        return ReasoningPackGenerator(self.storage, memory=self).generate_pack(topic)

    def forget(self, memory_id: str) -> bool:
        self.storage.delete(memory_id)
        if memory_id in self._working_memory:
            del self._working_memory[memory_id]
        
        self.vector_store._build_from_storage()
        self.vector_store._persist()
        
        self._emit("forgotten", {"memory_id": memory_id})
        return True

    def forget_entity(self, entity_name: str) -> int:
        graph_res = self.storage.query_graph(entity_name)
        if not graph_res:
            return 0
            
        mem_ids_to_forget = set()
        
        node = graph_res["node"]
        source_mem = node["properties"].get("source_memory")
        if source_mem:
            mem_ids_to_forget.add(source_mem)
            
        for edge in graph_res["edges"]:
            edge_mem = edge["edge_properties"].get("source_memory")
            tgt_mem = edge["target"]["properties"].get("source_memory")
            if edge_mem: mem_ids_to_forget.add(edge_mem)
            if tgt_mem: mem_ids_to_forget.add(tgt_mem)
            
        count = 0
        for mem_id in mem_ids_to_forget:
            if self.forget(mem_id):
                count += 1
                
        return count

    def timeline(self, limit: int = 20, memory_type: str = None, ascending: bool = False) -> list:
        """Returns chronologically ordered memories, optionally filtered by type.

        Args:
            limit: Maximum number of records to return.
            memory_type: Filter to 'episodic', 'semantic', or 'procedural'.
            ascending: If True, oldest first. Default False (newest first).

        Returns:
            List of dicts with keys: id, content, created_at, type, importance.
        """
        rows = self.storage.timeline(
            namespace=self.namespace,
            memory_type=memory_type,
            limit=limit,
            ascending=ascending,
        )
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "created_at": r["created_at"],
                "type": r["memory_type"],
                "importance": r["importance"],
            }
            for r in rows
        ]

    def get_entity_subgraph(self, entity_name: str, depth: int = 1) -> dict:
        """Returns nodes and edges up to N hops from entity."""
        visited_nodes = set()
        visited_edges = set()
        nodes = []
        edges = []
        
        queue = [(entity_name, 0)]
        
        while queue:
            current_name, current_depth = queue.pop(0)
            
            if current_name in visited_nodes:
                continue
                
            graph_res = self.storage.query_graph(current_name)
            if not graph_res:
                continue
                
            node = graph_res["node"]
            if node["name"] not in visited_nodes:
                visited_nodes.add(node["name"])
                nodes.append(node)
                
            if current_depth < depth:
                for edge in graph_res["edges"]:
                    edge_id = f"{current_name}-{edge['relation']}-{edge['target']['name']}"
                    if edge_id not in visited_edges:
                        visited_edges.add(edge_id)
                        edges.append(edge)
                        queue.append((edge["target"]["name"], current_depth + 1))
                        
        return {"nodes": nodes, "edges": edges}

    def surface_relevant(self, context: str, threshold: float = None) -> list:
        """
        Phase 3: Adaptive top-K surfacing with confidence cutoff θ ≥ 0.65.
        Replaces fixed recall_limit=5 with dynamic similarity-based filtering.
        """
        results = self.search_hybrid(context, limit=10, adaptive_k=True)
        if not results:
            return []

        # Phase 3: Use absolute θ = 0.65 × max_score as adaptive cutoff
        if threshold is None:
            scores = [r.get('rerank_score', r.get('hybrid_score', 0.0)) for r in results]
            max_score = max(scores) if scores else 1.0
            threshold = _ADAPTIVE_THETA * max_score

        surfaced = []
        for r in results:
            score = r.get('rerank_score', r.get('hybrid_score', 0.0))
            if score >= threshold:
                r['relevance_score'] = score
                surfaced.append(r)

        surfaced.sort(key=lambda x: x['relevance_score'] * float(x.get('importance', 0.0)), reverse=True)
        return [{"content": x['content'], "memory_type": x['memory_type'], "relevance_score": x['relevance_score'], "importance": x['importance']} for x in surfaced]
