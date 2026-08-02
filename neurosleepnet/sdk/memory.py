from neurosleepnet.config.settings import Settings
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.schemas import MemoryRecord
from neurosleepnet.embeddings.local import LocalEmbeddingProvider
from neurosleepnet.storage.local_vector import FAISSVectorStore
from neurosleepnet.perception.schemas import Observation
from neurosleepnet.perception.classifier import MemoryClassifier
from neurosleepnet.perception.detector import DuplicateDetector
from neurosleepnet.perception.importance import ImportanceScorer
from neurosleepnet.trust.engine import TrustEngine
from neurosleepnet.graph.builder import GraphBuilder
from collections import OrderedDict
from dataclasses import dataclass
import json
import datetime

@dataclass
class ObserveResult:
    stored: bool
    memory_id: str = None
    memory_type: str = None
    importance: float = None
    trust_score: float = None
    is_duplicate: bool = False
    reason: str = ""

class Memory:
    """
    Intelligence Orchestrator for NeuroSleepNet.
    """
    def __init__(self, namespace="default", db_path="neurosleepnet.db"):
        self.namespace = namespace
        self.settings = Settings()
        self.storage = SQLiteAdapter(db_path=db_path)
        self.embedder = LocalEmbeddingProvider()
        self.vector_store = FAISSVectorStore(
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
        
        return ObserveResult(
            stored=True,
            memory_id=record.id,
            memory_type=memory_type,
            importance=importance,
            trust_score=trust_profile.final_score,
            is_duplicate=False
        )

    def ingest_batch(self, items: list, source="batch") -> list[ObserveResult]:
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

    def search_graph(self, query: str, limit: int = 5) -> list:
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
                            
        results = list(results_map.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def search_hybrid(self, query: str, limit: int = 5, semantic_weight: float = 1.5, keyword_weight: float = 0.8, graph_weight: float = 1.0):
        semantic_results = self.search(query, limit=limit*2)
        keyword_results = self.search_keyword(query, limit=limit*2)
        graph_results = self.search_graph(query, limit=limit*2)
        
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
        
        final_results = []
        for id_ in sorted_ids[:limit]:
            record = all_records[id_]
            record['hybrid_score'] = rrf_scores[id_]
            final_results.append(record)
            
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

    def timeline(self, limit: int = 10, memory_type: str = None) -> list:
        """Returns chronological list of memories."""
        import sqlite3
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        query = "SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories WHERE namespace = ?"
        params = [self.namespace]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        
        records = []
        for row in cursor.fetchall():
            record_dict = {
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": float(row[4]),
                "trust_score": float(row[5]),
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": int(row[9]),
                "last_accessed_at": row[10]
            }
            records.append({
                "id": record_dict["id"],
                "content": record_dict["content"],
                "created_at": record_dict["created_at"],
                "type": record_dict["memory_type"],
                "importance": record_dict["importance"]
            })
            
        conn.close()
        return records

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

    def surface_relevant(self, context: str, threshold: float = 0.75) -> list:
        results = self.search_hybrid(context, limit=10)
        surfaced = []
        for r in results:
            score = r.get('hybrid_score', 0.0)
            if score >= threshold:
                r['relevance_score'] = score
                surfaced.append(r)
        surfaced.sort(key=lambda x: x['relevance_score'] * float(x.get('importance', 0.0)), reverse=True)
        return [{"content": x['content'], "memory_type": x['memory_type'], "relevance_score": x['relevance_score'], "importance": x['importance']} for x in surfaced]
