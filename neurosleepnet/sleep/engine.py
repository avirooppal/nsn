import logging
import json
import uuid
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SleepEngine")

class SleepEngine:
    """
    Coordinates offline memory consolidation.
    """
    def __init__(self, memory):
        self.memory = memory
        self.is_sleeping = False

    def nrem_consolidation(self):
        """
        Phase 1: NREM Consolidation
        Aggregates unconsolidated episodic memories into semantic knowledge.
        """
        logger.info("Starting NREM consolidation phase...")
        
        all_memories = self.memory.storage.list_namespace(self.memory.namespace)
        
        episodic_mems = []
        for m in all_memories:
            meta = m.get('metadata', {})
            if meta.get('type') == 'EPISODIC' and not meta.get('consolidated'):
                episodic_mems.append(m)
                
        if not episodic_mems:
            logger.info("No new episodic memories to consolidate.")
            return None
            
        import re
        from collections import Counter
        from neurosleepnet.compression.compressor import ContextCompressor
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'it', 'this', 'that'}
        
        all_words = []
        for m in episodic_mems:
            words = re.findall(r'\b[a-zA-Z]+\b', m['content'].lower())
            all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
            
        common_words = [w for w, count in Counter(all_words).most_common(5)]
        synthesis_query = " ".join(common_words)
        
        compressed_content = ContextCompressor(max_tokens=150).compress(episodic_mems, query=synthesis_query)
        
        semantic_id = str(uuid.uuid4())
        
        self.memory.storage.store(
            memory_id=semantic_id,
            content=compressed_content,
            created_at=datetime.datetime.utcnow().isoformat(),
            metadata=json.dumps({
                "source": "NREM_CONSOLIDATION", 
                "consolidated_count": len(episodic_mems),
                "source_ids": [m['id'] for m in episodic_mems],
                "type": "SEMANTIC"
            }),
            importance=0.8,
            trust_score=0.9,
            memory_type="semantic",
            namespace=episodic_mems[0].get('namespace', 'default')
        )
        
        # Mark originals as consolidated
        for m in episodic_mems:
            meta = m.get('metadata', {})
            meta['consolidated'] = True
            self.memory.storage.store(
                memory_id=m['id'],
                content=m['content'],
                created_at=m['created_at'],
                metadata=json.dumps(meta),
                importance=m.get('importance', 0.0),
                trust_score=m.get('trust_score', 0.5),
                embedding=json.dumps(m.get('embedding', [])),
                namespace=m.get('namespace', 'default'),
                memory_type=m.get('memory_type', 'episodic')
            )
            
        # Add synthesized memory to FAISS so it is visible to semantic search
        try:
            nrem_embedding = self.memory.embedder.embed(compressed_content)
            self.memory.vector_store.add(semantic_id, nrem_embedding)
        except Exception as e:
            logger.warning(f"NREM: Could not add synthesized memory to vector store: {e}")

        logger.info(f"NREM Complete: Aggregated {len(episodic_mems)} episodic memories.")
        return semantic_id

    def rem_consolidation(self):
        """
        Phase 2: REM Consolidation
        Resolves contradictions and prunes conflicting data.
        """
        logger.info("Starting REM consolidation phase...")
        all_memories = self.memory.storage.list_namespace(self.memory.namespace)
        
        negation_words = {'not', 'never', 'false', 'no', 'cannot', "don't", "doesn't", "isn't", 'dislike', 'hate'}
        
        to_delete = []
        for i in range(len(all_memories)):
            if all_memories[i]['id'] in to_delete: continue
            
            for j in range(i + 1, len(all_memories)):
                if all_memories[j]['id'] in to_delete: continue
                
                m1 = all_memories[i]
                m2 = all_memories[j]
                
                w1 = set(m1['content'].lower().split())
                w2 = set(m2['content'].lower().split())
                
                c1 = w1 - negation_words
                c2 = w2 - negation_words
                
                if len(c1) == 0 or len(c2) == 0:
                    continue
                    
                overlap = len(c1.intersection(c2))
                if overlap / max(len(c1), len(c2)) > 0.6:
                    has_neg_1 = any(w in w1 for w in negation_words)
                    has_neg_2 = any(w in w2 for w in negation_words)
                    
                    if has_neg_1 != has_neg_2:
                        logger.warning(f"Contradiction detected between {m1['id']} and {m2['id']}")
                        t1 = m1.get('trust_score', 0.5)
                        t2 = m2.get('trust_score', 0.5)
                        if t1 >= t2:
                            to_delete.append(m2['id'])
                        else:
                            to_delete.append(m1['id'])
                            
        for mem_id in to_delete:
            self.memory.storage.delete(mem_id)
            logger.info(f"REM Pruning: Deleted contradicted memory {mem_id}")
            
        logger.info(f"REM Complete: Resolved {len(to_delete)} contradictions.")
        return len(to_delete)

    def apply_decay(self, min_importance=0.05):
        logger.info("Starting Decay phase...")
        all_memories = self.memory.storage.list_namespace(self.memory.namespace)
        
        for m in all_memories:
            access_count = m.get('access_count', 0)
            importance = float(m.get('importance', 0.0))
            memory_type = m.get('memory_type', 'semantic')
            meta = m.get('metadata', {})
            
            needs_update = False
            
            if access_count == 0:
                new_importance = max(min_importance, importance * 0.85)
                if new_importance != importance:
                    importance = new_importance
                    needs_update = True
            else:
                new_importance = min(1.0, importance + min(access_count * 0.02, 0.2))
                if new_importance != importance:
                    importance = new_importance
                    needs_update = True
                    
            if memory_type == "episodic" and access_count >= 3:
                memory_type = "semantic"
                meta["promoted_from"] = "episodic"
                needs_update = True
                
            if needs_update:
                self.memory.storage.store(
                    memory_id=m['id'],
                    content=m['content'],
                    created_at=m['created_at'],
                    metadata=json.dumps(meta),
                    importance=importance,
                    trust_score=m.get('trust_score', 0.5),
                    embedding=json.dumps(m.get('embedding', [])),
                    namespace=m.get('namespace', 'default'),
                    memory_type=memory_type
                )
        logger.info("Decay phase complete.")

    def trigger(self):
        """
        Initiates the sleep cycle.
        """
        if self.is_sleeping:
            logger.warning("Sleep cycle is already in progress.")
            return False
            
        self.is_sleeping = True
        logger.info("Initiating sleep cycle. Offline consolidation started...")
        
        # Sleep phases
        self.nrem_consolidation()
        self.rem_consolidation()
        self.apply_decay()
        
        # Simulate waking up
        self.is_sleeping = False
        logger.info("Sleep cycle complete.")
        
        return True
