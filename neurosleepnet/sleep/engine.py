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
        
        all_memories = self.memory.storage.list()
        
        episodic_mems = []
        for m in all_memories:
            meta = m.get('metadata', {})
            if meta.get('type') == 'EPISODIC' and not meta.get('consolidated'):
                episodic_mems.append(m)
                
        if not episodic_mems:
            logger.info("No new episodic memories to consolidate.")
            return None
            
        aggregated_content = " | ".join([m['content'] for m in episodic_mems])
        semantic_id = str(uuid.uuid4())
        
        self.memory.storage.store(
            memory_id=semantic_id,
            content=f"CONSOLIDATED NREM KNOWLEDGE: {aggregated_content}",
            created_at=datetime.datetime.utcnow().isoformat(),
            metadata=json.dumps({"type": "SEMANTIC", "source": "NREM_CONSOLIDATION", "consolidated_count": len(episodic_mems)}),
            importance=0.8,
            trust_score=0.9
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
                embedding=json.dumps(m.get('embedding', []))
            )
            
        logger.info(f"NREM Complete: Aggregated {len(episodic_mems)} episodic memories.")
        return semantic_id

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
        
        # Simulate waking up
        self.is_sleeping = False
        logger.info("Sleep cycle complete.")
        
        return True
