import numpy as np
from neurosleepnet.perception.schemas import Observation

PROTOTYPES = {
    "episodic": [
        "This is something that happened to me, a personal event or experience.",
        "The user performed an action or something occurred at a specific time.",
        "This is a log of what happened during this session.",
    ],
    "semantic": [
        "This is a general fact, piece of knowledge, or objective truth about the world.",
        "This is a definition or established concept that is universally true.",
        "This is factual information extracted from a document or knowledge base.",
    ],
    "procedural": [
        "This is a step-by-step guide, instruction, or workflow on how to do something.",
        "This describes a repeatable process, habit, or procedure.",
        "Follow these steps in order to complete the task.",
    ],
}

class MemoryClassifier:
    """
    Classifies an observation into episodic, semantic, or procedural memory.
    """
    def __init__(self, embedder=None):
        self.embedder = embedder
        self._prototype_embeddings = None

    def _build_prototypes(self):
        if not self.embedder:
            return
            
        self._prototype_embeddings = {}
        for mem_type, sentences in PROTOTYPES.items():
            embeddings = self.embedder.embed_batch(sentences)
            sentence_vecs = [np.mean(tok_embs, axis=0) for tok_embs in embeddings]
            mean_vec = np.mean(sentence_vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            self._prototype_embeddings[mem_type] = mean_vec

    def classify(self, observation: Observation) -> str:
        if not self.embedder:
            return self._keyword_fallback(observation)

        if self._prototype_embeddings is None:
            self._build_prototypes()

        input_emb = self.embedder.embed(observation.content)
        input_vec = np.mean(input_emb, axis=0)
        norm = np.linalg.norm(input_vec)
        if norm > 0:
            input_vec = input_vec / norm

        best_type = 'semantic'
        best_score = -1.0
        
        for mem_type, proto_vec in self._prototype_embeddings.items():
            score = np.dot(input_vec, proto_vec)
            if score > best_score:
                best_score = score
                best_type = mem_type
                
        return best_type

    def _keyword_fallback(self, observation: Observation) -> str:
        content_lower = observation.content.lower()
        
        # Procedural: How-to knowledge
        if any(kw in content_lower for kw in ['how to', 'step by step', 'instruction', 'guide']):
            return 'procedural'
            
        # Episodic: Personal experiences or events
        if any(kw in content_lower for kw in ['yesterday', 'today', 'i saw', 'we went', 'happened', 'user clicked']):
            return 'episodic'
            
        # Default to Semantic: Factual knowledge
        return 'semantic'
