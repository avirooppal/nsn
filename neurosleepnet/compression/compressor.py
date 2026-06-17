import re

class ContextCompressor:
    """
    Compresses a list of memories to fit within an SLM context window.
    Uses extractive summarization heuristics.
    """
    def __init__(self, max_tokens: int = 100):
        self.max_tokens = max_tokens
        
    def compress(self, memories: list, query: str = "") -> str:
        """
        Compresses memory contents prioritizing relevance to the query.
        For MVP, it extracts the most relevant sentence from each memory until max_tokens is reached.
        """
        if not memories:
            return ""
            
        compressed_text = []
        current_tokens = 0
        
        query_words = set(re.findall(r'\w+', query.lower())) if query else set()
        
        def score_memory(mem):
            content = mem.get('content', '')
            score = mem.get('score', 0)
            importance = mem.get('importance', 0)
            
            if query_words:
                words = set(re.findall(r'\w+', content.lower()))
                overlap = len(query_words.intersection(words))
                score += overlap * 0.1
                
            return score + (importance * 0.2)
            
        sorted_mems = sorted(memories, key=score_memory, reverse=True)
        
        for mem in sorted_mems:
            content = mem.get('content', '')
            
            # Simple sentence splitting
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', content) if s.strip()]
            if not sentences:
                continue
                
            best_sentence = sentences[0]
            if query_words and len(sentences) > 1:
                best_score = -1
                for s in sentences:
                    s_words = set(re.findall(r'\w+', s.lower()))
                    overlap = len(query_words.intersection(s_words))
                    if overlap > best_score:
                        best_score = overlap
                        best_sentence = s
                        
            tokens = len(best_sentence.split())
            if current_tokens + tokens > self.max_tokens:
                break
                
            compressed_text.append(best_sentence)
            current_tokens += tokens
            
        return " ".join(compressed_text)
