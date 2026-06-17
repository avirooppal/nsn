import re

class EntityExtractor:
    """
    Extracts entities from text using local heuristics.
    Serves as a lightweight alternative to full NER models.
    """
    def extract(self, text: str):
        # Split by punctuation to identify sentence boundaries roughly
        sentences = re.split(r'[.!?]', text)
        entities = set()
        
        stop_words = {'The', 'A', 'An', 'It', 'He', 'She', 'They', 'This', 'That'}
        
        for sentence in sentences:
            words = sentence.strip().split()
            if not words:
                continue
                
            current_entity = []
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^a-zA-Z]', '', word)
                if clean_word and clean_word[0].isupper():
                    current_entity.append(clean_word)
                else:
                    if current_entity:
                        entity_str = " ".join(current_entity)
                        # Skip if it's just the first word of the sentence and it's a stop word
                        if not (i == len(current_entity) and entity_str in stop_words):
                            entities.add(entity_str)
                        current_entity = []
            
            if current_entity:
                entity_str = " ".join(current_entity)
                if not (len(words) == len(current_entity) and entity_str in stop_words):
                    entities.add(entity_str)
                    
        return list(entities)
