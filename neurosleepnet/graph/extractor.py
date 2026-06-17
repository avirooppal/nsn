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

class RelationshipExtractor:
    """
    Extracts relationships between entities using local heuristics.
    Serves as a lightweight alternative to full relation extraction models.
    """
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
        self.relation_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'owns', 'likes', 'loves', 'knows', 'built', 'created']

    def extract(self, text: str):
        entities = self.entity_extractor.extract(text)
        if len(entities) < 2:
            return []
            
        relationships = []
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1 = entities[i]
                e2 = entities[j]
                
                relation = "RELATED_TO"
                
                e1_idx = text.find(e1)
                e2_idx = text.find(e2)
                
                if e1_idx != -1 and e2_idx != -1:
                    # Find text between the two entities
                    if e1_idx < e2_idx:
                        start_idx = e1_idx + len(e1)
                        end_idx = e2_idx
                    else:
                        start_idx = e2_idx + len(e2)
                        end_idx = e1_idx
                        
                    between_text = text[start_idx:end_idx].lower().split()
                    
                    for verb in self.relation_verbs:
                        if verb in between_text:
                            relation = verb.upper()
                            break
                            
                # Determine source and target based on order in text
                if e1_idx < e2_idx:
                    source, target = e1, e2
                else:
                    source, target = e2, e1
                    
                relationships.append({
                    "source": source,
                    "target": target,
                    "relation": relation
                })
                
        return relationships
