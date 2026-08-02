import re

_SPACY_NLP = None
try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    pass

SPACY_LABEL_MAP = {
    "PERSON": "Person", "ORG": "Organization", "GPE": "Location",
    "LOC": "Location", "PRODUCT": "Product", "WORK_OF_ART": "Concept",
    "EVENT": "Event", "NORP": "Group",
}

class EntityExtractor:
    """
    Extracts entities from text using spaCy NER with a fallback to local heuristics.
    """
    def extract(self, text: str) -> list[dict]:
        if _SPACY_NLP is not None:
            doc = _SPACY_NLP(text)
            entities = []
            seen = set()
            for ent in doc.ents:
                name = ent.text
                if name.lower() not in seen:
                    label = SPACY_LABEL_MAP.get(ent.label_, "Entity")
                    entities.append({"name": name, "label": label})
                    seen.add(name.lower())
            return entities
        return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> list[dict]:
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
                        if not (i == len(current_entity) and entity_str in stop_words):
                            entities.add(entity_str)
                        current_entity = []
            
            if current_entity:
                entity_str = " ".join(current_entity)
                if not (len(words) == len(current_entity) and entity_str in stop_words):
                    entities.add(entity_str)
                    
        return [{"name": e, "label": "Entity"} for e in entities]

class RelationshipExtractor:
    """
    Extracts relationships between entities using local heuristics.
    """
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
        self.relation_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'owns', 'likes', 'loves', 'knows', 'built', 'created']

    def extract(self, text: str) -> list[dict]:
        entities = self.entity_extractor.extract(text)
        if len(entities) < 2:
            return []
            
        relationships = []
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1 = entities[i]
                e2 = entities[j]
                
                relation = "RELATED_TO"
                
                e1_idx = text.find(e1["name"])
                e2_idx = text.find(e2["name"])
                
                if e1_idx != -1 and e2_idx != -1:
                    # Find text between the two entities
                    if e1_idx < e2_idx:
                        start_idx = e1_idx + len(e1["name"])
                        end_idx = e2_idx
                    else:
                        start_idx = e2_idx + len(e2["name"])
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
