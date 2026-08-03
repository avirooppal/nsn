import unittest
from neurosleepnet.graph.extractor import EntityExtractor, RelationshipExtractor

class TestRelationshipExtractor(unittest.TestCase):
    def setUp(self):
        self.entity_extractor = EntityExtractor()
        self.rel_extractor = RelationshipExtractor(self.entity_extractor)

    def test_extract_relationships(self):
        text = "Alice knows Bob. Charlie loves David."
        relations = self.rel_extractor.extract(text)
        
        # Check that we got relationships between entities in the same sentence
        # Alice knows Bob -> KNOWS
        # Charlie loves David -> LOVES
        # Depending on order, it might find Alice RELATED_TO Charlie, but let's check for the specific ones
        
        knows_rel = next((r for r in relations if r["source"]["name"] == "Alice" and r["target"]["name"] == "Bob" and r["relation"] == "KNOWS"), None)
        self.assertIsNotNone(knows_rel, "Failed to find Alice KNOWS Bob")
        
        loves_rel = next((r for r in relations if r["source"]["name"] == "Charlie" and r["target"]["name"] == "David" and r["relation"] == "LOVES"), None)
        self.assertIsNotNone(loves_rel, "Failed to find Charlie LOVES David")

if __name__ == '__main__':
    unittest.main()
