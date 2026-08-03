import unittest
from neurosleepnet.graph.extractor import EntityExtractor

class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EntityExtractor()

    def test_extract_entities(self):
        text = "The quick brown fox jumps over the lazy dog in New York. John Smith was there."
        entities = self.extractor.extract(text)
        
        self.assertTrue(any(e["name"] == "New York" for e in entities))
        self.assertTrue(any(e["name"] == "John Smith" for e in entities))
        # 'The' at the start should not be an entity
        self.assertFalse(any(e["name"] == "The" for e in entities))

    def test_no_entities(self):
        text = "this is a lower case sentence with no entities."
        entities = self.extractor.extract(text)
        self.assertEqual(len(entities), 0)

if __name__ == '__main__':
    unittest.main()
