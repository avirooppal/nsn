import unittest
from neurosleepnet.compression.compressor import ContextCompressor

class TestContextCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = ContextCompressor(max_tokens=20) # Small limit for testing

    def test_compression(self):
        memories = [
            {"content": "The Eiffel Tower is in Paris. It was built in 1889.", "score": 0.8},
            {"content": "Apples are delicious red fruits. Bananas are yellow.", "score": 0.2},
            {"content": "Python is a programming language. It is great for AI.", "score": 0.5}
        ]
        
        # Querying about Paris
        compressed = self.compressor.compress(memories, query="Where is Paris?")
        
        # It should prioritize the Eiffel Tower sentence and drop others if limit reached
        self.assertIn("Eiffel Tower is in Paris", compressed)
        
        # Test max tokens cutoff (20 words)
        words = compressed.split()
        self.assertLessEqual(len(words), 20)

if __name__ == '__main__':
    unittest.main()
