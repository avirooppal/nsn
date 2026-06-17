import unittest
import os
from neurosleepnet import Memory
from neurosleepnet.trust.consistency import ConsistencyScorer

class TestConsistencyScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists('test_consistency.db'):
            try:
                os.remove('test_consistency.db')
            except Exception:
                pass
                
        cls.memory = Memory()
        # Overwrite to use test db so we don't pollute
        cls.memory.storage.db_path = 'test_consistency.db'
        cls.memory.storage._initialize_db()
        
        # Store a baseline memory
        cls.memory.store("User likes apples.")
        cls.scorer = ConsistencyScorer(cls.memory)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test_consistency.db'):
            try:
                os.remove('test_consistency.db')
            except Exception:
                pass

    def test_score_novel(self):
        # Completely different topic
        score = self.scorer.score("The car is red.")
        self.assertEqual(score, 0.8)

    def test_score_consistent(self):
        # Similar meaning, no negation conflict
        score = self.scorer.score("User loves apples.")
        self.assertEqual(score, 1.0)

    def test_score_conflict(self):
        # High similarity but introduces a negation
        score = self.scorer.score("User does not like apples.")
        self.assertEqual(score, 0.2)

if __name__ == '__main__':
    unittest.main()
