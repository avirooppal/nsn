import unittest
import os
import tempfile
from neurosleepnet import Memory
from neurosleepnet.trust.consistency import ConsistencyScorer

class TestConsistencyScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="nsn_trust_test_")
        cls.memory = Memory(db_path=os.path.join(cls.directory.name, 'memory.db'))

        cls.memory.store("User likes apples.")
        cls.scorer = ConsistencyScorer(cls.memory)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

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
        self.assertEqual(score, 0.15)

if __name__ == '__main__':
    unittest.main()
