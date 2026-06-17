import unittest
from neurosleepnet.trust.schemas import TrustProfile

class TestTrustSchema(unittest.TestCase):
    def test_trust_profile_creation(self):
        profile = TrustProfile(source_score=0.8, recency_score=0.9, consistency_score=0.7, final_score=0.8)
        self.assertEqual(profile.source_score, 0.8)
        self.assertEqual(profile.recency_score, 0.9)
        self.assertEqual(profile.consistency_score, 0.7)
        self.assertEqual(profile.final_score, 0.8)
        self.assertIsInstance(profile.metadata, dict)

if __name__ == '__main__':
    unittest.main()
