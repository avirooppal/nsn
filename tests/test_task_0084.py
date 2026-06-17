import unittest
import os
from neurosleepnet import Memory
from neurosleepnet.perception.schemas import Observation
from neurosleepnet.trust.engine import TrustEngine

class TestTrustEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists('test_trust_engine.db'):
            try:
                os.remove('test_trust_engine.db')
            except Exception:
                pass
                
        cls.memory = Memory()
        cls.memory.storage.db_path = 'test_trust_engine.db'
        cls.memory.storage._initialize_db()
        cls.engine = TrustEngine(cls.memory)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test_trust_engine.db'):
            try:
                os.remove('test_trust_engine.db')
            except Exception:
                pass

    def test_calculate_trust(self):
        obs = Observation(content="This is a test observation.", source="system")
        profile = self.engine.calculate(obs)
        
        self.assertIsNotNone(profile)
        self.assertGreater(profile.final_score, 0.0)
        self.assertEqual(profile.source_score, 1.0) # system source

if __name__ == '__main__':
    unittest.main()
