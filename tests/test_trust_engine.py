import unittest
import os
import tempfile
from neurosleepnet import Memory
from neurosleepnet.perception.schemas import Observation
from neurosleepnet.trust.engine import TrustEngine

class TestTrustEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="nsn_trust_test_")
        cls.memory = Memory(db_path=os.path.join(cls.directory.name, 'memory.db'))

        cls.engine = TrustEngine(cls.memory)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_calculate_trust(self):
        obs = Observation(content="This is a test observation.", source="system")
        profile = self.engine.calculate(obs)
        
        self.assertIsNotNone(profile)
        self.assertGreater(profile.final_score, 0.0)
        self.assertEqual(profile.source_score, 1.0) # system source

if __name__ == '__main__':
    unittest.main()
