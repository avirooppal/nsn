import unittest
from neurosleepnet import Memory
from neurosleepnet.sleep.engine import SleepEngine

class TestSleepEngine(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.sleep_engine = SleepEngine(self.memory)

    def test_trigger_sleep(self):
        self.assertFalse(self.sleep_engine.is_sleeping)
        
        # Trigger sleep cycle
        result = self.sleep_engine.trigger()
        
        self.assertTrue(result)
        # Should return to False after the synchronous sleep finishes
        self.assertFalse(self.sleep_engine.is_sleeping)

if __name__ == '__main__':
    unittest.main()
