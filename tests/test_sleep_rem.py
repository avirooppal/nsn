import unittest
import os
import json
import uuid
import datetime
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.sleep.engine import SleepEngine

class DummyMemory:
    def __init__(self, storage):
        self.storage = storage
        self.namespace = 'default'

class TestSleepREM(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_rem.db'):
            try:
                os.remove('test_rem.db')
            except Exception:
                pass
        self.storage = SQLiteAdapter('test_rem.db')
        self.memory = DummyMemory(self.storage)
        self.engine = SleepEngine(self.memory)

    def tearDown(self):
        if os.path.exists('test_rem.db'):
            try:
                os.remove('test_rem.db')
            except Exception:
                pass

    def test_rem_consolidation_resolves_contradictions(self):
        # Insert two contradictory memories
        mem1_id = str(uuid.uuid4())
        mem2_id = str(uuid.uuid4())
        
        # Memory 1 has high trust score
        self.storage.store(
            mem1_id, 
            "The sky is blue.", 
            datetime.datetime.utcnow().isoformat(), 
            json.dumps({"type": "SEMANTIC"}),
            trust_score=0.9
        )
        
        # Memory 2 has low trust score and contradicts memory 1
        self.storage.store(
            mem2_id, 
            "The sky is not blue.", 
            datetime.datetime.utcnow().isoformat(), 
            json.dumps({"type": "SEMANTIC"}),
            trust_score=0.4
        )
        
        # Run REM consolidation
        deleted_count = self.engine.rem_consolidation()
        self.assertEqual(deleted_count, 1)
        
        # Verify mem2 was deleted and mem1 was kept
        all_mems = self.storage.list()
        self.assertEqual(len(all_mems), 1)
        self.assertEqual(all_mems[0]['id'], mem1_id)

if __name__ == '__main__':
    unittest.main()
