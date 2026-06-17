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

class TestSleepNREM(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_nrem.db'):
            try:
                os.remove('test_nrem.db')
            except Exception:
                pass
        self.storage = SQLiteAdapter('test_nrem.db')
        self.memory = DummyMemory(self.storage)
        self.engine = SleepEngine(self.memory)

    def tearDown(self):
        if os.path.exists('test_nrem.db'):
            try:
                os.remove('test_nrem.db')
            except Exception:
                pass

    def test_nrem_consolidation(self):
        # Insert 2 unconsolidated episodic memories
        self.storage.store(str(uuid.uuid4()), "Alice likes apples.", datetime.datetime.utcnow().isoformat(), json.dumps({"type": "EPISODIC"}))
        self.storage.store(str(uuid.uuid4()), "Bob likes bananas.", datetime.datetime.utcnow().isoformat(), json.dumps({"type": "EPISODIC"}))
        
        # Insert 1 semantic memory (should be ignored)
        self.storage.store(str(uuid.uuid4()), "The sky is blue.", datetime.datetime.utcnow().isoformat(), json.dumps({"type": "SEMANTIC"}))

        semantic_id = self.engine.nrem_consolidation()
        self.assertIsNotNone(semantic_id)

        # Verify new memory is semantic
        semantic_mem = self.storage.get(semantic_id)
        self.assertEqual(semantic_mem['metadata']['type'], 'SEMANTIC')
        self.assertEqual(semantic_mem['metadata']['consolidated_count'], 2)
        self.assertIn("Alice likes apples.", semantic_mem['content'])
        self.assertIn("Bob likes bananas.", semantic_mem['content'])

        # Verify old episodic memories are marked as consolidated
        all_mems = self.storage.list()
        for m in all_mems:
            if m['id'] != semantic_id and m['metadata'].get('type') == 'EPISODIC':
                self.assertTrue(m['metadata'].get('consolidated', False))

if __name__ == '__main__':
    unittest.main()
