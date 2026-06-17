import unittest
import os
import uuid
import datetime
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.graph.builder import GraphBuilder

class TestGraphMemoryLink(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_graph_link.db'):
            try:
                os.remove('test_graph_link.db')
            except Exception:
                pass
        self.storage = SQLiteAdapter('test_graph_link.db')
        self.builder = GraphBuilder(self.storage)

    def tearDown(self):
        if os.path.exists('test_graph_link.db'):
            try:
                os.remove('test_graph_link.db')
            except Exception:
                pass

    def test_graph_memory_link(self):
        mem_id = str(uuid.uuid4())
        
        # Initial memory store
        self.storage.store(
            memory_id=mem_id,
            content="Alice built Neurosleepnet.",
            created_at=datetime.datetime.utcnow().isoformat()
        )
        
        memory_record = self.storage.get(mem_id)
        
        # Build graph and link
        self.builder.process_memory(memory_record)
        
        # Check that memory now has graph links
        updated_memory = self.storage.get(mem_id)
        self.assertIn('linked_graph_nodes', updated_memory['metadata'])
        self.assertGreater(len(updated_memory['metadata']['linked_graph_nodes']), 0)
        
        # Check that nodes point back to memory
        node_id = updated_memory['metadata']['linked_graph_nodes'][0]
        node = self.storage.get_graph_node(node_id)
        self.assertEqual(node['properties']['source_memory'], mem_id)

if __name__ == '__main__':
    unittest.main()
