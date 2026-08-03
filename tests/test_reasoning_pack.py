import unittest
import os
import json
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.compression.pack import ReasoningPackGenerator
from neurosleepnet.graph.schemas import GraphNode, GraphEdge

class TestReasoningPackGenerator(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_pack.db'):
            try:
                os.remove('test_pack.db')
            except Exception:
                pass
        self.storage = SQLiteAdapter('test_pack.db')
        self.generator = ReasoningPackGenerator(self.storage)

    def tearDown(self):
        if os.path.exists('test_pack.db'):
            try:
                os.remove('test_pack.db')
            except Exception:
                pass

    def test_generate_pack(self):
        # 1. Store a memory
        self.storage.store("mem1", "Alice is an engineer who built Neurosleepnet in 2026.", "2026-01-01T00:00:00", "{}", 1.0, 1.0, "[]")
        
        # 2. Store Graph Nodes and Edges
        node1 = GraphNode(label="Person", name="Alice")
        node2 = GraphNode(label="Project", name="Neurosleepnet")
        self.storage.store_graph_node(node1.id, node1.label, node1.name, "{}", node1.created_at)
        self.storage.store_graph_node(node2.id, node2.label, node2.name, "{}", node2.created_at)
        
        edge = GraphEdge(source_id=node1.id, target_id=node2.id, relation="BUILT")
        self.storage.store_graph_edge(edge.id, edge.source_id, edge.target_id, edge.relation, "{}", edge.created_at)
        
        # Generate Pack for "Alice"
        pack_json = self.generator.generate_pack("Alice")
        pack = json.loads(pack_json)
        
        self.assertEqual(pack["topic"], "Alice")
        self.assertTrue(any("Alice is an engineer" in c for c in pack["context"]))
        self.assertIn("(Alice) -[BUILT]-> (Neurosleepnet)", pack["logical_rules"])
        self.assertIn("system_prompt", pack)

if __name__ == '__main__':
    unittest.main()
