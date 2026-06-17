import unittest
import os
import json
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.graph.schemas import GraphNode, GraphEdge

class TestGraphQuery(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_graph_query.db'):
            try:
                os.remove('test_graph_query.db')
            except Exception:
                pass
        self.storage = SQLiteAdapter('test_graph_query.db')

    def tearDown(self):
        if os.path.exists('test_graph_query.db'):
            try:
                os.remove('test_graph_query.db')
            except Exception:
                pass

    def test_query_graph(self):
        node1 = GraphNode(label="Person", name="Alice")
        node2 = GraphNode(label="Person", name="Bob")
        
        self.storage.store_graph_node(node1.id, node1.label, node1.name, json.dumps(node1.properties), node1.created_at)
        self.storage.store_graph_node(node2.id, node2.label, node2.name, json.dumps(node2.properties), node2.created_at)
        
        edge = GraphEdge(source_id=node1.id, target_id=node2.id, relation="KNOWS")
        self.storage.store_graph_edge(edge.id, edge.source_id, edge.target_id, edge.relation, json.dumps(edge.properties), edge.created_at)

        result = self.storage.query_graph("Alice")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["node"]["name"], "Alice")
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["edges"][0]["relation"], "KNOWS")
        self.assertEqual(result["edges"][0]["target"]["name"], "Bob")

if __name__ == '__main__':
    unittest.main()
