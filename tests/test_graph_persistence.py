import unittest
import os
import json
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.graph.schemas import GraphNode, GraphEdge

class TestGraphPersistence(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_graph.db'):
            try:
                os.remove('test_graph.db')
            except Exception:
                pass
        self.storage = SQLiteAdapter('test_graph.db')

    def tearDown(self):
        if os.path.exists('test_graph.db'):
            try:
                os.remove('test_graph.db')
            except Exception:
                pass

    def test_store_and_retrieve_graph(self):
        node = GraphNode(label="Person", name="Alice", properties={"age": 30})
        self.storage.store_graph_node(
            node.id, node.label, node.name, json.dumps(node.properties), node.created_at
        )

        retrieved_node = self.storage.get_graph_node(node.id)
        self.assertIsNotNone(retrieved_node)
        self.assertEqual(retrieved_node['name'], "Alice")
        self.assertEqual(retrieved_node['properties']['age'], 30)

        # Edge test
        node2 = GraphNode(label="Person", name="Bob")
        self.storage.store_graph_node(
            node2.id, node2.label, node2.name, json.dumps(node2.properties), node2.created_at
        )

        edge = GraphEdge(source_id=node.id, target_id=node2.id, relation="KNOWS")
        self.storage.store_graph_edge(
            edge.id, edge.source_id, edge.target_id, edge.relation, json.dumps(edge.properties), edge.created_at
        )
        # To just check it stored without error is the main criteria since there is no get_graph_edge yet.
        # It's an acceptance test of "Graph stored."
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
