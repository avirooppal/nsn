import unittest
from neurosleepnet.graph.schemas import GraphNode, GraphEdge

class TestGraphEdge(unittest.TestCase):
    def test_edge_creation(self):
        node1 = GraphNode(label="Person", name="Alice")
        node2 = GraphNode(label="Person", name="Bob")
        
        edge = GraphEdge(source_id=node1.id, target_id=node2.id, relation="KNOWS", properties={"since": 2020})
        
        self.assertEqual(edge.source_id, node1.id)
        self.assertEqual(edge.target_id, node2.id)
        self.assertEqual(edge.relation, "KNOWS")
        self.assertIsNotNone(edge.id)
        self.assertIsNotNone(edge.created_at)
        self.assertEqual(edge.properties["since"], 2020)

if __name__ == '__main__':
    unittest.main()
