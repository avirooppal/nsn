import unittest
from neurosleepnet.graph.schemas import GraphNode

class TestGraphNode(unittest.TestCase):
    def test_node_creation(self):
        node = GraphNode(label="Person", name="Alice", properties={"age": 30})
        self.assertEqual(node.label, "Person")
        self.assertEqual(node.name, "Alice")
        self.assertIsNotNone(node.id)
        self.assertIsNotNone(node.created_at)
        self.assertEqual(node.properties["age"], 30)

if __name__ == '__main__':
    unittest.main()
