import unittest
from neurosleepnet.perception.schemas import Observation
from neurosleepnet.perception.classifier import MemoryClassifier

class TestMemoryClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = MemoryClassifier()

    def test_classify_procedural(self):
        obs = Observation(content="Here is a step by step guide on how to build it.")
        self.assertEqual(self.classifier.classify(obs), 'procedural')

    def test_classify_episodic(self):
        obs = Observation(content="The user clicked the login button today.")
        self.assertEqual(self.classifier.classify(obs), 'episodic')

    def test_classify_semantic(self):
        obs = Observation(content="Paris is the capital of France.")
        self.assertEqual(self.classifier.classify(obs), 'semantic')

if __name__ == '__main__':
    unittest.main()
