import unittest
from neurosleepnet.trust.source import SourceScorer

class TestSourceScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = SourceScorer()

    def test_score_system(self):
        self.assertEqual(self.scorer.score('system'), 1.0)

    def test_score_user(self):
        self.assertEqual(self.scorer.score('user'), 0.9)

    def test_score_web(self):
        self.assertEqual(self.scorer.score('web'), 0.4)

    def test_score_unknown(self):
        self.assertEqual(self.scorer.score('unknown'), 0.5)

    def test_score_unregistered(self):
        self.assertEqual(self.scorer.score('random_source'), 0.5)

if __name__ == '__main__':
    unittest.main()
