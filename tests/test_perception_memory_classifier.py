import unittest
import datetime
from neurosleepnet.trust.recency import RecencyScorer

class TestRecencyScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = RecencyScorer()

    def test_score_recent(self):
        now = datetime.datetime.utcnow().isoformat()
        self.assertEqual(self.scorer.score(now), 1.0)

    def test_score_few_days_old(self):
        dt = datetime.datetime.utcnow() - datetime.timedelta(days=3)
        self.assertEqual(self.scorer.score(dt.isoformat()), 0.8)

    def test_score_weeks_old(self):
        dt = datetime.datetime.utcnow() - datetime.timedelta(days=15)
        self.assertEqual(self.scorer.score(dt.isoformat()), 0.6)

    def test_score_months_old(self):
        dt = datetime.datetime.utcnow() - datetime.timedelta(days=60)
        self.assertEqual(self.scorer.score(dt.isoformat()), 0.4)

    def test_score_years_old(self):
        dt = datetime.datetime.utcnow() - datetime.timedelta(days=400)
        self.assertEqual(self.scorer.score(dt.isoformat()), 0.2)

    def test_invalid_timestamp(self):
        self.assertEqual(self.scorer.score("invalid-date"), 0.5)

if __name__ == '__main__':
    unittest.main()
