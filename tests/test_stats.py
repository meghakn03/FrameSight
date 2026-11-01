import unittest
from framesight.stats import Detection, Statistics

class StatisticsTests(unittest.TestCase):
    def test_unique_tracks_are_not_frame_counts(self):
        stats = Statistics()
        person = Detection('person', .9, 1, (0, 0, 10, 10))
        stats.update([person], 0, .1)
        stats.update([person, Detection('car', .8, 2, (0, 0, 2, 2))], 1, .1)
        stats.update([], 2, .1)
        result = stats.snapshot()
        self.assertEqual(result['unique'], {'person': 1, 'car': 1})
        self.assertEqual(result['observations'], {'person': 2, 'car': 1})
        self.assertEqual(result['current'], {})
        self.assertAlmostEqual(result['fps'], 10)

    def test_untracked_boxes_and_bounded_history(self):
        stats = Statistics(history_limit=1, detection_limit=1)
        for i in range(2):
            stats.update([Detection('dog', .5, None, (0, 0, 1, 1))], i, .2)
        result = stats.snapshot()
        self.assertEqual(result['unique'], {})
        self.assertEqual(len(result['history']), 1)
        self.assertEqual(len(result['detections']), 1)
        self.assertEqual(result['observations'], {'dog': 2})

    def test_class_changes_do_not_double_count_id(self):
        stats = Statistics()
        for category in ['car', 'truck']:
            stats.update([Detection(category, .7, 1, (0, 0, 1, 1))], 0, .1)
        self.assertEqual(stats.snapshot()['unique'], {'car': 1})

if __name__ == '__main__':
    unittest.main()
