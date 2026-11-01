"""Dashboard smoke tests; run after installing requirements."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

class DashboardTests(unittest.TestCase):
    def test_upload_screen_starts_without_model_download(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, 'Every frame. A clearer picture.')
        self.assertTrue(app.button[0].disabled)

    def test_completed_statistics_render(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
        app.session_state['upload_result'] = dict(frames=1, fps=10, current={'person': 1},
            unique={'person': 1}, observations={'person': 1},
            history=[dict(frame=1, time_seconds=0, objects=1)], detections=[])
        app.run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[1].value, '1')

if __name__ == '__main__':
    unittest.main()
