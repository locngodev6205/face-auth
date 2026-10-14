"""Display-only hysteresis must not change raw recognition or hide invalidation."""
import unittest
from unittest.mock import patch

import numpy as np
import app
from tracking.tracker import Track


class TestIdentityDisplay(unittest.TestCase):
    def setUp(self):
        self.track = Track(1, (20,20,120,120))

    def update(self, identity):
        self.track.update_result(*identity)
        self.track.update_display_identity(identity, 2, 3)

    def test_one_unknown_does_not_replace_label_and_raw_result_is_preserved(self):
        self.update(("A", "Alice", .9))
        self.update((None, "UNKNOWN", .2))
        self.assertEqual(self.track.name, "UNKNOWN")
        self.assertEqual(self.track.display_result()[:2], ("Alice", .9))
        self.assertTrue(self.track.display_result()[2])
        self.update(("A", "Alice", .8))
        self.assertFalse(self.track.display_result()[2])

    def test_two_consecutive_new_id_results_change_label(self):
        self.update(("A", "Alice", .9))
        self.update(("B", "Bob", .9))
        self.assertEqual(self.track.employee_id, "B")
        self.assertEqual(self.track.display_result()[0], "Alice")
        self.update(("B", "Bob", .8))
        self.assertEqual(self.track.display_result()[0], "Bob")

    def test_three_unknowns_drop_label(self):
        self.update(("A", "Alice", .9))
        for _ in range(2):
            self.update((None, "UNKNOWN", .1))
            self.assertEqual(self.track.display_result()[0], "Alice")
        self.update((None, "UNKNOWN", .1))
        self.assertEqual(self.track.display_result()[0], "UNKNOWN")

    def test_disagreement_expires_without_a_new_cloud_result(self):
        with patch("tracking.tracker.time.monotonic", return_value=100):
            self.update(("A", "Alice", .9))
            self.update(("B", "Bob", .9))
        with patch("tracking.tracker.time.monotonic", return_value=101):
            self.assertEqual(self.track.display_result(1.5)[0], "Alice")
        with patch("tracking.tracker.time.monotonic", return_value=102):
            self.assertEqual(self.track.display_result(1.5)[0], "UNKNOWN")

    def test_invalidation_immediately_clears_display(self):
        self.update(("A", "Alice", .9))
        self.track.invalidate_verification()
        self.assertEqual(self.track.display_result()[0], "UNKNOWN")

    def test_worker_status_does_not_change_identity_header(self):
        headers = []
        for status, reverifying in [("CLOUD 0.3s", False), ("WAIT 0.1s", True), ("CACHE 0.1s", False)]:
            with patch("app.cv2.putText") as text:
                app.draw_track(np.zeros((180,180,3), np.uint8), (20,20,120,120),
                               "Alice", .9, False, reverifying,
                               recognition_status=status, stable_label=True)
            headers.append(text.call_args_list[0].args[1])
            self.assertIn(status, text.call_args_list[1].args[1])
        self.assertEqual(headers, ["[ID] Alice"] * 3)


if __name__ == "__main__":
    unittest.main()
