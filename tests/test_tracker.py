"""Unit tests for Track PAD gating and recognition permission."""

import unittest
import numpy as np
from tracking.tracker import Track
from tracking.tracker import FaceTracker


class TestTrackerAndGating(unittest.TestCase):

    def test_pad_pending_blocks_recognition(self):
        """PAD chưa đủ vote thì recognition bị chặn."""
        track = Track(track_id=1, bbox=(10, 10, 50, 50), pad_smooth_window=5, pad_min_votes=5)

        # Ban đầu (0 votes) -> PAD_PENDING -> recognition bị chặn
        self.assertEqual(track.pad_status, "PAD_PENDING")
        self.assertFalse(track.needs_recognition(interval_seconds=1.0, pad_enabled=True))

        # Tích lũy 4 votes (chưa đủ 5) -> vẫn bị chặn
        for _ in range(4):
            track.update_pad(is_real=True, pad_score=2.0)
            self.assertEqual(track.pad_status, "PAD_PENDING")
            self.assertFalse(track.needs_recognition(interval_seconds=1.0, pad_enabled=True))

    def test_pad_real_allows_recognition(self):
        """Đủ vote REAL thì recognition được phép."""
        track = Track(track_id=1, bbox=(10, 10, 50, 50), pad_smooth_window=5, pad_min_votes=5)

        for _ in range(5):
            track.update_pad(is_real=True, pad_score=2.0)

        self.assertTrue(track.pad_ready)
        self.assertTrue(track.is_real)
        self.assertEqual(track.pad_status, "REAL")
        self.assertTrue(track.needs_recognition(interval_seconds=1.0, pad_enabled=True))

    def test_pad_spoof_blocks_recognition(self):
        """SPOOF thì recognition bị chặn."""
        track = Track(track_id=1, bbox=(10, 10, 50, 50), pad_smooth_window=5, pad_min_votes=5)

        # 1 vote REAL, 4 votes SPOOF -> 80% spoof >= 60% ngưỡng -> SPOOF
        track.update_pad(is_real=True, pad_score=1.0)
        for _ in range(4):
            track.update_pad(is_real=False, pad_score=-2.0)

        self.assertTrue(track.pad_ready)
        self.assertTrue(track.is_spoof)
        self.assertEqual(track.pad_status, "SPOOF")
        self.assertFalse(track.needs_recognition(interval_seconds=1.0, pad_enabled=True))

    def test_intermediate_update_propagates_bbox_without_detector(self):
        """Detector skip phải trả bbox hiện tại từ optical-flow tracker."""
        rng = np.random.default_rng(7)
        old_frame = np.zeros((140, 180, 3), dtype=np.uint8)
        patch = rng.integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
        old_frame[35:85, 40:90] = patch
        new_frame = np.zeros_like(old_frame)
        new_frame[35:85, 47:97] = patch

        tracker = FaceTracker(max_missing_frames=2)
        first = tracker.update(
            [{"bbox": (40, 35, 90, 85), "score": 0.99, "landmarks": None}],
            frame=old_frame,
            detector_called=True,
        )
        self.assertEqual(first[0][1].track_id, 1)

        intermediate = tracker.update([], frame=new_frame, detector_called=False)
        self.assertEqual(len(intermediate), 1)
        detection, track = intermediate[0]
        self.assertEqual(detection["bbox_source"], "TRACKER")
        self.assertEqual(track.bbox_source, "TRACKER")
        self.assertAlmostEqual(track.bbox[0], 47, delta=3)
        self.assertFalse(tracker.needs_redetection())

    def test_periodic_redetection_keeps_track_id(self):
        tracker = FaceTracker()
        first = tracker.update(
            [{"bbox": (10, 10, 50, 50), "score": 0.9, "landmarks": None}],
            detector_called=True,
        )
        self.assertFalse(tracker.needs_redetection())
        refreshed = tracker.update(
            [{"bbox": (12, 10, 52, 50), "score": 0.95, "landmarks": None}],
            detector_called=True,
        )
        self.assertEqual(first[0][1].track_id, refreshed[0][1].track_id)
        self.assertEqual(refreshed[0][0]["bbox_source"], "DETECTOR")
        self.assertEqual(refreshed[0][1].bbox, (12, 10, 52, 50))

    def test_missing_frame_forces_redetection(self):
        tracker = FaceTracker(max_missing_frames=2)
        tracker.update(
            [{"bbox": (10, 10, 50, 50), "score": 0.9, "landmarks": None}],
            detector_called=True,
        )
        self.assertEqual(tracker.update([], frame=None, detector_called=False), [])
        self.assertTrue(tracker.needs_redetection())

    def test_pad_smoothing_vote_sequences(self):
        cases = [
            ([True, True, True, True, True], "REAL"),
            ([False, False, False, True, True], "SPOOF"),
            ([True, False, True, False, True], "REAL"),
            ([False, True, True, True, True], "REAL"),
        ]
        for votes, expected in cases:
            with self.subTest(votes=votes):
                track = Track(
                    track_id=1,
                    bbox=(10, 10, 50, 50),
                    pad_smooth_window=5,
                    pad_min_votes=5,
                    pad_spoof_min_ratio=0.6,
                )
                for vote in votes:
                    track.update_pad(is_real=vote)
                self.assertEqual(track.pad_status, expected)


if __name__ == "__main__":
    unittest.main()
