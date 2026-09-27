"""Unit tests for Track PAD gating and recognition permission."""

import unittest
from unittest.mock import patch
import numpy as np
from tracking.tracker import Track
from tracking.tracker import FaceTracker


class TestPADCropSmoothing(unittest.TestCase):
    def test_small_jitter_is_reduced_without_changing_track_bbox(self):
        track = Track(1, (100, 100, 200, 200))
        with patch("tracking.tracker.time.monotonic", return_value=100):
            self.assertEqual(track.pad_crop_bbox(track.bbox, (400, 400), 1.55), track.bbox)
        with patch("tracking.tracker.time.monotonic", return_value=100.1):
            result = track.pad_crop_bbox((102, 100, 202, 200), (400, 400), 1.55)
        self.assertGreater(result[0], 100)
        self.assertLess(result[0], 102)
        self.assertEqual(track.bbox, (100, 100, 200, 200))
        self.assertIsNone(track.last_pad_time)
        self.assertEqual(len(track._pad_window), 0)

    def test_large_motion_scale_border_and_gap_snap_to_input(self):
        for bbox, shape, dt in [
            ((120, 100, 220, 200), (400, 400), .1),
            ((94, 94, 206, 206), (400, 400), .1),
            ((102, 100, 202, 200), (225, 225), .1),
            ((102, 100, 202, 200), (400, 400), .6),
        ]:
            with self.subTest(bbox=bbox, shape=shape, dt=dt):
                track = Track(1, (100, 100, 200, 200))
                with patch("tracking.tracker.time.monotonic", return_value=100):
                    track.pad_crop_bbox(track.bbox, (400, 400), 1.55)
                with patch("tracking.tracker.time.monotonic", return_value=100 + dt):
                    self.assertEqual(track.pad_crop_bbox(bbox, shape, 1.55), bbox)

    def test_invalidation_clears_crop_history(self):
        track = Track(1, (100, 100, 200, 200))
        track.pad_crop_bbox(track.bbox, (400, 400), 1.55)
        track.invalidate_verification()
        self.assertIsNone(track._pad_crop_geometry)
        self.assertIsNone(track._pad_crop_time)
        bbox = (102, 100, 202, 200)
        self.assertEqual(track.pad_crop_bbox(bbox, (400, 400), 1.55), bbox)


class TestTrackerAndGating(unittest.TestCase):

    def test_reacquisition_requires_fresh_verification(self):
        tracker = FaceTracker(pad_stale_timeout=3.0)
        det = {"bbox": (30, 30, 130, 130), "score": 0.99, "landmarks": None}
        track = tracker.update([det])[0][1]
        for _ in range(5):
            track.update_pad(True, 2.0)
        track.update_result("A", "Alice", 0.9)
        tracker.update([])
        reacquired = tracker.update([det])[0][1]
        self.assertEqual(reacquired.track_id, track.track_id)
        self.assertEqual(reacquired.pad_status, "PAD_PENDING")
        self.assertIsNone(reacquired.employee_id)
        self.assertTrue(reacquired.is_pending)
        self.assertFalse(reacquired.needs_recognition(pad_enabled=True))

    def test_flow_loss_invalidates_pad_identity_and_stability(self):
        for flow in [(None, None, None),
                     (np.zeros((5, 1, 2), np.float32), np.zeros((5, 1), np.uint8), None)]:
            with self.subTest(flow=flow[0] is None):
                frame = np.zeros((180, 180, 3), np.uint8)
                tracker = FaceTracker()
                track = tracker.update([{"bbox": (30, 30, 130, 130)}], frame=frame)[0][1]
                for _ in range(5):
                    track.update_pad(True, 2.0)
                for _ in range(3):
                    track.update_result("A", "Alice", 0.9)
                with patch("tracking.tracker.cv2.calcOpticalFlowPyrLK", return_value=flow):
                    self.assertEqual(tracker.update([], frame=frame, detector_called=False), [])
                self.assertEqual(track.stable_recognitions, 0)
                self.assertIsNone(track.employee_id)
                self.assertFalse(track.pad_ready)
                self.assertTrue(tracker.needs_redetection())

    def test_flow_scale_transforms_landmarks(self):
        frame = np.zeros((180, 180, 3), np.uint8)
        points = np.array([[30, 30], [130, 30], [130, 130], [30, 130], [80, 80]], np.float32)
        landmarks = np.array([[55, 60], [105, 60], [80, 85], [60, 105], [100, 105]])
        tracker = FaceTracker()
        tracker.update([{"bbox": (30, 30, 130, 130), "landmarks": landmarks.tolist()}], frame=frame)
        moved = (points - 80) * 1.2 + 80
        with patch("tracking.tracker.cv2.calcOpticalFlowPyrLK", return_value=(
            moved.reshape(-1, 1, 2), np.ones((5, 1), np.uint8), None
        )):
            track = tracker.update([], frame=frame, detector_called=False)[0][1]
        self.assertEqual(track.bbox, (20, 20, 140, 140))
        np.testing.assert_allclose(track.landmarks, (landmarks - 80) * 1.2 + 80, atol=1e-5)

    def test_unsafe_flow_is_not_returned_for_inference(self):
        frame = np.zeros((180, 180, 3), np.uint8)
        points = np.array([[30, 30], [130, 30], [130, 130], [30, 130], [80, 80]], np.float32)
        tracker = FaceTracker()
        tracker.update([{"bbox": (30, 30, 130, 130)}], frame=frame)
        with patch("tracking.tracker.cv2.calcOpticalFlowPyrLK", return_value=(
            points.reshape(-1, 1, 2), np.array([[1], [1], [1], [0], [1]], np.uint8), None
        )):
            self.assertEqual(tracker.update([], frame=frame, detector_called=False), [])
        self.assertTrue(tracker.needs_redetection())

    def test_pad_and_recognition_cadence_ignore_wall_clock_changes(self):
        track = Track(1, (10, 10, 50, 50), pad_stale_timeout=3)
        with patch("tracking.tracker.time.monotonic", return_value=100):
            for _ in range(5):
                track.update_pad(True, 2.0)
            track.update_result("A", "Alice", 0.9)
        for wall_time in [-10000, 100000]:
            with patch("tracking.tracker.time.time", return_value=wall_time), patch(
                "tracking.tracker.time.monotonic", return_value=100.2
            ):
                self.assertTrue(track.pad_ready)
                self.assertTrue(track.needs_pad(0.1))
                self.assertFalse(track.needs_recognition(0.5))
        with patch("tracking.tracker.time.monotonic", return_value=104):
            self.assertFalse(track.pad_ready)

    def test_detector_refresh_preserves_votes_only_when_detection_matches(self):
        frame = np.zeros((180, 180, 3), np.uint8)
        points = np.array([[30, 30], [130, 30], [130, 130], [30, 130], [80, 80]], np.float32)
        refresh_cases = [
            (points, np.array([[1], [1], [1], [0], [1]], np.uint8)),
            ((points - 80) * 1.3 + 80, np.ones((5, 1), np.uint8)),
            (points + [70, 0], np.ones((5, 1), np.uint8)),
        ]
        for moved, status in refresh_cases:
            for matched in [True, False]:
                with self.subTest(matched=matched, moved=moved.tolist()):
                    tracker = FaceTracker()
                    det = {"bbox": (30, 30, 130, 130)}
                    track = tracker.update([det], frame=frame)[0][1]
                    for _ in range(5):
                        track.update_pad(True, 2.0)
                    track.update_result("A", "Alice", 0.9)
                    pad_time = track.last_pad_time
                    with patch("tracking.tracker.cv2.calcOpticalFlowPyrLK", return_value=(
                        moved.astype(np.float32).reshape(-1, 1, 2), status, None
                    )):
                        self.assertEqual(tracker.update([], frame=frame, detector_called=False), [])
                    self.assertTrue(tracker.needs_redetection())
                    self.assertEqual(track.bbox, det["bbox"])
                    tracker.update([det] if matched else [], frame=frame)
                    if matched:
                        self.assertTrue(track.pad_ready)
                        self.assertEqual(track.employee_id, "A")
                        self.assertEqual(track.last_pad_time, pad_time)
                    else:
                        self.assertFalse(track.pad_ready)
                        self.assertIsNone(track.employee_id)

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
