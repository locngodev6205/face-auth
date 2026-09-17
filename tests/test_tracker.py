"""Unit tests for Track PAD gating and recognition permission."""

import unittest
from tracking.tracker import Track


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


if __name__ == "__main__":
    unittest.main()
