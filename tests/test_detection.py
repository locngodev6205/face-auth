"""Focused tests for SCRFD detection filtering."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from detection.detector import FaceDetector


class TestScrfdDetectionFiltering(unittest.TestCase):
    def test_min_face_size_filters_small_scrfd_boxes(self):
        detector = FaceDetector.__new__(FaceDetector)
        detector.min_face_size = 60
        detector._model = MagicMock()
        detector._model.detect.return_value = (
            np.array(
                [
                    [10, 10, 80, 90, 0.95],   # 70x80: keep
                    [100, 100, 150, 180, 0.99],  # 50x80: reject
                    [200, 200, 280, 250, 0.98],  # 80x50: reject
                ],
                dtype=np.float32,
            ),
            np.zeros((3, 5, 2), dtype=np.float32),
        )

        detections = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], (10, 10, 80, 90))


if __name__ == "__main__":
    unittest.main()
