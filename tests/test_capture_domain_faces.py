"""Offline checks; no camera, GUI, model download or database required."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_file("capture_domain_faces", ROOT / "scripts/capture_domain_faces.py")
preprocess = load_file("domain_crop_reference", ROOT / "antispoof/preprocess.py")


class TestDomainCapture(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(0, 256, (241, 319, 3), dtype=np.uint8)
        self.detections = [
            {"bbox": (80, 60, 180, 160), "score": 0.95, "landmarks": None},
            {"bbox": (240, 155, 317, 239), "score": 0.85, "landmarks": None},
        ]

    def test_native_crop_matches_runtime_including_edges(self):
        boxes = [d["bbox"] for d in self.detections] + [
            (0, 0, 90, 100), (70, 40, 190, 180), (0, 0, 318, 240),
            (-2, -3, 80, 90), (270, 190, 325, 248),
        ]
        for bbox in boxes:
            with self.subTest(bbox=bbox):
                (left, top, right, bottom), _ = capture.expanded_bounds(self.frame.shape, bbox)
                actual = cv2.resize(self.frame[top:bottom + 1, left:right + 1], (80, 80))
                expected = preprocess.crop_minifasnet_train_v1(self.frame, bbox, 1.5, 80)
                np.testing.assert_array_equal(actual, expected)

    def test_save_all_faces_losslessly_and_keep_source(self):
        original = self.frame.copy()
        with tempfile.TemporaryDirectory() as temp:
            output = capture.save_capture(Path(temp), self.frame, self.detections, 1.5, "test-time")
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(len(metadata["faces"]), 2)
            self.assertEqual(metadata["captured_at_utc"], "test-time")
            np.testing.assert_array_equal(cv2.imread(str(output / "frame.png")), original)
            for face in metadata["faces"]:
                self.assertIsNone(face["label"])
                left, top, right, bottom = face["crop_xyxy_inclusive"]
                np.testing.assert_array_equal(cv2.imread(str(output / face["file"])),
                                              original[top:bottom + 1, left:right + 1])
            second = capture.save_capture(Path(temp), self.frame, self.detections, 1.5, "test-time")
            self.assertNotEqual(output, second)
        np.testing.assert_array_equal(self.frame, original)

    def test_no_faces_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(capture.save_capture(Path(temp), self.frame, [], 1.5, "test-time"))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_failed_write_does_not_publish_partial_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(capture, "write_png", side_effect=[None, OSError("disk full")]):
                with self.assertRaises(OSError):
                    capture.save_capture(Path(temp), self.frame, self.detections, 1.5, "test-time")
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_invalid_geometry_and_scale(self):
        for bbox, scale in [((10, 10, 10, 20), 1.5), ((10, 10, 20, 20), 0.5),
                            ((10, 10, 20, 20), float("nan"))]:
            with self.subTest(bbox=bbox, scale=scale), self.assertRaises(ValueError):
                capture.expanded_bounds(self.frame.shape, bbox, scale)


if __name__ == "__main__":
    unittest.main()
