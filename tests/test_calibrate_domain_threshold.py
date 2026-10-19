"""Deterministic calibration checks; independent of camera and real data."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrate_domain_threshold", ROOT / "scripts/calibrate_domain_threshold.py")
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class TestThresholdSearch(unittest.TestCase):
    def test_separable_and_inclusive_boundary(self):
        best, curve = calibrate.select_threshold([-3, -1, 2, 4], [0, 0, 1, 1])
        self.assertEqual(best["threshold_logit"], 2)
        self.assertEqual(best["ACER"], 0)
        self.assertEqual(best["HTER"], 0)
        self.assertEqual(len(curve), 5)

    def test_all_tied_scores_prefers_no_false_accepts(self):
        best, curve = calibrate.select_threshold([0, 0, 0], [0, 1, 1])
        self.assertEqual(len(curve), 2)
        self.assertGreater(best["threshold_logit"], 0)
        self.assertEqual(best["APCER"], 0)
        self.assertEqual(best["BPCER"], 1)
        self.assertEqual(best["HTER"], 0.5)

    def test_balanced_error_not_accuracy_on_imbalanced_data(self):
        # 100 fake samples score 0; 3 fake and 2 real score 2.
        # Accepting score 2 minimizes balanced error despite three false accepts.
        best, _ = calibrate.select_threshold([0] * 100 + [2] * 5, [0] * 103 + [1] * 2)
        self.assertEqual(best["threshold_logit"], 2)
        self.assertEqual(best["fake_as_real"], 3)
        self.assertEqual(best["real_as_fake"], 0)
        self.assertAlmostEqual(best["HTER"], 3 / 103 / 2)

    def test_random_sweeps_match_brute_force_with_ties(self):
        rng = np.random.default_rng(2026)
        for _ in range(100):
            scores = rng.integers(-5, 6, size=40).astype(float)
            labels = np.concatenate(([0, 1], rng.integers(0, 2, size=38)))
            best, curve = calibrate.select_threshold(scores, labels)
            candidates = np.append(np.unique(scores), np.nextafter(scores.max(), np.inf))
            n_real, n_fake = int(labels.sum()), int((labels == 0).sum())
            brute = []
            for threshold in candidates:
                actual = calibrate.metrics(scores, labels, float(threshold))
                brute.append(actual)
            expected = min(brute, key=lambda m: (
                m["real_as_fake"] * n_fake + m["fake_as_real"] * n_real,
                m["fake_as_real"], -m["threshold_logit"]))
            self.assertEqual(best, expected)
            for row, actual in zip(curve, brute):
                self.assertEqual(row["ACER"], actual["ACER"])
                self.assertEqual(row["fake_as_real"], actual["fake_as_real"])
                self.assertEqual(row["real_as_fake"], actual["real_as_fake"])

    def test_invalid_scores_or_missing_class_fail(self):
        for scores, labels in [([], []), ([1, 2], [1, 1]), ([float("nan"), 0], [1, 0]),
                               ([1, 2], [1]), ([1, 2], [1, 2])]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                calibrate.select_threshold(scores, labels)


class TestImageInput(unittest.TestCase):
    def test_native_rectangular_crop_gets_runtime_resize_without_recrop(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "real/p01").mkdir(parents=True)
            (directory / "fake").mkdir()
            rng = np.random.default_rng(1)
            real = rng.integers(0, 256, (61, 103, 3), dtype=np.uint8)
            fake = rng.integers(0, 256, (117, 81, 3), dtype=np.uint8)
            cv2.imwrite(str(directory / "real/p01/face.PNG"), real)
            cv2.imwrite(str(directory / "fake/face.png"), fake)
            directory, files = calibrate.collect_images(directory)
            seen = []

            def predict(patches):
                seen.extend(patches)
                return [{"pad_score": 1.0}, {"pad_score": -1.0}]

            predictor = SimpleNamespace(input_metadata=SimpleNamespace(shape=["batch", 3, 80, 80]),
                                        crop_mode="minifasnet_train_v1", model_img_size=80, predict_crops=predict)
            rows = calibrate.score_images(directory, files, predictor, 32)
            self.assertEqual([r["label"] for r in rows], ["real", "fake"])
            np.testing.assert_array_equal(seen[0], cv2.resize(real, (80, 80)))
            np.testing.assert_array_equal(seen[1], cv2.resize(fake, (80, 80)))
            self.assertEqual(calibrate.select_threshold(*calibrate.row_arrays(rows))[0]["ACER"], 0)

    def test_corrupt_image_fails_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "bad.png"
            path.write_bytes(b"invalid")
            predictor = SimpleNamespace(input_metadata=SimpleNamespace(shape=["batch", 3, 80, 80]))
            with self.assertRaisesRegex(ValueError, "Ảnh hỏng"):
                calibrate.score_images(directory, [(path, 1)], predictor, 1)

    def test_missing_folder_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            calibrate.collect_images(Path(temp))

    def test_failed_inference_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "face.png"
            cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))
            predictor = SimpleNamespace(input_metadata=SimpleNamespace(shape=["batch", 3, 80, 80]),
                                        crop_mode="minifasnet_train_v1", model_img_size=80,
                                        predict_crops=lambda patches: [])
            with self.assertRaises(RuntimeError):
                calibrate.score_images(directory, [(path, 1)], predictor, 1)


class TestRuntimeExport(unittest.TestCase):
    def test_export_copies_contract_and_updates_both_threshold_units(self):
        source = {"model_file": "old.onnx", "calibrated_logit_threshold": -4,
                  "predictor_threshold_probability": 0.018, "model_sha256": "abc"}
        predictor = SimpleNamespace(model_path=Path("/tmp/model.onnx"), model_img_size=80,
            bbox_expansion_factor=1.5, crop_mode="minifasnet_train_v1", apply_gamma=False,
            color_order="BGR", mean=None, std=None)
        best = calibrate.metrics([2, -1], [1, 0], 2)
        exported = calibrate.export_runtime(source, predictor, best, {"unit": "image"})
        self.assertEqual(source["calibrated_logit_threshold"], -4)
        self.assertEqual(exported["calibrated_logit_threshold"], 2)
        self.assertEqual(exported["predictor_threshold_probability"], calibrate.probability(2))
        self.assertEqual(exported["model_sha256"], "abc")
        self.assertEqual(exported["model_file"], "/tmp/model.onnx")

    def test_unrepresentable_runtime_probability_is_reported(self):
        with self.assertRaisesRegex(ValueError, "độ chính xác"):
            calibrate.export_runtime({}, None, {"threshold_logit": 100, "threshold_probability": 1}, {})


if __name__ == "__main__":
    unittest.main()
