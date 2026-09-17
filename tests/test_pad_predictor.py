"""Tests for antispoof predictor and mathematical contracts."""

import math
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from antispoof.predictor import AntiSpoofPredictor


class TestPadMathAndPredictor(unittest.TestCase):

    def test_2class_pad_score_identity(self):
        """Verify 2-class pad_score equals exactly z0 - z1 and matches logit_threshold."""
        with patch("antispoof.predictor.load_model") as mock_load:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.shape = [1, 3, 128, 128]
            mock_session.get_inputs.return_value = [mock_input]
            mock_load.return_value = (mock_session, "input")

            predictor = AntiSpoofPredictor(
                model_path="antispoof/models/best_model_quantized.onnx",
                threshold=0.5,
            )
            self.assertAlmostEqual(predictor.logit_threshold, 0.0, places=6)

            # Test real logits [3.0, 1.0] -> pad_score = 2.0 > 0 -> is_real True
            res = predictor.process_with_logits(np.array([3.0, 1.0]))
            self.assertAlmostEqual(res["pad_score"], 2.0, places=6)
            self.assertAlmostEqual(res["logit_diff"], 2.0, places=6)
            self.assertTrue(res["is_real"])
            self.assertEqual(res["status"], "real")

            # Test spoof logits [1.0, 3.0] -> pad_score = -2.0 < 0 -> is_real False
            res = predictor.process_with_logits(np.array([1.0, 3.0]))
            self.assertAlmostEqual(res["pad_score"], -2.0, places=6)
            self.assertFalse(res["is_real"])
            self.assertEqual(res["status"], "spoof")

    def test_3class_logsumexp_softmax_equivalence(self):
        """Verify 3-class pad_score >= ln(p / (1-p)) is mathematically equivalent to P(REAL) >= p."""
        with patch("antispoof.predictor.load_model") as mock_load:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.shape = [1, 3, 224, 224]
            mock_session.get_inputs.return_value = [mock_input]
            mock_load.return_value = (mock_session, "input")

            for p in [0.3, 0.5, 0.7, 0.85]:
                predictor = AntiSpoofPredictor(
                    model_path="antispoof/models/mnv3_large_3class_best.onnx",
                    threshold=p,
                )
                expected_logit_thresh = math.log(p / (1.0 - p))
                self.assertAlmostEqual(predictor.logit_threshold, expected_logit_thresh, places=5)

                test_cases = [
                    np.array([5.0, 1.0, 2.0]),
                    np.array([1.0, 4.0, 2.0]),
                    np.array([0.0, 0.0, 0.0]),
                    np.array([2.5, 2.5, 1.0]),
                    np.array([10.0, 8.0, 9.0]),
                ]

                for logits in test_cases:
                    res = predictor.process_with_logits(logits)
                    exp_l = np.exp(logits - np.max(logits))
                    softmax_real_prob = exp_l[0] / np.sum(exp_l)
                    is_real_by_softmax = bool(softmax_real_prob >= p)

                    self.assertEqual(
                        res["is_real"],
                        is_real_by_softmax,
                        f"Mismatch for p={p}, logits={logits}: predictor={res['is_real']} vs softmax={is_real_by_softmax}",
                    )

    def test_color_order_detection(self):
        """Verify BGR for MiniFASNet 128 and RGB for MobileNet 224."""
        with patch("antispoof.predictor.load_model") as mock_load:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.shape = [1, 3, 128, 128]
            mock_session.get_inputs.return_value = [mock_input]
            mock_load.return_value = (mock_session, "input")

            pred_minifas = AntiSpoofPredictor(
                model_path="antispoof/models/best_model_quantized.onnx",
            )
            self.assertEqual(pred_minifas.color_order, "BGR")
            self.assertFalse(pred_minifas.convert_rgb)

            mock_input.shape = [1, 3, 224, 224]
            pred_mnv3 = AntiSpoofPredictor(
                model_path="antispoof/models/mnv3_large_3class_best.onnx",
            )
            self.assertEqual(pred_mnv3.color_order, "RGB")
            self.assertTrue(pred_mnv3.convert_rgb)


if __name__ == "__main__":
    unittest.main()
