"""Offline deployment checks against the selected notebook17 checkpoint."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch
from torch import nn

import config
from antispoof.predictor import AntiSpoofPredictor
from antispoof.preprocess import crop_for_runtime, preprocess_batch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'antispoof/models/R7_SC_100K_crop15.onnx'
CONTRACT = ROOT / 'antispoof/models/R7_SC_100K_crop15_runtime_config.json'
NOTEBOOK = ROOT / 'antispoof/notebooks/final/binary_crossdomain_v2/17_scale100k_C_P3SF_R7SC_crop15.ipynb'
CHECKPOINT = ROOT / 'antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15/best.pth'


class TestR7Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT.read_text())
        cls.predictor = AntiSpoofPredictor(
            model_path=str(MODEL), model_img_size=80, bbox_expansion_factor=1.5,
            threshold_logit=cls.contract['calibrated_logit_threshold'],
            apply_gamma=False, color_order='BGR', crop_mode='minifasnet_train_v1',
        )
        cls.predictor.output_classes = 2
        notebook = json.loads(NOTEBOOK.read_text())
        source = ''.join(notebook['cells'][2]['source'])
        nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        spec = importlib.util.spec_from_file_location('r7_test_source', ROOT / 'antispoof/models/MiniFASNet.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ns = dict(module=module, nn=nn, np=np, cv2=cv2, CROP_SCALE=1.5, INPUT_SIZE=80)
        for name in ('PADNet', 'mini_crop_bgr'):
            exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), name, 'exec'), ns)
        cls.notebook_crop = staticmethod(ns['mini_crop_bgr'])
        cls.notebook_model = ns['PADNet'](True).eval()
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        cls.notebook_model.load_state_dict(state['state_dict'], strict=True)
        torch.set_num_threads(1)

    def test_selected_config_is_r7_binary_crop15(self):
        self.assertEqual(Path(config.PAD_MODEL_PATH), MODEL)
        self.assertEqual(config.PAD_OUTPUT_CLASSES, 2)
        self.assertEqual(config.PAD_MODEL_IMG_SIZE, 80)
        self.assertEqual(config.PAD_BBOX_EXPANSION_FACTOR, 1.5)
        self.assertEqual(config.PAD_COLOR_ORDER, 'BGR')
        self.assertIsNone(config.PAD_MEAN)
        self.assertIsNone(config.PAD_STD)
        self.assertFalse(config.PAD_GAMMA_ENABLED)
        self.assertFalse(config.PAD_CROP_SMOOTHING)
        self.assertEqual(config.PAD_THRESHOLD_LOGIT, -0.373016357421875)
        self.assertAlmostEqual(config.PAD_THRESHOLD, 1 / (1 + math.exp(-config.PAD_THRESHOLD_LOGIT)))
        self.assertEqual(self.predictor.output_metadata.shape, ['batch', 2])

    def test_actual_crop_preprocess_and_onnx_match_notebook(self):
        rng = np.random.default_rng(100)
        frame = rng.integers(0, 256, (241, 319, 3), dtype=np.uint8)
        # Interior, rectangular, border and full-image bboxes exercise boundary-limited crops.
        boxes = [(80, 60, 180, 160), (70.5, 40.5, 190.25, 180.75),
                 (0, 0, 90, 100), (240, 155, 317, 239), (0, 0, 318, 240)]
        patches = []
        expected_tensors = []
        for box in boxes:
            expected = self.notebook_crop(frame, box)
            actual = crop_for_runtime(frame, box, 1.5, 80, 'minifasnet_train_v1')
            np.testing.assert_array_equal(actual, expected)
            patches.append(actual)
            expected_tensors.append(expected.transpose(2, 0, 1).astype(np.float32) / 255.)
        expected_batch = np.stack(expected_tensors)
        actual_batch = preprocess_batch(patches, 80, mean=None, std=None,
                                        apply_gamma=False, convert_rgb=False)
        np.testing.assert_array_equal(actual_batch, expected_batch)
        with torch.inference_mode():
            expected_logits = self.notebook_model(torch.from_numpy(expected_batch)).numpy()
        actual_logits = self.predictor.session.run(['logits'], {'input': actual_batch})[0]
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)
        results = self.predictor.predict_crops(patches)
        for index, result in enumerate(results):
            score = float(expected_logits[index, 0]) - float(expected_logits[index, 1])
            self.assertAlmostEqual(result['pad_score'], score, delta=2e-5)
            self.assertEqual(result['is_real'], score >= self.predictor.logit_threshold)
            self.assertEqual(set(result['class_probs']), {'real', 'spoof'})
            self.assertIn(result['detailed_status'], ('REAL', 'SPOOF'))

    def test_binary_threshold_boundary_and_invalid_outputs(self):
        t = self.predictor.logit_threshold
        for score in (t - 1e-6, t, t + 1e-6, 0.):
            result = self.predictor.process_with_logits(np.array([score, 0.]))
            self.assertEqual(result['pad_score'], score)
            self.assertEqual(result['is_real'], score >= t)
        for logits in ([0, 0, 0], [0], [[0, 0]], [np.nan, 0], [0, np.inf]):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                self.predictor.process_with_logits(np.asarray(logits))

    def test_legacy_three_logit_math_remains_supported(self):
        predictor = AntiSpoofPredictor.__new__(AntiSpoofPredictor)
        predictor.output_classes = None
        predictor.logit_threshold = math.log(.3 / .7)
        predictor.threshold_probability = .3
        for logits in ([5., 1., 2.], [1., 4., 2.], [0., 0., 0.]):
            probabilities = np.exp(np.asarray(logits) - max(logits))
            probabilities /= probabilities.sum()
            result = predictor.process_with_logits(np.asarray(logits))
            self.assertEqual(result['is_real'], probabilities[0] >= .3)


if __name__ == '__main__':
    unittest.main()
