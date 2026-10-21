"""Offline capture: preserve pixels, geometry and unlabelled metadata."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import config
from pad_capture import PADCapture
from antispoof.preprocess import crop_for_runtime
from vision_pipeline import limit_faces


class CaptureTests(unittest.TestCase):
    def test_limit_selects_three_largest_without_mutating_input(self):
        detections = [{'bbox': (10, 10, 10 + n, 10 + n)} for n in (10, 50, 30, 20, 40)]
        original = detections.copy()
        with patch.object(config, 'MAX_FACES_PER_FRAME', 3):
            selected = limit_faces(detections)
            self.assertEqual(limit_faces(detections[:2]), detections[:2])
        self.assertEqual([d['bbox'][2] - 10 for d in selected], [50, 40, 30])
        self.assertEqual(detections, original)

    def test_save_native_and_model_crop_without_relabelling_spoof(self):
        frame = np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        original = frame.copy()
        detection = {'bbox': np.array([80, 60, 180, 160]), 'score': np.float32(.99)}
        predictor = Mock(bbox_expansion_factor=1.5, model_img_size=80,
                         crop_mode='minifasnet_train_v1', model_path=Path('model.onnx'),
                         logit_threshold=1.65)
        predictor.predict_crops.return_value = [{'is_real': False, 'pad_score': np.float32(-2)}]
        with tempfile.TemporaryDirectory() as output:
            capture = PADCapture(output, 'person01', 'backlight', 'session01')
            paths = capture.save(frame, [(detection, SimpleNamespace(track_id=1))], predictor)
            self.assertEqual(len(paths), 1)
            self.assertEqual(cv2.imread(str(paths[0])).shape, (151, 151, 3))
            meta = json.loads(next(capture.directory.glob('*.json')).read_text())
            self.assertIsNone(meta['label'])
            self.assertEqual(meta['subject_id'], 'person01')
            self.assertEqual(meta['condition'], 'backlight')
            self.assertFalse(meta['capture_pad_result']['is_real'])
            saved_input = cv2.imread(str(capture.directory / meta['input_file']))
            expected = crop_for_runtime(frame, detection['bbox'], 1.5, 80, 'minifasnet_train_v1')
            np.testing.assert_array_equal(saved_input, expected)
            np.testing.assert_array_equal(frame, original)
            self.assertTrue(meta['already_expanded'])
            self.assertFalse(meta['lighting_adjusted'])
            predictor.predict_crops.return_value = []
            self.assertEqual(len(capture.save(frame, [(detection, SimpleNamespace(track_id=1))], predictor)), 1)
