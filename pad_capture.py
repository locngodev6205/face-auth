"""Unlabelled PAD crops from clean camera frames; no alignment or lighting edits."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
import config
from antispoof.preprocess import crop_for_runtime, crop_minifasnet_train_v1
from vision_pipeline import _pad_face_skip_reason


class PADCapture:
    def __init__(self, output, subject, condition, session=None):
        self.directory = Path(output).expanduser() / (session or uuid.uuid4().hex)
        self.subject = subject
        self.condition = condition

    def save(self, frame, tracked, predictor):
        if predictor is None:
            raise ValueError('Cần model PAD để lưu đúng crop và threshold.')
        batch = uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)
        saved = []
        for detection, track in tracked[:config.MAX_FACES_PER_FRAME]:
            bbox = detection['bbox']
            factor, size, mode = (predictor.bbox_expansion_factor,
                                  predictor.model_img_size, predictor.crop_mode)
            model_crop = crop_for_runtime(frame, bbox, factor, size, mode)
            native = (crop_minifasnet_train_v1(frame, bbox, factor, None)
                      if mode == 'minifasnet_train_v1' else model_crop)
            try:
                results = predictor.predict_crops([model_crop])
                result = results[0] if results else {'error': 'empty PAD result'}
            except Exception as exc:
                result = {'error': str(exc)}
            stem = f'{batch}_track{track.track_id}'
            crop_path = self.directory / f'{stem}_crop.png'
            input_path = self.directory / f'{stem}_input.png'
            gray = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY)
            metadata = {
                'schema_version': 1, 'captured_at': datetime.now(timezone.utc).isoformat(),
                'capture_batch': batch, 'session_id': self.directory.name,
                'subject_id': self.subject, 'condition': self.condition,
                'label': None, 'label_options': ['real', 'fake'],
                'track_id': track.track_id, 'bbox_xyxy': list(bbox),
                'detector_score': detection.get('score'),
                'landmarks': detection.get('landmarks'),
                'frame_shape': list(frame.shape), 'crop_shape': list(native.shape),
                'bbox_expansion_factor': factor, 'crop_mode': mode, 'model_img_size': size,
                'already_expanded': True, 'aligned': False, 'lighting_adjusted': False,
                'crop_smoothing_applied': False,
                'validity_skip_reason': _pad_face_skip_reason(detection, frame.shape),
                'model_path': str(predictor.model_path),
                'runtime_config': config.PAD_RUNTIME_CONFIG_PATH,
                'model_sha256': config.PAD_RUNTIME_MODEL_SHA256,
                'threshold_logit': predictor.logit_threshold,
                'runtime_gamma_enabled': config.PAD_GAMMA_ENABLED,
                'gamma_target': config.PAD_GAMMA_TARGET,
                'color_order': config.PAD_COLOR_ORDER,
                'mean': config.PAD_MEAN, 'std': config.PAD_STD,
                'capture_pad_result': result,
                'luma_mean': float(gray.mean()),
                'dark_fraction': float(np.mean(gray <= 10)),
                'bright_fraction': float(np.mean(gray >= 245)),
                'crop_file': crop_path.name, 'input_file': input_path.name,
            }
            # Model verdict is diagnostic only; label remains null even for SPOOF.
            for path, pixels in ((crop_path, native), (input_path, model_crop)):
                if not cv2.imwrite(str(path), pixels):
                    raise OSError(f'Không ghi được {path}')
            def serializable(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)
            (self.directory / f'{stem}.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=serializable),
                encoding='utf-8',
            )
            saved.append(crop_path)
        return saved
