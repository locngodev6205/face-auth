"""Synthetic CASIA integration checks: fake detector, toy scores, no real PAD/ONNX inference."""
import ast
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import time
import unittest
import zipfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from tests.test_r7_gate_ablation_notebook import notebook_namespace, NOTEBOOK


class ToyPAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        score = (.5 - x.mean((1, 2, 3))) * self.scale
        return torch.stack((score, -score), dim=1)


class FakeDetector:
    def __init__(self):
        self.session = SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def prepare(self, **kwargs):
        pass

    def detect(self, image, **kwargs):
        # Failed preferred test copy must never fall back to its valid train duplicate.
        if int(image[0, 0, 0]) == 245:
            return np.empty((0, 5)), None
        return np.array([[2., 2., 17., 17., .95]]), None


def fixture(temp, calibration):
    _, ns = notebook_namespace(101)
    root = Path(temp)
    data = root / 'casia_input'
    for split in ('train', 'test'):
        folder = data / f'{split}_img/{split}_img/color'
        folder.mkdir(parents=True)
        for code, label, frames in [('1', 'real', range(1, 6) if split == 'train' else range(2, 7)),
                                    ('2', 'real', range(1, 4)), ('3', 'fake', range(1, 6))]:
            for frame in frames:
                value = 200 if label == 'fake' else (40 if split == 'train' else 60)
                if split == 'test' and code == '1' and frame == 2:
                    value = 245
                image = np.full((20, 20, 3), value, dtype=np.uint8)
                assert cv2.imwrite(str(folder / f'1_{code}.avi_{frame}_{label}.jpg'), image)
    detector = root / 'mock_detector.onnx'
    detector.write_bytes(b'SYNTHETIC detector identity, never loaded by ONNX Runtime')
    out = root / 'r7_ablation_s101'
    name = ns['ABLATION_RUN_ID']
    folder = out / 'runs' / name
    folder.mkdir(parents=True)
    cfg = ns['run_config'](name, ns['RUNS'][name])
    (folder / 'config.json').write_text(json.dumps(cfg))
    torch.save({'run_id': name, 'epoch': 2, 'config': cfg, 'state_dict': ToyPAD().state_dict()}, folder / 'best.pth')
    digest = hashlib.sha256((folder / 'best.pth').read_bytes()).hexdigest()
    (folder / 'evaluation_freeze.json').write_text(json.dumps(dict(run_id=name, best_epoch=2,
        locked_threshold=.0, checkpoint_sha256=digest)))
    (folder / 'completion.json').write_text(json.dumps(dict(run_id=name, best_epoch=2, locked_threshold=.0)))
    ns.update(out=out, CASIA_ROOT=str(data), SCRFD_MODEL_PATH=str(detector),
              CALIBRATE_CASIA_THRESHOLD=calibration, re=re, cv2=cv2, pd=pd, json=json,
              gc=gc, time=time, zipfile=zipfile, plt=plt, hashlib=hashlib,
              ort=SimpleNamespace(__version__='synthetic'),
              model_zoo=SimpleNamespace(get_model=lambda *a, **kw: FakeDetector()),
              roc_curve=roc_curve, roc_auc_score=roc_auc_score,
              display=lambda value: None, FileLink=str,
              read_json=lambda path: json.loads(Path(path).read_text()),
              write_json=lambda path, value: Path(path).write_text(json.dumps(value, allow_nan=False)),
              init_model=lambda spec: (ToyPAD(), {}))
    return ns, folder, name


class R7CasiaNotebookTests(unittest.TestCase):
    def test_source_frozen_eval_dedup_video_mean_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ns, folder, name = fixture(tmp, calibration=False)
            original_best = (folder / 'best.pth').read_bytes()
            self.assertTrue(ns['run_casia_for_completed_run'](name))
            casia = folder / 'casia'
            self.assertTrue(all(p.is_file() for p in ns['casia_required_outputs'](folder)))
            self.assertFalse((casia / 'casia_calibrated_threshold_summary.json').exists())
            frames = pd.read_csv(casia / 'combined/frame_predictions.csv')
            videos = pd.read_csv(casia / 'combined/video_predictions.csv')
            self.assertFalse(frames.duplicated(['video_id', 'frame_index']).any())
            # Train copy of this failed test frame was detected, but must not enter Combined.
            self.assertFalse(((frames.video_id == '1_1.avi') & (frames.frame_index == 2)).any())
            self.assertEqual(len(frames), 13)
            self.assertEqual(len(videos), 3)
            self.assertEqual(int(videos.scorable.sum()), 2)
            unscorable = videos[videos.video_id == '1_2.avi'].iloc[0]
            self.assertFalse(unscorable.scorable)
            self.assertTrue(pd.isna(unscorable.predicted_real))
            means = frames.groupby('video_id').pad_score.mean()
            for _, row in videos.iterrows():
                self.assertAlmostEqual(row.video_score, means[row.video_id], places=6)
            metrics = json.loads((casia / 'combined/video_metrics.json').read_text())
            self.assertEqual(metrics['candidate_n'], 3)
            self.assertEqual(metrics['scored_n'], 2)
            self.assertEqual(metrics['locked_source_threshold'], .0)
            config = json.loads((casia / 'config.json').read_text())
            self.assertEqual(config['train_seed'], 101)
            self.assertIn('all_selected_worst', config['loss'])
            self.assertEqual(config['checkpoint_sha256'], hashlib.sha256(original_best).hexdigest())
            self.assertFalse(ns['run_casia_for_completed_run'](name))
            self.assertEqual((folder / 'best.pth').read_bytes(), original_best)
            # ZIPs must retain reports/predictions but exclude the large shared crop cache.
            ns['export_run_zip'](name)
            ns['export_overall_zip']()
            for path in [ns['out'].parent / (name + '_s101.zip'), ns['out'].parent / (ns['out'].name + '.zip')]:
                with zipfile.ZipFile(path) as archive:
                    self.assertFalse(any(n.endswith('shared_crops_bgr_uint8.npy') for n in archive.namelist()))
                    self.assertTrue(any(n.endswith('combined/video_metrics.json') for n in archive.namelist()))
            (casia / 'combined/video_predictions.csv').unlink()
            with self.assertRaisesRegex(RuntimeError, 'outputs incomplete'):
                ns['run_casia_for_completed_run'](name)

    def test_optional_train_only_calibration_separate_and_settings_guarded(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ns, folder, name = fixture(tmp, calibration=True)
            self.assertTrue(ns['run_casia_for_completed_run'](name))
            casia = folder / 'casia'
            for unit in ('frame', 'video'):
                calibration = json.loads((casia / f'{unit}_casia_calibration.json').read_text())
                self.assertEqual(calibration['fit_split'], 'train')
                for split in ('train', 'test', 'combined'):
                    base = json.loads((casia / split / f'{unit}_metrics.json').read_text())
                    tuned = json.loads((casia / split / f'{unit}_metrics_casia_calibrated.json').read_text())
                    self.assertEqual(base['auc'], tuned['auc'])
                    self.assertEqual(base['eer'], tuned['eer'])
                    self.assertEqual(base['locked_source_threshold'], .0)
                    self.assertTrue((casia / split / f'{unit}_predictions_casia_calibrated.csv').exists())
            ns['CALIBRATE_CASIA_THRESHOLD'] = False
            with self.assertRaisesRegex(RuntimeError, 'different checkpoint/settings'):
                ns['run_casia_for_completed_run'](name)
            ns['CALIBRATE_CASIA_THRESHOLD'] = True
            with (folder / 'best.pth').open('ab') as h:
                h.write(b'mutated checkpoint')
            with self.assertRaisesRegex(RuntimeError, 'differs from frozen'):
                ns['run_casia_for_completed_run'](name)

    def test_notebook20_algorithm_ast_alignment(self):
        other = NOTEBOOK.with_name('20_casia_fasd_train_test_C_P3SF_R7SC_100k_fullscale.ipynb')
        def tree(path):
            n = json.loads(path.read_text())
            return ast.parse('\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code'))
        a, b = tree(other), tree(NOTEBOOK)
        names = ('mini_crop_bgr', 'metrics', 'read_split', 'combined_unique',
                 'video_manifest', 'detect_boxes', 'safe_metrics', 'optimize_casia_threshold')
        for name in names:
            with self.subTest(function=name):
                x = next(n for n in ast.walk(a) if isinstance(n, ast.FunctionDef) and n.name == name)
                y = next(n for n in ast.walk(b) if isinstance(n, ast.FunctionDef) and n.name == name)
                self.assertEqual(ast.dump(x), ast.dump(y))
        x = next(n for n in ast.walk(a) if isinstance(n, ast.ClassDef) and n.name == 'PADNet')
        y = next(n for n in ast.walk(b) if isinstance(n, ast.ClassDef) and n.name == 'PADNet')
        self.assertEqual(ast.dump(x), ast.dump(y))
        for name in ('EXPECTED_CODES', 'FILENAME_RE'):
            def assignment(t):
                return next(n for n in ast.walk(t) if isinstance(n, ast.Assign)
                            and any(isinstance(v, ast.Name) and v.id == name for v in n.targets))
            self.assertEqual(ast.dump(assignment(a)), ast.dump(assignment(b)))

    def test_notebook20_end_to_end_parity_on_same_synthetic_inputs(self):
        other = NOTEBOOK.with_name('20_casia_fasd_train_test_C_P3SF_R7SC_100k_fullscale.ipynb')
        book20 = json.loads(other.read_text())
        for calibration in (False, True):
            with self.subTest(calibration=calibration), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                ns, folder, name = fixture(tmp, calibration=calibration)
                ns['run_casia_for_completed_run'](name)
                output22 = folder / 'casia'
                output20 = Path(tmp) / 'standalone20'
                output20.mkdir()
                info = dict(run_id=name, method='synthetic comparator', scale='full-scale',
                            best_epoch=2, threshold=.0, amp=False, train_seed=101,
                            input='80x80 BGR float [0,1] crop1.5')
                # Execute the unmodified notebook20 evaluation cells with identical mock inputs.
                # Its run-discovery cell is replaced by an in-memory toy checkpoint loader only.
                ns20 = dict(ns)
                ns20.update(OUTPUT_ROOT=output20, CASIA=Path(ns['CASIA_ROOT']),
                            detector_path=Path(ns['SCRFD_MODEL_PATH']), SEED=100,
                            SELECTED_RUNS=[name], LABELS={name: 'synthetic comparator'},
                            provenance=[info], load_run=lambda run_id: (ToyPAD().eval(), dict(info)))
                for index in (6, 7, 8, 9, 10):
                    exec(compile(''.join(book20['cells'][index]['source']),
                                 str(other) + '#cell' + str(index), 'exec'), ns20)
                run20 = output20 / 'runs' / name
                for split in ('train', 'test', 'combined'):
                    for unit in ('frame', 'video'):
                        a = json.loads((run20 / split / f'{unit}_metrics.json').read_text())
                        b = json.loads((output22 / split / f'{unit}_metrics.json').read_text())
                        self.assertEqual(a, b)
                        a = pd.read_csv(run20 / split / f'{unit}_predictions.csv')
                        b = pd.read_csv(output22 / split / f'{unit}_predictions.csv')
                        pd.testing.assert_frame_equal(a, b)
                        if calibration:
                            a = json.loads((run20 / split / f'{unit}_metrics_casia_calibrated.json').read_text())
                            b = json.loads((output22 / split / f'{unit}_metrics_casia_calibrated.json').read_text())
                            self.assertEqual(a, b)
                            a = pd.read_csv(run20 / split / f'{unit}_predictions_casia_calibrated.csv')
                            b = pd.read_csv(output22 / split / f'{unit}_predictions_casia_calibrated.csv')
                            pd.testing.assert_frame_equal(a, b)
                for filename in ('casia_frame_detection.csv', 'casia_combined_frame_detection.csv',
                                 'casia_video_coverage.csv', 'casia_combined_frame_manifest.csv'):
                    pd.testing.assert_frame_equal(pd.read_csv(output20 / filename),
                                                  pd.read_csv(output22 / filename))

    def test_setup_and_runner_order(self):
        book = json.loads(NOTEBOOK.read_text())
        code = [''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code']
        setup = next(i for i, s in enumerate(code) if 'SCRFD preflight providers:' in s)
        source_setup = next(i for i, s in enumerate(code) if 'def run_one(' in s)
        evaluator = next(i for i, s in enumerate(code) if 'def evaluate_casia(' in s)
        runner = next(i for i, s in enumerate(code) if "selected=select_runs(RUN_FILTER)" in s)
        self.assertLess(setup, source_setup)
        self.assertLess(evaluator, runner)
        self.assertLess(code[runner].index('run_one(name'), code[runner].index('run_casia_for_completed_run(name)'))
        self.assertIn('finally:', code[runner])


if __name__ == '__main__':
    unittest.main()
