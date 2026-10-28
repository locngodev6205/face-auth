"""Offline notebook-contract and tiny CPU checks; no dataset/model training."""
import ast
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import zipfile
import unittest
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'md/source_of_truth/notebooks/22_fullscale_R7_harmful_gate_ablation_crop15.ipynb'


def notebook_namespace(seed=101):
    book = json.loads(NOTEBOOK.read_text())
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=__import__('math'),
              random=random, Counter=Counter, Path=Path)
    config = ''.join(book['cells'][1]['source'])
    if seed != 101:
        config = config.replace('TRAIN_SEED=101 ', f'TRAIN_SEED={seed} ')
    exec(compile(config, str(NOTEBOOK), 'exec'), ns)
    ns.update(device=torch.device('cpu'), use_amp=False, GAIN_RANGE=(.65, .90),
              GAIN_SEED=seed + 11000, _BBOX_RNG=random.Random(seed + 500000),
              MANIFEST_HASHES={'full_train.csv': 'train-sha', 'full_test.csv': 'test-sha'},
              INITIAL_STATE_SHA='initial-sha', root=Path('/offline/resources'),
              pretrained=Path('/offline/pretrained.pth'), source=Path('/offline/MiniFASNet.py'),
              AUGMENTATION_API='legacy_limits', A=SimpleNamespace(__version__='2.0.8'))
    needed = {'radial_masks', 'spectral_views', 'aligned_margin', 'auxiliary_loss',
              'select_worst_view', 'training_terms', 'snapshot_rng', 'restore_rng',
              'bbox_jitter', 'run_config', 'assert_fairness_contract', 'advance_patience',
              'patience_stop_reached', 'phase_patience_sanity_checks', 'mechanism_sanity_checks',
              'result_row', 'value_text', 'export_summary', 'export_run_zip', 'export_overall_zip',
              'evaluate_casia', 'casia_evaluation_contract', 'casia_required_outputs',
              'run_casia_for_completed_run', 'sha', 'mini_crop_bgr', 'metrics'}
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in needed:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
    return book, ns


class R7GateAblationNotebookTests(unittest.TestCase):
    def test_seed_roots_selection_and_notebook_cleanliness(self):
        roots = []
        for seed in (100, 101, 102):
            book, ns = notebook_namespace(seed)
            roots.append(ns['OUTPUT_ROOT'])
            self.assertTrue(ns['OUTPUT_ROOT'].endswith(f'_s{seed}'))
            self.assertEqual(ns['select_runs'](ns['RUN_FILTER']), ['R7_SC_NOGATE_FULL_crop15'])
            self.assertEqual(len(ns['select_runs'](None)), 4)
            self.assertEqual(ns['DATA_SEED'], 43)
            self.assertEqual(ns['RUNS'][ns['ABLATION_RUN_ID']],
                             dict(ns['RUNS']['R7_SC_FULL_crop15'], method='WORST'))
            for cell in book['cells']:
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
        self.assertEqual(len(set(roots)), 3)

    def test_fairness_contract_accepts_only_gate_policy_change(self):
        _, ns = notebook_namespace()
        configs = {name: ns['run_config'](name, spec) for name, spec in ns['RUNS'].items()}
        hashes = dict.fromkeys(ns['RUNS'], ns['INITIAL_STATE_SHA'])
        self.assertTrue(ns['assert_fairness_contract'](configs, hashes))
        a = configs[ns['ABLATION_RUN_ID']]
        self.assertEqual(a['spectral']['sigma'], .10)
        self.assertEqual(a['ablation']['reference_run_id'], 'R7_SC_FULL_crop15')
        self.assertTrue(a['ablation']['full_selected_batch_forward'])
        old = ns['RUNS'][ns['ABLATION_RUN_ID']]
        for key, bad in [('sigma', 1/18), ('clean_ce_weight', 1.), ('warmup_epochs', 0),
                         ('centers', [0.1, 0.2, 0.3])]:
            ns['RUNS'][ns['ABLATION_RUN_ID']] = dict(old, **{key: bad})
            with self.assertRaises(RuntimeError):
                ns['assert_fairness_contract'](configs, hashes)
        ns['RUNS'][ns['ABLATION_RUN_ID']] = old
        broken = copy.deepcopy(configs)
        broken[ns['ABLATION_RUN_ID']]['train_seed'] = 100
        with self.assertRaises(RuntimeError):
            ns['assert_fairness_contract'](broken, hashes)
        with self.assertRaises(ValueError):
            ns['select_runs'](['R7_SC_NOGATE_FULL_crop15', 'unknown'])

    def test_exports_keep_pending_comparators_and_seeded_zip_names(self):
        _, ns = notebook_namespace(102)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'r7_ablation_s102'
            (out / 'manifests').mkdir(parents=True)
            (out / 'manifests/full_train.csv').write_text('sample_key\nsynthetic\n')
            name = ns['ABLATION_RUN_ID']
            folder = out / 'runs' / name
            (folder / 'metrics').mkdir(parents=True)
            splits = ('test', 'lcc_training', 'lcc_development', 'lcc_evaluation', 'lcc_combined')
            metric = dict(auc=.8, eer=.2, apcer=.1, bpcer=.3, acer=.2, hter=.2,
                          binary_accuracy=.8, global_argmax_accuracy=.8, tpr_at_fpr_1pct=.1,
                          candidate_n=10, scored_n=10, detector_coverage=1.)
            def write(path, value):
                Path(path).write_text(json.dumps(value))
            write(folder / 'completion.json', {'best_epoch': 2})
            for split in splits:
                write(folder / 'metrics' / (split + '_metrics.json'), metric)
                write(folder / 'metrics' / (split + '_reference_metrics.json'), metric)
            ns.update(out=out, read_json=lambda path: json.loads(Path(path).read_text()),
                      write_json=write, pd=__import__('pandas'), json=json, zipfile=zipfile,
                      FileLink=str, display=lambda value: None, DATA_COUNTS={'synthetic': True},
                      SCORE_SPLITS=splits, TABLE_COLUMNS=('selected_epoch', 'lcc_combined_auc'),
                      SCALE100K_REFERENCES={'C': .7}, REFERENCE_BENCHMARKS={'rows': []},
                      COMPLEXITY={'shared_clean_inference': dict(params_m=.434434,
                          total_params=441658, trainable_params=441658, macs=40810892,
                          gflops=.081621784)})
            rows = ns['export_summary']()
            self.assertEqual(sum(row['status'] == 'complete' for row in rows), 1)
            report = json.loads((out / 'fullscale_project_metrics.json').read_text())
            self.assertIsNone(report['pairwise_combined_deltas']['R7_minus_R7_NOGATE'])
            self.assertEqual(set(report['runs']), {name})
            with contextlib.redirect_stdout(io.StringIO()):
                ns['export_run_zip'](name)
                ns['export_overall_zip']()
            self.assertTrue((out.parent / (name + '_s102.zip')).is_file())
            self.assertTrue((out.parent / (out.name + '.zip')).is_file())
            self.assertFalse((out.parent / '18_fullscale_C_P3SF_R7SC_crop15.zip').exists())

    def test_cpu_mechanism_and_phase_checks(self):
        _, ns = notebook_namespace()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter('ignore', FutureWarning)
                result = ns['mechanism_sanity_checks']()
                phase = ns['phase_patience_sanity_checks']()
            self.assertTrue(all(result.values()))
            self.assertTrue(result['R7_no_gate_same_sigma_and_all_sample_CE'])
            self.assertTrue(result['R7_no_gate_preserves_full_batch_BN'])
            self.assertTrue(phase)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
