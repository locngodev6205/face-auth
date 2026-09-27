"""Offline validation of embedded notebook code; no data downloads or training.
Run: python3 antispoof/notebooks/validation/validate_e3_cross_dataset.py
"""
import ast
import glob
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import tempfile
import time
import unittest

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
E3 = ROOT / 'antispoof/notebooks/e3_spatial_frequency_concat'
CROSS = ROOT / 'antispoof/notebooks/cross_dataset'
TRAIN = E3 / 'antispoof_e3_spatial_frequency_concat_v1.ipynb'
TEST = E3 / 'antispoof_e3_heldout_test_v1.ipynb'
EXTERNAL = CROSS / 'antispoof_cross_dataset_e1_vs_e3_v1.ipynb'


def parsed(path):
    notebook = json.loads(path.read_text())
    source = '\n\n'.join('\n'.join(line for line in ''.join(cell['source']).splitlines()
                                   if not line.lstrip().startswith(('%', '!')))
                          for cell in notebook['cells'] if cell['cell_type'] == 'code')
    return ast.parse(source), source


def namespace(path):
    tree, _ = parsed(path)
    ns = dict(globals(), INPUT_SIZE=224, FREQ_DIM=64, BBOX_JITTER_P=.2,
              BBOX_JITTER_SCALE=(.95, 1.05), BBOX_JITTER_TRANSLATE=.05, DCT_EPS=1e-6,
              CELEBA_MEAN=[.5931, .4690, .4229], CELEBA_STD=[.2471, .2214, .2157],
              ORT_THREADS=2, FRAMES_PER_VIDEO=16, MIN_VALID_FRAMES_PER_VIDEO=1,
              IMAGE_SUFFIXES={'.jpg', '.png'}, FACE_BINS=[-np.inf, 48, 100, 140, 180, np.inf],
              FACE_LABELS=['<48', '48–99', '100–139', '140–179', '>=180'],
              PREDICTOR_MIN_LOGIT_THRESHOLD=float(np.log(1e-6/(1-1e-6))),
              PREDICTOR_MAX_LOGIT_THRESHOLD=float(-np.log(1e-6/(1-1e-6))))
    wanted = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
              and (not isinstance(n, ast.ClassDef) or n.name in
                   {'E3ConcatPAD', 'DepthwiseSeparableBlock', 'FrequencyBranch'})]
    for node in wanted:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DCT_SPEC'
                                           for t in n.targets):
            exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), 'exec'), ns)
    return ns


class NotebookChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = namespace(TRAIN)
        cls.cross = namespace(EXTERNAL)
        cls.test = namespace(TEST)
        torch.set_num_threads(2)

    def test_all_code_cells_parse(self):
        for path in [TRAIN, TEST, EXTERNAL]:
            n = json.loads(path.read_text())
            self.assertEqual(n['nbformat'], 4)
            for i, c in enumerate(n['cells']):
                if c['cell_type'] == 'code':
                    source = '\n'.join(l for l in ''.join(c['source']).splitlines()
                                       if not l.lstrip().startswith(('%', '!')))
                    compile(source, f'{path.name}:{i}', 'exec')
                    self.assertEqual(c['outputs'], [])

    def test_exact_frozen_e2_functions(self):
        frozen = ROOT/'antispoof/notebooks/e2_frequency_dct/e2_frequency_only_dct_train.ipynb'
        tree, _ = parsed(frozen)
        names = {'rgb_to_luminance_float', 'luminance_to_dct_map', 'rgb_to_frequency_tensor',
                 'resize_rgb_to_input', 'FrequencyBranch', 'DepthwiseSeparableBlock'}
        expected = {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
        for path in [TRAIN, TEST, EXTERNAL]:
            tree, _ = parsed(path)
            actual = {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
            for name in names - ({'FrequencyBranch', 'DepthwiseSeparableBlock'} if path != TRAIN else set()):
                self.assertEqual(actual[name], expected[name], name)

    def test_dct_spatial_input_and_forward_onnx(self):
        ns = self.train
        crop = np.random.default_rng(42).integers(0, 256, (83, 83, 3), dtype=np.uint8)
        rgb, frequency = ns['branch_inputs'](crop)
        self.assertEqual(tuple(rgb.shape), (3,224,224))
        self.assertEqual(tuple(frequency.shape), (1,224,224))
        self.assertTrue(torch.isfinite(frequency).all())
        self.assertAlmostEqual(float(frequency.mean()), 0, places=5)
        self.assertAlmostEqual(float(frequency.numpy().std()), 1, places=5)
        model = ns['E3ConcatPAD'](pretrained=False).eval()
        self.assertEqual(sum(p.numel() for p in model.frequency_branch.parameters()),8480)
        self.assertEqual(sum(p.numel() for p in model.parameters()),1125187)
        with torch.no_grad():
            self.assertEqual(tuple(model(rgb[None],frequency[None]).shape),(1,3))
        with self.assertRaises(ValueError):
            model(torch.zeros(2,3,224,224),torch.zeros(1,1,224,224))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.onnx'
            torch.onnx.export(model,(rgb[None],frequency[None]),str(p),input_names=['rgb_input','frequency_map'],output_names=['logits'],dynamic_axes={'rgb_input':{0:'batch'},'frequency_map':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
            onnx.checker.check_model(str(p));session=ns['cpu_session'](p)
            deltas=[]
            for batch in [1,2]:
                r=rgb[None].repeat(batch,1,1,1);f=frequency[None].repeat(batch,1,1,1)
                with torch.no_grad(): expected=model(r,f).numpy()
                actual=session.run(None,{'rgb_input':r.numpy(),'frequency_map':f.numpy()})[0]
                delta=float(np.abs(actual-expected).max());deltas.append(delta)
                self.assertLess(delta,1e-4)
            print('Offline random-init ONNX parity max |delta|:',max(deltas))

    def test_split_cache_fingerprints_and_overlap_guards(self):
        p=ROOT/'antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small'
        config=json.loads((p/'metadata/mnv3s_e1_preliminary_v5_3_edge_run_config.json').read_text())
        with np.load(p/'data_protocol/celeba_spoof_preliminary_v5_3_edge_mnv3_small_seed42.npz') as manifest:
            splits=[manifest[s+'_keys'].astype(str) for s in ['train','val','test']]
            self.train['verify_disjoint'](splits)
            for name,keys in zip(['train','val','test'],splits):
                self.assertEqual(self.train['split_fingerprint'](keys),config['split_fingerprints'][name])
            selected=list(dict.fromkeys(list(manifest['train_keys_requested'].astype(str))+list(splits[1])+list(splits[2])))
        payload=json.loads((p/'data_protocol/celeba_scrfd_bbox_cache_v5_3_preliminary_seed42.json').read_text())
        actual=self.train['selected_cache_fingerprint'](selected,payload['records'])
        self.assertEqual(actual,config['bbox_cache_selected_fingerprint'])
        changed=dict(payload['records']);key=selected[0];changed[key]={**changed[key],'crop_factor':9.0}
        self.assertNotEqual(self.train['selected_cache_fingerprint'](selected,changed),actual)
        with self.assertRaises(RuntimeError): self.train['verify_disjoint']([['Data/train/1/a'],['Data/train/1/a'],['Data/test/2/a']])
        with self.assertRaises(RuntimeError): self.train['verify_disjoint']([['Data/train/1/a'],['Data/train/1/b'],['Data/test/2/a']])

    def test_score_calibration_and_empty_size_bins(self):
        ns=self.train
        scores=ns['binary_pad_score_from_logits'](np.array([[2,0,0],[0,2,2]],float))
        self.assertGreater(scores[0],0);self.assertLess(scores[1],0)
        metric=ns['pad_metrics']([0,1],scores,0)
        self.assertEqual(metric['HTER'],0);self.assertEqual(metric['AUC'],1)
        cal=ns['calibrate_threshold_min_acer'](scores,np.array([0,1]))
        self.assertEqual(cal['ACER'],0)
        empty=ns['pad_metrics']([],[],0);self.assertIsNone(empty['HTER'])
        df=pd.DataFrame({'face_min_side':[47,48,99,100,139,140,179,180],'true_class':[0]*8,'d':[1.]*8})
        bins=ns['face_size_breakdown'](df,0)
        self.assertEqual(bins.N.tolist(),[1,2,2,2,1])
        self.assertTrue(bins.APCER.isna().all());self.assertTrue(bins.ACER.isna().all())

    def test_video_aggregation_zero_one_multiple_frames(self):
        ns=self.cross
        units=pd.DataFrame([{'dataset':'casia_fasd','unit_id':name,'video_id':name,'label':label,'candidate_frames':16,'kind':'video'} for name,label in [('zero',0),('one',1),('many',0)]])
        pred=pd.DataFrame([{'model':m,'dataset':'casia_fasd','unit_id':u,'d':d} for m in ['E1','E3'] for u,d in [('one',-.5),('many',-.5),('many',1.5)]])
        for m in ['E1','E3']:
            result=ns['aggregate_units'](units,pred,m,0).set_index('unit_id')
            self.assertEqual(result.loc['zero','status'],'NO_VALID_FACE')
            self.assertTrue(pd.isna(result.loc['zero','d']))
            self.assertTrue(pd.isna(result.loc['zero','pred_real']))
            self.assertEqual(result.loc['one','pred_real'],0)
            self.assertEqual(result.loc['many','d'],.5)
            self.assertEqual(result.loc['many','pred_real'],1)
            self.assertTrue(result.loc['one','low_coverage'])
        self.assertEqual(ns['uniform_indices'](1),[0])
        self.assertEqual(ns['uniform_indices'](100),np.linspace(0,99,16,dtype=int).tolist())
        self.assertEqual(ns['uniform_indices'](0),[])

    def test_adapter_labels_provenance_and_image_video_semantics(self):
        ns=self.cross
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ['evaluation/real','evaluation/spoof']:
                (root/folder).mkdir(parents=True)
                cv2.imwrite(str(root/folder/'a.jpg'),np.zeros((8,8,3),np.uint8))
            cfg={'kind':'image','identity':'LCC-FASD','partition':'evaluation','version':'synthetic','source':'synthetic unit test','license':'synthetic','provenance_confirmed':True,'manifest_csv':'','folder_mapping':{'evaluation/real':0,'evaluation/spoof':1}}
            resolved=ns['resolve_dataset_manifest']('lcc_fasd',str(root),cfg)
            sampled,units=ns['sample_manifest'](resolved)
            self.assertEqual(len(sampled),2);self.assertEqual(len(units),2)
            self.assertTrue(sampled.video_id.eq('').all())
            with self.assertRaises(RuntimeError): ns['resolve_dataset_manifest']('lcc_fasd',str(root),{**cfg,'provenance_confirmed':False})
            with self.assertRaises(RuntimeError): ns['resolve_dataset_manifest']('lcc_fasd',str(root),{**cfg,'folder_mapping':{'evaluation/real':9}})
            with self.assertRaises(RuntimeError): ns['secure_path'](root,'../escape.jpg')

    def test_frozen_e1_runtime_loader_and_threshold_conflicts(self):
        ns=self.test
        p=ROOT/'antispoof/models/smartface_pad_artifacts/E1_v5_3_mnv3_small'
        bundle=ns['load_frozen_model']('E1',str(p),'preliminary')
        self.assertAlmostEqual(bundle['threshold'],-.4650222063064575)
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            for f in (p/'deployment').iterdir():
                if f.suffix in {'.onnx','.data'}: (d/f.name).symlink_to(f)
            cfg=json.loads((p/'deployment/mnv3s_e1_preliminary_v5_3_edge_runtime_config.json').read_text())
            meta=json.loads((p/'metadata/mnv3s_e1_preliminary_v5_3_edge_best_meta.json').read_text())
            meta['calibrated_logit_threshold']=99
            (d/'mnv3s_e1_preliminary_v5_3_edge_runtime_config.json').write_text(json.dumps(cfg))
            (d/'mnv3s_e1_preliminary_v5_3_edge_best_meta.json').write_text(json.dumps(meta))
            with self.assertRaises(RuntimeError): ns['load_frozen_model']('E1',str(d),'preliminary')

    def test_static_protocol_invariants(self):
        _,train=parsed(TRAIN);_,test=parsed(TEST);_,external=parsed(EXTERNAL)
        self.assertIn('INIT_MODE = "independent"',train)
        self.assertNotIn('test_loader=',train)
        self.assertNotIn('calibrate_threshold_min_acer',test)
        self.assertNotIn('calibrate_threshold_min_acer',external)
        self.assertNotIn('optimizer=',external)
        self.assertIn("common_ids=set(coverage.loc[coverage.status=='VALID','frame_id'])",external)
        self.assertIn("g.d.mean()",external)
        self.assertIn("current_app_compatible':False",train)
        for s in [train,test,external]:
            self.assertNotIn('gamma=True',s)
            self.assertNotIn('MobileNetV3-Large',s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
