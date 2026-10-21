"""Deterministic geometry and preprocessing checks for domain optimization."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from optimize_pad_domain import reconstruct_crop, fit, profiles
from antispoof.preprocess import crop_minifasnet_train_v1
from antispoof.lighting import LightingTransform, BrightnessThreshold
from antispoof.predictor import AntiSpoofPredictor
from tracking.tracker import Track


class DomainTests(unittest.TestCase):
    def test_adaptive_threshold_is_bounded_and_never_mutates_base(self):
        adapter = BrightnessThreshold(dict(reference_luma=110.,scale=50.,slope=3.,max_offset=2.))
        base = -1.815
        self.assertEqual(adapter(np.zeros((80,80,3),np.uint8),base),base-2)
        self.assertEqual(adapter(np.full((80,80,3),255,np.uint8),base),base+2)
        self.assertEqual(adapter(np.full((80,80,3),110,np.uint8),base),base)
        self.assertEqual(BrightnessThreshold()(None,base),base)
        predictor=AntiSpoofPredictor.__new__(AntiSpoofPredictor)
        predictor.logit_threshold=0.;predictor.threshold_probability=.5;predictor.output_classes=2
        self.assertFalse(predictor.process_with_logits([.5,0],1.)['is_real'])
        self.assertTrue(predictor.process_with_logits([.5,0])['is_real'])
        self.assertEqual(predictor.logit_threshold,0.)
        track=Track(1,(10,10,100,100))
        track.update_pad(True,.5,.1)
        self.assertEqual(track.last_pad_threshold,.1)
        track.reset_pad()
        self.assertIsNone(track.last_pad_threshold)

    def test_reconstructed_roi_exactly_matches_original_frame_crop(self):
        frame = np.random.default_rng(1).integers(0,256,(240,320,3),dtype=np.uint8)
        for bbox in ((80,60,180,160),(15,20,115,150),(170.5,100.5,290.5,200.5)):
            sample = {'metadata': dict(frame_shape=frame.shape,bbox_xyxy=bbox,bbox_expansion_factor=1.5),
                      'crop': crop_minifasnet_train_v1(frame,bbox,1.5,None),'file':'synthetic.png'}
            for factor in (1.,1.25,1.4,1.5):
                np.testing.assert_array_equal(reconstruct_crop(sample,factor),
                                              crop_minifasnet_train_v1(frame,bbox,factor,80))
            with self.assertRaisesRegex(ValueError,'Cannot expand'):
                reconstruct_crop(sample,1.6)

    def test_transforms_are_finite_uint8_and_preserve_source(self):
        original=np.random.default_rng(2).integers(0,256,(80,80,3),dtype=np.uint8)
        for profile in profiles()+profiles(True):
            transform=LightingTransform(profile)
            for image in (original,np.zeros_like(original),np.full_like(original,255)):
                before=image.copy()
                with np.errstate(divide='ignore',invalid='ignore'):
                    result=transform(image)
                    repeated=transform(image)
                self.assertEqual(result.dtype,np.uint8)
                self.assertEqual(result.shape,image.shape)
                np.testing.assert_array_equal(image,before)
                np.testing.assert_array_equal(result,repeated)
        self.assertIs(LightingTransform()(original),original)
        np.testing.assert_array_equal(LightingTransform({'method':'gamma','gamma':1})(original),original)

    def test_threshold_midpoint_preserves_predictions(self):
        result=fit([-2,-1,1,2],[0,0,1,1])
        self.assertEqual(result['threshold_logit'],0)
        self.assertEqual(result['ACER'],0)

    def test_invalid_profiles_fail(self):
        for profile in ([], {'method':'typo'}, {'method':'gamma','gamma':0},
                        {'method':'gamma','gamma':float('nan')},
                        {'method':'clahe_y','tile':0}, {'method':'clahe_y','blend':2}):
            with self.assertRaises(ValueError):
                LightingTransform(profile)
