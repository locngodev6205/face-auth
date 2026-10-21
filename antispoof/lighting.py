"""Optional lightweight PAD lighting transforms; no-op unless a profile is selected."""
import math
import cv2
import numpy as np


class LightingTransform:
    def __init__(self, profile=None):
        if profile is not None and not isinstance(profile, dict):
            raise ValueError('PAD lighting profile must be an object')
        self.profile = dict(profile or {'method': 'none'})
        self.method = self.profile.get('method', 'none')
        if self.method not in ('none', 'gamma', 'gamma_y', 'gamma_clahe_y', 'adaptive_gamma', 'legacy_gamma', 'gain_y', 'clahe_y', 'stretch_y'):
            raise ValueError(f'Unknown PAD lighting method: {self.method}')
        for key, value in self.profile.items():
            if key != 'method' and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'Invalid lighting parameter: {key}')
        p = self.profile
        if self.method in ('gamma', 'gamma_y', 'gamma_clahe_y') and not 0 < p.get('gamma', 0) <= 4:
            raise ValueError('PAD gamma must be in (0,4]')
        if not 0 < p.get('target', 110) < 255 or p.get('deadband', 10) < 0:
            raise ValueError('Invalid lighting target/deadband')
        for lo, hi in (('min_gamma', 'max_gamma'), ('min_gain', 'max_gain')):
            if not 0 < p.get(lo, .7) <= p.get(hi, 1.3):
                raise ValueError(f'Invalid lighting bounds: {lo}/{hi}')
        if not 0 <= p.get('blend', 1) <= 1 or p.get('clip', 1) <= 0:
            raise ValueError('Invalid lighting blend/clip')
        if not 0 <= p.get('low_percentile', 1) < p.get('high_percentile', 99) <= 100 or p.get('max_gain', 1.5) <= 0:
            raise ValueError('Invalid lighting percentile/gain')
        self.luts = {}
        self.clahe = None
        if self.method in ('clahe_y', 'gamma_clahe_y'):
            tile = int(self.profile.get('tile', 4))
            if tile != self.profile.get('tile', 4) or not 1 <= tile <= 16:
                raise ValueError('PAD CLAHE tile must be an integer in [1,16]')
            self.clahe = cv2.createCLAHE(self.profile.get('clip', 1.5), (tile, tile))

    def gamma(self, image, gamma):
        gamma = round(float(gamma), 2)
        if gamma not in self.luts:
            self.luts[gamma] = np.clip((np.arange(256) / 255.) ** gamma * 255., 0, 255).astype(np.uint8)
        return cv2.LUT(image, self.luts[gamma])

    def __call__(self, image):
        p, method = self.profile, self.method
        if method == 'none':
            return image
        if method == 'legacy_gamma':
            from .preprocess import adaptive_gamma
            return adaptive_gamma(image)
        if method == 'gamma':
            return self.gamma(image, p['gamma'])
        if method == 'gamma_clahe_y':
            image = self.gamma(image, p['gamma'])
        ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]
        if method == 'gamma_y':
            ycc[:, :, 0] = self.gamma(y, p['gamma'])
            return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        if method == 'adaptive_gamma':
            mean = float(y.mean())
            target = p.get('target', 110.)
            if abs(mean - target) <= p.get('deadband', 10.):
                return image
            gamma = math.log(target / 255.) / math.log(np.clip(mean, 1., 254.) / 255.)
            return self.gamma(image, np.clip(gamma, p.get('min_gamma', .7), p.get('max_gamma', 1.3)))
        if method == 'gain_y':
            gain = np.clip(p.get('target', 110.) / max(float(y.mean()), 1.),
                           p.get('min_gain', .8), p.get('max_gain', 1.3))
            adjusted = np.clip(y.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        elif method in ('clahe_y', 'gamma_clahe_y'):
            adjusted = self.clahe.apply(y)
        else:
            lo, hi = np.percentile(y, [p.get('low_percentile', 1), p.get('high_percentile', 99)])
            if hi - lo < 1:
                return image
            gain = min(255. / (hi - lo), p.get('max_gain', 1.5))
            adjusted = np.clip((y.astype(np.float32) - lo) * gain, 0, 255).astype(np.uint8)
        blend = p.get('blend', 1.)
        ycc[:, :, 0] = cv2.addWeighted(y, 1 - blend, adjusted, blend, 0)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


class BrightnessThreshold:
    """Bounded input-luminance offset; base remains controlled by runtime/env."""
    def __init__(self, profile=None):
        if profile is not None and not isinstance(profile, dict):
            raise ValueError('PAD adaptive_threshold must be an object')
        self.profile = profile
        if profile is not None:
            required = {'reference_luma', 'scale', 'slope', 'max_offset'}
            if set(profile) != required or not all(
                isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                for v in profile.values()
            ):
                raise ValueError('Invalid adaptive threshold parameters')
            if not 0 <= profile['reference_luma'] <= 255 or profile['scale'] <= 0 or profile['max_offset'] < 0:
                raise ValueError('Invalid adaptive threshold bounds')

    def __call__(self, crop, base_threshold):
        if self.profile is None:
            return float(base_threshold)
        p = self.profile
        mean = float(cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)[:, :, 0].mean())
        offset = np.clip(p['slope'] * (mean - p['reference_luma']) / p['scale'],
                         -p['max_offset'], p['max_offset'])
        return float(base_threshold + offset)
