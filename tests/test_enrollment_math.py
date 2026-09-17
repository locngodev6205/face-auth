"""Unit tests for enrollment centroid L2-normalization and outlier filtering."""

import unittest
import numpy as np

from enrollment.enroll import build_identity_embedding


class TestEnrollmentMath(unittest.TestCase):

    def test_centroid_l2_normalized(self):
        """Centroid được L2-normalize."""
        # 3 identical unit vectors
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        embeddings = [v.copy() for _ in range(3)]

        centroid, warnings, valid = build_identity_embedding(embeddings, outlier_threshold=0.35)
        self.assertEqual(len(warnings), 0)
        self.assertEqual(len(valid), 3)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=6)
        self.assertAlmostEqual(float(centroid[0]), 1.0, places=6)

    def test_clear_outlier_is_filtered(self):
        """Outlier rõ ràng bị loại."""
        # 4 close vectors along axis 0
        v_main = np.zeros(512, dtype=np.float32)
        v_main[0] = 1.0

        v1 = v_main.copy()
        v2 = v_main.copy(); v2[1] = 0.1; v2 /= np.linalg.norm(v2)
        v3 = v_main.copy(); v3[1] = -0.1; v3 /= np.linalg.norm(v3)
        v4 = v_main.copy(); v4[2] = 0.1; v4 /= np.linalg.norm(v4)

        # 1 outlier orthogonal to v_main (dot product 0.0 < 0.35)
        v_outlier = np.zeros(512, dtype=np.float32)
        v_outlier[50] = 1.0

        embeddings = [v1, v2, v3, v4, v_outlier]
        centroid, warnings, valid = build_identity_embedding(embeddings, outlier_threshold=0.35)

        # Outlier must be dropped
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Loại 1/5", warnings[0])
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=6)
        self.assertAlmostEqual(float(centroid[50]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
