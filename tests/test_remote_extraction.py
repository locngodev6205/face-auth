"""Exact Pi preprocessing parity and protected API tests; never access a database."""
import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remote_extraction import AlignedFaceExtractor


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.model = tempfile.NamedTemporaryFile()
        self.addCleanup(self.model.close)
        self.model.write(b'test-model'); self.model.flush()
        self.session = Mock()
        self.session.get_inputs.return_value = [Mock(name='input')]
        self.session.get_inputs.return_value[0].name = 'input'
        self.session.run.return_value = [np.ones((1, 512), np.float32)]
        self.extractor = AlignedFaceExtractor(self.session, self.model.name, 'buffalo_s')
        self.face = np.random.default_rng(17).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        self.png = base64.b64encode(cv2.imencode('.png', self.face)[1]).decode()

    def extract(self, png=None, sha=None):
        return self.extractor.extract(png or self.png, sha or self.extractor.model_sha256, 'buffalo_s')

    def test_exact_hardware_preprocessing_and_normalized_output(self):
        result = self.extract()
        blob = self.session.run.call_args.args[1]['input']
        expected = ((self.face[:,:,::-1].astype(np.float32).transpose(2,0,1) - 127.5)/127.5)[None]
        np.testing.assert_array_equal(blob, expected)
        self.assertAlmostEqual(float(np.linalg.norm(np.frombuffer(base64.b64decode(result['embedding_b64']), '<f4'))), 1, places=6)

    def test_bad_images_dimensions_and_models_never_reach_onnx(self):
        for png, sha in [('bad!', None), (self.png, '0'*64),
                         (base64.b64encode(cv2.imencode('.png', np.zeros((224,224,3),np.uint8))[1]).decode(), None),
                         (base64.b64encode(cv2.imencode('.jpg', self.face)[1]).decode(), None),
                         ('x'*60001, None)]:
            with self.subTest(length=len(png)):
                with self.assertRaises(ValueError): self.extract(png, sha)
        self.session.run.assert_not_called()

    def test_nonfinite_and_degenerate_outputs_rejected(self):
        for vector in (np.zeros(512), np.ones(511), np.full(512, np.nan)):
            self.session.run.return_value = [vector]
            with self.assertRaises(RuntimeError): self.extract()

    def test_api_auth_busy_and_valid_request_without_startup_or_db(self):
        from fastapi.testclient import TestClient
        import api_service as api
        body = dict(request_id='a'*32, image_png_b64=self.png, model_sha256=self.extractor.model_sha256, model_version='buffalo_s')
        with patch.object(api, 'remote_extractor', self.extractor), patch.dict(os.environ, {'FACE_AUTH_INTERNAL_KEY':'k'*32}):
            client = TestClient(api.app)  # No lifespan/startup: no DB or model discovery.
            self.assertEqual(client.post('/internal/face-embedding', json=body).status_code, 401)
            headers = {'X-Face-Auth-Key':'k'*32}
            result = client.post('/internal/face-embedding', json=body, headers=headers)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['request_id'], 'a'*32)
            api.remote_slot.acquire()
            try: self.assertEqual(client.post('/internal/face-embedding', json=body, headers=headers).status_code, 429)
            finally: api.remote_slot.release()
            self.assertEqual(client.post('/internal/face-embedding', json={**body, 'model_sha256':'0'*64}, headers=headers).status_code, 422)
            self.assertEqual(client.post('/internal/face-embedding', json=body, headers=headers).status_code, 200)
        with patch.dict(os.environ, {'FACE_AUTH_INTERNAL_KEY':''}):
            self.assertEqual(client.post('/internal/face-embedding', json=body, headers=headers).status_code, 503)

    def test_actual_onnx_matches_hardware_on_identical_pixels(self):
        path = ROOT.parent/'hardware/models/w600k_mbf.onnx'
        if not path.exists(): self.skipTest('Existing hardware ArcFace artifact unavailable')
        spec = importlib.util.spec_from_file_location('pi_embedder', ROOT.parent/'hardware/recognition/embedder.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        local = module.FaceEmbedder(str(path))
        server = AlignedFaceExtractor(local.session, str(path), 'buffalo_s')
        # Same existing ONNX, same image, lossless PNG; no recognition labels/accuracy claim.
        for seed in range(3):
            face = np.random.default_rng(seed).integers(0, 256, (112,112,3), dtype=np.uint8)
            png = base64.b64encode(cv2.imencode('.png', face)[1]).decode()
            output = server.extract(png, server.model_sha256, 'buffalo_s')
            remote = np.frombuffer(base64.b64decode(output['embedding_b64']), '<f4')
            np.testing.assert_array_equal(local.extract(face), remote)
        self.assertEqual(server.model_sha256, '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f')


    def test_production_insightface_session_matches_existing_pi_model(self):
        pack = Path.home()/'.insightface/models/buffalo_s'
        if not all((pack/name).exists() for name in ('w600k_mbf.onnx', 'det_500m.onnx')):
            self.skipTest('Existing server model pack unavailable; do not download during tests')
        from recognition.embedder import FaceEmbedder
        server = FaceEmbedder(model_name='buffalo_s', ctx_id=-1)
        extractor = AlignedFaceExtractor(server._model.session, server._model.model_file, 'buffalo_s')
        spec = importlib.util.spec_from_file_location('pi_embedder', ROOT.parent/'hardware/recognition/embedder.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        local = module.FaceEmbedder(str(ROOT.parent/'hardware/models/w600k_mbf.onnx'))
        for seed in range(3):
            face = np.random.default_rng(seed).integers(0,256,(112,112,3),dtype=np.uint8)
            png = base64.b64encode(cv2.imencode('.png',face)[1]).decode()
            output = extractor.extract(png,extractor.model_sha256,'buffalo_s')
            np.testing.assert_array_equal(np.frombuffer(base64.b64decode(output['embedding_b64']),'<f4'),local.extract(face))


if __name__ == '__main__': unittest.main()
