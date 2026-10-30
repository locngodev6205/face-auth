"""Lossless aligned-face contract, matching hardware FaceEmbedder.extract exactly."""
import base64
import hashlib
import struct
import time

import cv2
import numpy as np


class AlignedFaceExtractor:
    def __init__(self, session, model_file, model_version):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        with open(model_file, "rb") as model:
            self.model_sha256 = hashlib.file_digest(model, "sha256").hexdigest()
        self.model_version = model_version

    def extract(self, image_png_b64, model_sha256, model_version):
        if model_sha256 != self.model_sha256 or model_version != self.model_version:
            raise ValueError("ArcFace model contract mismatch")
        if not isinstance(image_png_b64, str) or len(image_png_b64) > 60000:
            raise ValueError("Aligned PNG exceeds limit")
        raw = base64.b64decode(image_png_b64, validate=True)
        # Check dimensions before decoding, so a compressed image bomb is rejected.
        if (len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
                or struct.unpack(">II", raw[16:24]) != (112, 112)
                or raw[24:26] != bytes((8, 2))):
            raise ValueError("Expected RGB8 PNG 112x112")
        face = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if face is None or face.shape != (112, 112, 3):
            raise ValueError("Invalid aligned PNG")
        # Keep the existing Pi normalization exactly, independently of model-pack defaults.
        rgb = face[:, :, ::-1].astype(np.float32)
        blob = np.expand_dims((rgb.transpose((2, 0, 1)) - 127.5) / 127.5, axis=0).astype(np.float32)
        started = time.monotonic()
        vector = np.asarray(self.session.run(None, {self.input_name: blob})[0], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if vector.size != 512 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("Invalid ArcFace output")
        vector = np.asarray(vector / norm, dtype="<f4")
        return {"embedding_b64": base64.b64encode(vector.tobytes()).decode("ascii"),
                "model_sha256": self.model_sha256, "model_version": self.model_version,
                "inference_ms": round((time.monotonic() - started) * 1000, 2)}
