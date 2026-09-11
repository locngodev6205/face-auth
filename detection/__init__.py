# FaceDetector mặc định là SCRFD (InsightFace) — xử lý ảnh bất kỳ kích thước, dùng cho enroll/eval.
# YunNet dùng riêng cho edge inference (app.py import thẳng từ detection.yunnet_detector).
from .detector import FaceDetector, FaceDetector as SCRFDFaceDetector
from .yunnet_detector import FaceDetector as YunNetFaceDetector

__all__ = ["FaceDetector", "SCRFDFaceDetector", "YunNetFaceDetector"]
