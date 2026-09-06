"""
Package Anti-Spoofing (Chống giả mạo khuôn mặt) cho hệ thống SmartFace AIoT.

Cung cấp các lớp và hàm API chính:
    - AntiSpoofPredictor: Lớp suy luận chính kiểm tra khuôn mặt Thật (Real) hay Giả (Spoof).
    - load_model: Nạp mô hình ONNX Runtime với tự động phát hiện GPU/CPU.
    - crop, preprocess, preprocess_batch: Các tiện ích crop khuôn mặt và tiền xử lý ảnh.
    - get_cpu_info, get_gpu_info, get_execution_provider_name: Tiện ích đọc thông tin phần cứng hệ thống.
"""

from .predictor import AntiSpoofPredictor
from .loader import load_model
from .preprocess import crop, preprocess, preprocess_batch
from .system import get_cpu_info, get_gpu_info, get_execution_provider_name

__all__ = [
    "AntiSpoofPredictor",
    "load_model",
    "crop",
    "preprocess",
    "preprocess_batch",
    "get_cpu_info",
    "get_gpu_info",
    "get_execution_provider_name",
]
