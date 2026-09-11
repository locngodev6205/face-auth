"""
Module Lớp dự đoán Anti-Spoofing (Face Anti-Spoofing Predictor Class).

Chức năng chính:
    - Quản lý mô hình MiniFASNet V2 SE ONNX được nén dạng INT8 (best_model_quantized.onnx).
    - Nhận danh sách các hình ảnh crop khuôn mặt (hoặc trực tiếp khung hình + bbox).
    - Tính toán giá trị logit (real_logit vs spoof_logit) để phân loại khuôn mặt THẬT (real) hay GIẢ (spoof).
"""

import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .loader import load_model
from .preprocess import preprocess_batch, crop

# Đường dẫn mặc định tới trọng số mô hình Anti-Spoofing nén INT8
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "best_model_quantized.onnx"


class AntiSpoofPredictor:
    """
    Lớp dự đoán chống giả mạo khuôn mặt (Anti-Spoofing Predictor).

    Sử dụng kiến trúc MiniFASNet V2 SE tối ưu hóa bằng ONNX Runtime để đạt tốc độ mượt mà trên CPU.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        model_img_size: int = 128,
        bbox_expansion_factor: float = 1.5,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        apply_gamma: bool = True,
    ):
        """
        Khởi tạo Predictor:

        Tham số:
            model_path (Optional[str]): Đường dẫn tới file .onnx (mặc định dùng best_model_quantized.onnx).
            threshold (float): Ngưỡng xác xuất quyết định Real/Spoof (mặc định 0.5).
            model_img_size (int): Kích thước ảnh vuông đầu vào của mô hình (mặc định 128x128).
            bbox_expansion_factor (float): Tỷ lệ mở rộng khung bao bbox khi crop mặt (mặc định 1.5x).
            mean (Optional[List[float]]): Giá trị mean chuẩn hóa kênh màu [R, G, B].
            std (Optional[List[float]]): Giá trị std chuẩn hóa kênh màu [R, G, B].
            apply_gamma (bool): Bật adaptive gamma correction trước khi normalize (mặc định: True).
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH    # lấy đường dẫn model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số Anti-Spoofing tại: '{self.model_path}'"
            )

        self.threshold = threshold
        self.model_img_size = model_img_size
        self.bbox_expansion_factor = bbox_expansion_factor
        self.apply_gamma = apply_gamma

        # Quy đổi ngưỡng xác xuất threshold sang ngưỡng logit difference (logit_threshold) bằng hàm Logit/Sigmoid ngược
        p = max(1e-6, min(1 - 1e-6, threshold)) # ép không chạm biên gây lỗi
        self.logit_threshold = np.log(p / (1 - p))      

        # Nạp mô hình ONNX qua hàm load_model
        self.session, self.input_name = load_model(str(self.model_path))
        if self.session is None:
            raise RuntimeError(f"Không thể khởi tạo ONNX InferenceSession từ '{self.model_path}'")

        # Tự động phát hiện kích thước ảnh đầu vào (model_img_size) trực tiếp từ đồ thị mô hình ONNX
        try:
            input_shape = self.session.get_inputs()[0].shape
            # input_shape thường có dạng (batch_size, 3, height, width) hoặc [None, 3, H, W]
            if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
                self.model_img_size = input_shape[2]    # mặt định W = H -> lấy vuông
        except Exception:
            pass

        # Cấu hình mean/std chuẩn hóa: tự động thiết lập CelebA mean/std cho MobileNetV4 nếu không truyền vào
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif "mnv4" in str(self.model_path).lower() or self.model_img_size == 224:
            self.mean = [0.5931, 0.4690, 0.4229]
            self.std = [0.2471, 0.2214, 0.2157]
        else:
            self.mean = None
            self.std = None

    def process_with_logits(self, raw_logits: np.ndarray) -> Dict:
        """
        Chuyển đổi kết quả Logits thô từ đầu ra mô hình thành từ điển kết quả phân loại chi tiết.

        Tham số:
            raw_logits (np.ndarray): Mảng 1D gồm 2 hoặc 3 phần tử (2-class hoặc 3-class model).

        Trả về:
            Dict chứa:
                - is_real (bool): True nếu là mặt thật, False nếu là giả mạo.
                - status (str): "real" hoặc "spoof".
                - detailed_status (str): Chi tiết lớp ("REAL", "SPOOF (Print)", "SPOOF (Screen)").
                - class_probs (Dict[str, float]): Xác xuất (softmax) của từng lớp.
                - logit_diff (float): Khoảng chênh lệch (real_logit - spoof_logit).
                - real_logit (float): Điểm logit của lớp mặt thật.
                - spoof_logit (float): Điểm logit của lớp mặt giả.
                - confidence (float): Độ tin cậy abs(logit_diff).
        """
        # Tính xác xuất Softmax cho tất cả các lớp
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        probs = exp_logits / np.sum(exp_logits)

        if len(raw_logits) == 2:
            real_logit = float(raw_logits[0])
            spoof_logit = float(raw_logits[1])
            class_probs = {
                "real": float(probs[0]),
                "spoof": float(probs[1]),
            }
            detailed_status = "REAL" if real_logit >= spoof_logit else "SPOOF"
        else:
            # Mô hình 3 lớp (MobileNetV4 CelebA-Spoof): Lớp 0: Real, Lớp 1: Physical Spoof (Print), Lớp 2: Digital Spoof (Screen)
            real_logit = float(raw_logits[0])
            spoof_logit = float(np.max(raw_logits[1:]))

            class_probs = {
                "real": float(probs[0]),
                "physical_spoof": float(probs[1]),
                "screen_spoof": float(probs[2]),
            }

            top_class = int(np.argmax(raw_logits))
            if top_class == 0:
                detailed_status = "REAL"
            elif top_class == 1:
                detailed_status = "SPOOF (Print)"
            else:
                detailed_status = "SPOOF (Screen)"

        logit_diff = real_logit - spoof_logit
        is_real = logit_diff >= self.logit_threshold
        confidence = abs(logit_diff)

        return {
            "is_real": bool(is_real),
            "status": "real" if is_real else "spoof",
            "detailed_status": detailed_status,
            "class_probs": class_probs,
            "logit_diff": float(logit_diff),
            "real_logit": float(real_logit),
            "spoof_logit": float(spoof_logit),
            "confidence": float(confidence),
        }

    def predict_crops(self, face_crops: List[np.ndarray]) -> List[Dict]:
        """
        Thực hiện dự đoán đồng thời trên một danh sách các ảnh crop khuôn mặt (Batch Inference).

        Tham số:
            face_crops (List[np.ndarray]): Danh sách ảnh crop khuôn mặt (RGB/BGR).

        Trả về:
            List[Dict]: Danh sách các từ điển kết quả phân loại tương ứng với từng khuôn mặt.
        """
        if not face_crops or self.session is None:
            return []

        try:
            # 1. Chạy tiền xử lý hình ảnh thành Tensor đầu vào (Batch, 3, H, W)
            batch_input = preprocess_batch(
                face_crops, self.model_img_size, mean=self.mean, std=self.std,
                apply_gamma=self.apply_gamma,
            )

            # 2. Suy luận bằng ONNX Runtime Session
            logits = self.session.run([], {self.input_name: batch_input})[0]

            if logits.ndim != 2 or logits.shape[0] != len(face_crops):
                raise ValueError(
                    f"Kích thước đầu ra của mô hình không khớp (kỳ vọng ({len(face_crops)}, C), nhận được {logits.shape})"
                )

            # 3. Chuyển đổi từng kết quả logit về dict
            return [self.process_with_logits(logits[i]) for i in range(len(face_crops))]
        except Exception as e:
            print(f"Lỗi suy luận AntiSpoof: {e}", file=sys.stderr)
            return []

    def predict_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Hàm tiện ích: Tự động crop khuôn mặt từ frame ảnh gốc dựa theo bbox và trả về kết quả dự đoán.

        Tham số:
            frame (np.ndarray): Khung hình ảnh gốc (H, W, C).
            bbox (Tuple[int, int, int, int]): Tọa độ (x, y, w, h).

        Trả về:
            Dict: Từ điển kết quả phân loại khuôn mặt.
        """
        face_crop = crop(frame, bbox, self.bbox_expansion_factor)
        results = self.predict_crops([face_crop])
        return results[0] if results else {}
