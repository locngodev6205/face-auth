"""
Module phát hiện khuôn mặt YunNet (YunNet Face Detector Module).

Sử dụng API cv2.FaceDetectorYN tích hợp sẵn trong OpenCV kết hợp với mô hình ONNX siêu nhẹ (detector_quantized.onnx ~122 KB).
Được bọc thành lớp YunNetFaceDetector độc lập trong bộ module face_auth.detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Đường dẫn mặc định tới trọng số mô hình YunNet nén INT8
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "detector_quantized.onnx"


class YunNetFaceDetector:
    """
    Lớp phát hiện khuôn mặt YunNet (YunNet Face Detector Wrapper).

    Cung cấp khả năng phát hiện vị trí khuôn mặt và trích xuất 5 điểm landmark
    (2 mắt, 1 mũi, 2 khóe miệng) với tốc độ siêu nhanh trên CPU.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (320, 320),
        conf_thresh: float = 0.8,
        nms_thresh: float = 0.3,
        top_k: int = 5000,
    ):
        """
        Khởi tạo YunNet Detector:

        Tham số:
            model_path (Optional[str]): Đường dẫn file .onnx (mặc định dùng detector_quantized.onnx trong models/).
            input_size (Tuple[int, int]): Kích thước ảnh khung hình chuẩn ban đầu (mặc định (320, 320)).
            conf_thresh (float): Ngưỡng độ tin cậy confidence để chấp nhận một khuôn mặt (mặc định 0.8).
            nms_thresh (float): Ngưỡng NMS (Non-Maximum Suppression) lọc khung bao trùng lặp (mặc định 0.3).
            top_k (int): Số lượng khuôn mặt tối đa giữ lại trước NMS (mặc định 5000).
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số YunNet tại: '{self.model_path}'"
            )

        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

        # Khởi tạo đối tượng FaceDetectorYN của OpenCV
        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            input_size,
            conf_thresh,
            nms_thresh,
            top_k,
        )

    def detect(
        self, image: np.ndarray, min_face_size: int = 60, margin: int = 5
    ) -> List[Dict]:
        """
        Phát hiện danh sách các khuôn mặt trong khung hình ảnh truyền vào.

        Tham số:
            image (np.ndarray): Mảng ảnh đầu vào (BGR hoặc RGB).
            min_face_size (int): Kích thước nhỏ nhất (rộng & cao) của khuôn mặt được coi là hợp lệ (mặc định 60px).
            margin (int): Khoảng cách tối thiểu từ khuôn mặt tới mép viền ảnh để tránh cắt lẹm mặt (mặc định 5px).

        Trả về:
            List[Dict]: Danh sách các từ điển kết quả chứa:
                - 'bbox': Tọa độ khung bao dạng (x, y, width, height).
                - 'score': Độ tin cậy confidence (float).
                - 'landmarks': Danh sách 5 tọa độ điểm đặc trưng [(x1, y1), ..., (x5, y5)].
        """
        if self._detector is None or image is None or image.size == 0:
            return []

        # Tự động điều chỉnh kích thước đầu vào của detector theo kích thước thật của ảnh
        img_h, img_w = image.shape[:2]
        self._detector.setInputSize((img_w, img_h))

        # Gọi hàm detect của OpenCV FaceDetectorYN
        _, faces = self._detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        detections = []
        for face in faces:
            # 4 tham số đầu: Tọa độ x, y, width, height
            x, y, w, h = face[:4].astype(int)
            # Tham số thứ 15 (index 14): Độ tin cậy confidence
            conf = float(face[14])

            # Bỏ qua các bbox vượt ra ngoài khung hình
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                continue

            # Kiểm tra khoảng cách tới 4 mép viền ảnh (margin)
            dist_left = x
            dist_right = img_w - (x + w)
            dist_top = y
            dist_bottom = img_h - (y + h)
            if min(dist_left, dist_right, dist_top, dist_bottom) < margin:
                continue

            # Lọc các khuôn mặt đạt kích thước tối thiểu min_face_size
            if w >= min_face_size and h >= min_face_size:
                # 10 tham số tiếp theo (index 4-13): 5 tọa độ điểm landmark (x, y)
                landmarks = face[4:14].reshape(5, 2).astype(int)
                detections.append(
                    {
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "score": conf,
                        "landmarks": [(int(lx), int(ly)) for lx, ly in landmarks],
                    }
                )

        return detections
