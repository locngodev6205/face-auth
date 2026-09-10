"""
Module phát hiện khuôn mặt YunNet (YunNet Face Detector Module).

Sử dụng API cv2.FaceDetectorYN tích hợp sẵn trong OpenCV kết hợp với mô hình ONNX siêu nhẹ
(detector_quantized.onnx ~122 KB). Được bọc thành lớp FaceDetector độc lập trong bộ module
face_auth.detection, có cùng interface với bản SCRFD (detection/detector.py):

    Output format detect():
        list[dict] với các trường:
            - "bbox"      : (x1, y1, x2, y2)  — xyxy, giống SCRFD
            - "score"     : float              — độ tin cậy confidence
            - "landmarks" : list[list[float]]  — 5 điểm [[x,y],...] dạng float, giống SCRFD

    Thứ tự 5 điểm landmark khớp quy ước ArcFace:
        0: mắt bên trái ảnh  (right eye)
        1: mắt bên phải ảnh  (left eye)
        2: đỉnh mũi
        3: khóe miệng trái ảnh
        4: khóe miệng phải ảnh

    → Tương thích trực tiếp với alignment/aligner.py (ARCFACE_DST_112) và
      tracking/tracker.py (compute_iou dùng xyxy).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Đường dẫn mặc định tới trọng số mô hình YunNet nén INT8
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "detector_quantized.onnx"


try:
    import config
    _DEFAULT_CONF_THRESH   = getattr(config, "DETECTOR_CONF_THRESH", 0.5)
    _DEFAULT_NMS_THRESH    = getattr(config, "DETECTOR_NMS_THRESH", 0.3)
    _DEFAULT_TOP_K         = getattr(config, "DETECTOR_TOP_K", 5000)
    _DEFAULT_MIN_FACE_SIZE = getattr(config, "DETECTOR_MIN_FACE_SIZE", 60)
    _DEFAULT_MARGIN        = getattr(config, "DETECTOR_MARGIN", 5)
    _DEFAULT_INPUT_SIZE    = getattr(config, "DETECTOR_INPUT_SIZE", (320, 320))
except Exception:
    _DEFAULT_CONF_THRESH   = 0.5
    _DEFAULT_NMS_THRESH    = 0.3
    _DEFAULT_TOP_K         = 5000
    _DEFAULT_MIN_FACE_SIZE = 60
    _DEFAULT_MARGIN        = 5
    _DEFAULT_INPUT_SIZE    = (320, 320)


class FaceDetector:
    """
    Lớp phát hiện khuôn mặt YunNet — thay thế trực tiếp cho FaceDetector (SCRFD).

    Sử dụng cv2.FaceDetectorYN với model ONNX siêu nhẹ (~122 KB), hoạt động
    hoàn toàn trên CPU mà không cần insightface, phù hợp cho thiết bị Edge
    (Raspberry Pi, Jetson Nano…).

    Interface detect() giữ nguyên quy ước giống SCRFD:
        bbox      → (x1, y1, x2, y2)  xyxy
        landmarks → list[list[float]]  5 điểm float
    để không cần thay đổi gì ở aligner.py, tracker.py hay app.py.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Optional[Tuple[int, int]] = None,
        conf_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        top_k: Optional[int] = None,
        min_face_size: Optional[int] = None,
        margin: Optional[int] = None,
    ):
        """
        Khởi tạo YunNet Detector. Mặc định tự động lấy các giá trị từ config.py (nếu có).

        Tham số:
            model_path    (Optional[str])        : Đường dẫn file .onnx. Mặc định dùng
                                                   detector_quantized.onnx trong models/.
            input_size    (Optional[Tuple[int]]): Kích thước đầu vào khởi tạo (mặc định (320, 320)).
                                                   Sẽ được ghi đè tự động theo kích thước ảnh thực tế
                                                   mỗi khi gọi detect().
            conf_thresh   (Optional[float])      : Ngưỡng confidence để chấp nhận khuôn mặt.
            nms_thresh    (Optional[float])      : Ngưỡng NMS loại bỏ bbox trùng lặp.
            top_k         (Optional[int])        : Số khuôn mặt tối đa giữ lại trước NMS.
            min_face_size (Optional[int])        : Kích thước khuôn mặt tối thiểu (px) để nhận diện.
            margin        (Optional[int])        : Khoảng cách tối thiểu (px) tới 4 cạnh ảnh.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số YunNet tại: '{self.model_path}'"
            )

        self.input_size = input_size if input_size is not None else _DEFAULT_INPUT_SIZE
        self.conf_thresh = conf_thresh if conf_thresh is not None else _DEFAULT_CONF_THRESH
        self.nms_thresh = nms_thresh if nms_thresh is not None else _DEFAULT_NMS_THRESH
        self.top_k = top_k if top_k is not None else _DEFAULT_TOP_K
        self.min_face_size = min_face_size if min_face_size is not None else _DEFAULT_MIN_FACE_SIZE
        self.margin = margin if margin is not None else _DEFAULT_MARGIN

        # Khởi tạo đối tượng FaceDetectorYN của OpenCV
        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",             # config path — không cần thiết với file ONNX
            self.input_size,
            self.conf_thresh,
            self.nms_thresh,
            self.top_k,
        )

    def detect(
        self,
        image: np.ndarray,
        min_face_size: Optional[int] = None,
        margin: Optional[int] = None,
    ) -> List[Dict]:
        """
        Phát hiện khuôn mặt trong ảnh và trả về list theo quy ước chuẩn.

        Tham số:
            image         (np.ndarray)   : Ảnh đầu vào BGR hoặc RGB.
            min_face_size (Optional[int]): Kích thước bbox tối thiểu (px). Nếu None, dùng self.min_face_size.
            margin        (Optional[int]): Khoảng cách tối thiểu tới 4 cạnh ảnh (px). Nếu None, dùng self.margin.

        Trả về:
            List[Dict]: Mỗi phần tử gồm:
                - "bbox"      : (x1, y1, x2, y2)        — xyxy, kiểu int
                - "score"     : float                    — confidence
                - "landmarks" : list[list[float]]        — 5 điểm [[x,y],...] kiểu float
        """
        if self._detector is None or image is None or image.size == 0:
            return []

        min_size = self.min_face_size if min_face_size is None else min_face_size
        edge_margin = self.margin if margin is None else margin

        # Tự động điều chỉnh input size theo kích thước ảnh thực tế
        img_h, img_w = image.shape[:2]
        self._detector.setInputSize((img_w, img_h))

        # Gọi OpenCV FaceDetectorYN — kết quả shape: [num_faces, 15]
        # Cột 0-3 : x, y, w, h (xywh)
        # Cột 4-13: 5 landmark (x0,y0, x1,y1, ..., x4,y4)
        # Cột 14  : confidence score
        _, faces = self._detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        detections = []
        for face in faces:
            # Lấy bbox dạng xywh rồi chuyển sang xyxy để đồng nhất với SCRFD
            x, y, w, h = face[:4].astype(float)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            conf = float(face[14])

            # Bỏ qua bbox vượt ra ngoài khung hình
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                continue

            # Lọc theo khoảng cách tới mép ảnh (margin)
            if edge_margin > 0:
                if min(x1, y1, img_w - x2, img_h - y2) < edge_margin:
                    continue

            # Lọc khuôn mặt quá nhỏ
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue

            # 5 landmark: index 4-13, shape (5, 2), giữ dạng float để khớp với SCRFD
            landmarks = face[4:14].reshape(5, 2).astype(float).tolist()

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "score": conf,
                "landmarks": landmarks,
            })

        return detections


# Alias tương thích ngược với code cũ
YunNetFaceDetector = FaceDetector

