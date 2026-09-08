import cv2
import numpy as np

# state 2

# Template 5 điểm chuẩn của ArcFace cho input 112x112, thứ tự:
# mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải.
# Đây là template gốc insightface (arcface_dst) — khớp với thứ tự landmark
# mà hầu hết detector (kể cả YOLOv8-face có keypoints) xuất ra.
ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image, landmarks, target_points=ARCFACE_DST_112, output_size=(112, 112)):
    src = np.asarray(landmarks, dtype=np.float32)
    dst = np.asarray(target_points, dtype=np.float32)

    # tính toán một ma trận biến đổi hình học
    matrix, _ = cv2.estimateAffinePartial2D(src, dst)

    if matrix is None:
        raise ValueError("Could not estimate alignment transform")

    aligned = cv2.warpAffine(
        image,
        matrix,
        output_size,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return aligned


def get_input_face(frame, bbox, landmarks, output_size=(112, 112), crop_margin=0.2):
    """
    Trả về ảnh khuôn mặt sẵn sàng đưa vào recognition hoặc anti-spoofing model.

    - Có landmarks (5 điểm): align chuẩn theo template ArcFace (tự động scale target_points theo output_size).
    - Không có landmarks: fallback crop theo bbox với margin.
    """
    if landmarks is not None and len(landmarks) == 5:
        try:
            scale = output_size[0] / 112.0
            target_pts = ARCFACE_DST_112 * scale
            return align_face(frame, landmarks, target_pts, output_size)
        except Exception:
            pass

    # dựa vào bbox
    x1, y1, x2, y2 = bbox
    # Hỗ trợ tự động chuyển đổi nếu bbox có dạng (x, y, w, h)
    if x2 < x1 or y2 < y1:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    mx, my = int(bw * crop_margin), int(bh * crop_margin)
    x1m = max(0, x1 - mx)
    y1m = max(0, y1 - my)
    x2m = min(w, x2 + mx)
    y2m = min(h, y2 + my)

    if x2m <= x1m or y2m <= y1m:
        return None

    crop = frame[y1m:y2m, x1m:x2m]
    return cv2.resize(crop, output_size)