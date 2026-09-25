"""
Module xử lý ảnh tiền xử lý (Preprocessing) cho mô hình Anti-Spoofing.

Chức năng chính:
    1. crop(): Cắt vùng khuôn mặt vuông từ ảnh gốc với tỉ lệ mở rộng (mặc định 1.55x).
       - Thêm viền theo thuật toán BORDER_REFLECT_101 để tránh viền đen giả tạo gây nhận diện nhầm Spoof.
    2. adaptive_gamma(): Điều chỉnh gamma động theo độ sáng thực tế của ảnh crop.
       - Target luma ~110/255; gamma được clamp trong [0.4, 2.5] để tránh diverge.
    3. preprocess(): Resize ảnh mặt vuông về kích thước đầu vào mô hình (128x128),
       - Chuẩn hóa giá trị điểm ảnh về dải [0, 1] và chuyển đổi thứ tự kênh từ HWC sang CHW.
       - Tùy chọn áp adaptive gamma trước khi normalize (``apply_gamma=True``).
    4. preprocess_batch(): Xử lý hàng loạt (Batch Processing) cho danh sách nhiều khuôn mặt.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    import config as _cfg
    _GAMMA_TARGET_LUMA: float = float(_cfg.PAD_GAMMA_TARGET)
except Exception:
    _GAMMA_TARGET_LUMA: float = 110.0
# Giới hạn gamma để tránh diverge khi ảnh quá tối (gamma → ∞) hoặc quá sáng (gamma → 0)
_GAMMA_MIN: float = 0.4
_GAMMA_MAX: float = 2.5

# LUT cache: tránh tính lại mảng 256 phần tử mỗi frame
_lut_cache: dict = {}


def adaptive_gamma(img: np.ndarray) -> np.ndarray:
    """
    Điều chỉnh gamma của ảnh BGR dựa trên độ sáng trung bình thực tế.

    Cách tính:
        luma = mean(kênh V trong HSV)          # đại diện cho perceived brightness
        gamma = log(TARGET / 255) / log(luma / 255)   # solved from: luma^gamma = TARGET
        gamma = clamp(gamma, GAMMA_MIN, GAMMA_MAX)

    Áp dụng qua LUT 256-entry (nhanh hơn pixelwise pow() ~10x).
    LUT được cache theo giá trị gamma đã làm tròn 2 chữ số thập phân.

    Tham số:
        img (np.ndarray): Ảnh BGR uint8.

    Trả về:
        np.ndarray: Ảnh BGR uint8 sau khi hiệu chỉnh gamma.
    """
    if img is None or img.size == 0:
        return img

    # Tính luma qua kênh V của HSV (nhanh, đúng với perceived brightness)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_luma = float(np.mean(hsv[:, :, 2]))

    # Tránh log(0) khi ảnh quá tối
    if mean_luma < 1.0:
        mean_luma = 1.0

    # Giải g từ: (luma/255)^g = (TARGET/255)  =>  g = log(TARGET/255) / log(luma/255)
    # LUT áp: pixel_out = (pixel_in/255)^g * 255
    # - luma < target → g < 1 → brightening (đường cong lõm lên)
    # - luma > target → g > 1 → darkening   (đường cong lõm xuống)
    gamma = np.log(_GAMMA_TARGET_LUMA / 255.0) / np.log(mean_luma / 255.0)
    gamma = float(np.clip(gamma, _GAMMA_MIN, _GAMMA_MAX))

    # Ảnh đã đủ sáng (gamma ≈ 1) → bỏ qua để tiết kiệm tài nguyên
    if abs(gamma - 1.0) < 0.05:
        return img

    # Tra LUT từ cache (key làm tròn 2 chữ số để gộp gamma gần nhau)
    key = round(gamma, 2)
    if key not in _lut_cache:
        lut = np.array(
            [(i / 255.0) ** gamma * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        _lut_cache[key] = lut
    return cv2.LUT(img, _lut_cache[key])


def preprocess(
    img: np.ndarray,
    model_img_size: int,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    apply_gamma: bool = True,
    convert_rgb: bool = False,
) -> np.ndarray:
    """
    Tiền xử lý 1 ảnh khuôn mặt:
        - (Tùy chọn) Adaptive gamma correction để robust với điều kiện ánh sáng khác nhau (chạy trên BGR/HSV).
        - (Tùy chọn) Chuyển kênh màu BGR sang RGB nếu model yêu cầu (convert_rgb=True).
        - Resize theo đúng tỉ lệ (letterboxing).
        - Đệm viền BORDER_REFLECT_101 để đảm bảo ảnh vuông kích thước model_img_size x model_img_size.
        - Chuyển dải điểm ảnh từ [0, 255] sang [0.0, 1.0].
        - Chuẩn hóa bằng mean và std nếu được cung cấp: (x - mean) / std.
        - Đổi định dạng từ OpenCV HWC (Height, Width, Channel) sang PyTorch/ONNX CHW (Channel, Height, Width).

    Tham số:
        img (np.ndarray): Ảnh crop khuôn mặt (BGR uint8).
        model_img_size (int): Kích thước cạnh ảnh vuông đầu vào của mô hình (ví dụ: 128).
        mean (Optional[List[float]]): Giá trị trung bình để chuẩn hóa kênh màu [R, G, B].
        std (Optional[List[float]]): Độ lệch chuẩn để chuẩn hóa kênh màu [R, G, B].
        apply_gamma (bool): Nếu True, áp adaptive gamma trước khi normalize (mặc định: True).
        convert_rgb (bool): Nếu True, chuyển BGR sang RGB sau gamma và trước khi normalize (mặc định: False).

    Trả về:
        np.ndarray: Mảng 3D float32 kích thước (3, model_img_size, model_img_size).
    """
    # Adaptive gamma trên ảnh uint8 trước khi normalize — tránh double-scale
    if apply_gamma:
        img = adaptive_gamma(img)

    # Chuyển đổi BGR -> RGB nếu model profile yêu cầu (ví dụ MobileNetV3/V4 CelebA-Spoof)
    if convert_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    new_size = model_img_size
    old_size = img.shape[:2]

    # Tính tỉ lệ co giãn ảnh sao cho không làm méo tỷ lệ khuôn mặt
    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])

    # Chọn thuật toán nội suy phù hợp: INTER_LANCZOS4 khi phóng to, INTER_AREA khi thu nhỏ
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(
        img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation
    )

    # Tính khoảng cách viền cần đệm xung quanh để đạt kích thước vuông new_size x new_size
    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Sử dụng BORDER_REFLECT_101 để đệm viền mượt mà (tránh viền đen làm sai lệch mô hình AI)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    # Chuyển kênh màu từ (H, W, C) -> (C, H, W) và chuẩn hóa điểm ảnh về [0.0, 1.0]
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean_arr) / std_arr

    return img


def preprocess_batch(
    face_crops: List[np.ndarray],
    model_img_size: int,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    apply_gamma: bool = True,
    convert_rgb: bool = False,
) -> np.ndarray:
    """
    Tiền xử lý đồng thời một danh sách nhiều ảnh crop khuôn mặt (Batching).

    Tham số:
        face_crops (List[np.ndarray]): Danh sách các ảnh khuôn mặt đã crop.
        model_img_size (int): Kích thước đầu vào mô hình (ví dụ: 128).
        mean (Optional[List[float]]): Giá trị mean chuẩn hóa.
        std (Optional[List[float]]): Giá trị std chuẩn hóa.
        apply_gamma (bool): Nếu True, áp adaptive gamma cho từng crop (mặc định: True).
        convert_rgb (bool): Nếu True, chuyển BGR sang RGB cho từng crop (mặc định: False).

    Trả về:
        np.ndarray: Mảng 4D float32 kích thước (batch_size, 3, model_img_size, model_img_size).
    """
    if not face_crops:
        raise ValueError("Danh sách face_crops không được rỗng!")

    # Khởi tạo mảng batch chứa n khuôn mặt
    batch = np.zeros(
        (len(face_crops), 3, model_img_size, model_img_size), dtype=np.float32
    )
    for i, face_crop in enumerate(face_crops):
        batch[i] = preprocess(
            face_crop,
            model_img_size,
            mean=mean,
            std=std,
            apply_gamma=apply_gamma,
            convert_rgb=convert_rgb,
        )

    return batch


def preprocess_dct(
    img: np.ndarray,
    model_img_size: int = 224,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Tiền xử lý trích xuất miền tần số 2D-DCT cho mô hình E2 Frequency-Only PAD.
    Pipeline đồng bộ 100% với e2_frequency_only_dct_train.ipynb:
        1. BGR sang RGB
        2. Letterbox resize về model_img_size x model_img_size với đệm BORDER_REFLECT_101
        3. Tính luminance: 0.299 * R + 0.587 * G + 0.114 * B trong dải [0, 1]
        4. Biến đổi 2D DCT (cv2.dct)
        5. Nén động học có dấu: sign(C) * log1p(|C|)
        6. Chuẩn hóa z-score theo từng mẫu (per-sample standardisation)

    Trả về:
        np.ndarray: Mảng 3D float32 kích thước (1, model_img_size, model_img_size).
    """
    if img is None or img.size == 0:
        return np.zeros((1, model_img_size, model_img_size), dtype=np.float32)

    # 1. Chuyển sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize với giữ tỷ lệ và đệm BORDER_REFLECT_101
    old_h, old_w = img_rgb.shape[:2]
    ratio = float(model_img_size) / max(old_h, old_w)
    scaled_h = max(1, int(old_h * ratio))
    scaled_w = max(1, int(old_w * ratio))
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA

    resized = cv2.resize(
        img_rgb,
        (scaled_w, scaled_h),
        interpolation=interpolation,
    )

    delta_w = model_img_size - scaled_w
    delta_h = model_img_size - scaled_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_REFLECT_101,
    )

    # 3. Tính luminance float32 [0, 1]
    rgb_f = padded.astype(np.float32) / 255.0
    luma = (
        0.299 * rgb_f[..., 0]
        + 0.587 * rgb_f[..., 1]
        + 0.114 * rgb_f[..., 2]
    ).astype(np.float32)

    # 4. Biến đổi 2D DCT
    coeff = cv2.dct(np.ascontiguousarray(luma, dtype=np.float32))

    # 5. Signed-log1p compression
    coeff = np.sign(coeff) * np.log1p(np.abs(coeff))

    # 6. Per-sample standardisation (z-score)
    mean = float(coeff.mean())
    std = float(coeff.std())
    coeff = (coeff - mean) / max(std, eps)

    return coeff[None, ...].astype(np.float32)


def preprocess_dct_batch(
    face_crops: List[np.ndarray],
    model_img_size: int = 224,
) -> np.ndarray:
    """
    Tiền xử lý DCT hàng loạt cho danh sách crop khuôn mặt.
    Trả về mảng 4D float32 (batch_size, 1, model_img_size, model_img_size).
    """
    if not face_crops:
        raise ValueError("Danh sách face_crops không được rỗng!")

    batch = np.zeros(
        (len(face_crops), 1, model_img_size, model_img_size), dtype=np.float32
    )
    for i, face_crop in enumerate(face_crops):
        batch[i] = preprocess_dct(face_crop, model_img_size)
    return batch


def crop(img: np.ndarray, bbox: Tuple[int, int, int, int], bbox_expansion_factor: float = 1.55) -> np.ndarray:
    """
    Cắt vùng khuôn mặt từ khung hình gốc dựa theo Bounding Box (bbox),
    tự động mở rộng vùng cắt theo bbox_expansion_factor (mặc định 1.55x)
    để lấy thêm ngữ cảnh (tóc, tai, viền màn hình) phục vụ nhận diện giả mạo.

    Tham số:
        img (np.ndarray): Khung hình ảnh gốc (H, W, C).
        bbox (Tuple[int, int, int, int]): Tọa độ (x1, y1, x2, y2) — định dạng xyxy (SCRFD và YunNet đều xuất xyxy).
        bbox_expansion_factor (float): Tỷ lệ mở rộng khung bao (mặc định 1.55).

    Trả về:
        np.ndarray: Ảnh khuôn mặt đã cắt vuông.
    """
    original_height, original_width = img.shape[:2]

    # Luôn interpet bbox là xyxy — cả hai detector đều xuất định dạng này
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        raise ValueError("Kích thước bounding box không hợp lệ!")

    # Chọn cạnh lớn nhất để tạo khung vuông
    max_dim = max(w, h)
    center_x = x1 + w / 2
    center_y = y1 + h / 2

    # Tính tâm và tọa độ đỉnh mới sau khi nhân tỷ lệ mở rộng bbox_expansion_factor
    x_start = int(center_x - max_dim * bbox_expansion_factor / 2)
    y_start = int(center_y - max_dim * bbox_expansion_factor / 2)
    crop_size = int(max_dim * bbox_expansion_factor)

    # Giới hạn tọa độ trong phạm vi ảnh gốc
    crop_x1 = max(0, x_start)
    crop_y1 = max(0, y_start)
    crop_x2 = min(original_width, x_start + crop_size)
    crop_y2 = min(original_height, y_start + crop_size)

    # Tính khoảng lệch nếu vùng crop bị vượt ra ngoài mép ảnh
    top_pad = int(max(0, -y_start))
    left_pad = int(max(0, -x_start))
    bottom_pad = int(max(0, (y_start + crop_size) - original_height))
    right_pad = int(max(0, (x_start + crop_size) - original_width))

    # Cắt vùng ảnh hợp lệ
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        cropped_img = np.zeros((0, 0, 3), dtype=img.dtype)

    # Đệm viền BORDER_REFLECT_101 cho phần bị tràn ngoài mép ảnh
    result = cv2.copyMakeBorder(
        cropped_img,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_REFLECT_101,
    )

    # Kiểm tra và điều chỉnh chính xác kích thước vuông crop_size x crop_size
    if result.shape[0] != crop_size or result.shape[1] != crop_size:
        result = cv2.resize(result, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    return result
