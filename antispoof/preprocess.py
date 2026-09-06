"""
Module xử lý ảnh tiền xử lý (Preprocessing) cho mô hình Anti-Spoofing.

Chức năng chính:
    1. crop(): Cắt vùng khuôn mặt vuông từ ảnh gốc với tỉ lệ mở rộng (mặc định 1.5x).
       - Thêm viền theo thuật toán BORDER_REFLECT_101 để tránh viền đen giả tạo gây nhận diện nhầm Spoof.
    2. preprocess(): Resize ảnh mặt vuông về kích thước đầu vào mô hình (128x128),
       - Chuẩn hóa giá trị điểm ảnh về dải [0, 1] và chuyển đổi thứ tự kênh từ HWC sang CHW.
    3. preprocess_batch(): Xử lý hàng loạt (Batch Processing) cho danh sách nhiều khuôn mặt.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def preprocess(
    img: np.ndarray,
    model_img_size: int,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Tiền xử lý 1 ảnh khuôn mặt:
        - Resize theo đúng tỉ lệ (letterboxing).
        - Đệm viền BORDER_REFLECT_101 để đảm bảo ảnh vuông kích thước model_img_size x model_img_size.
        - Chuyển dải điểm ảnh từ [0, 255] sang [0.0, 1.0].
        - Chuẩn hóa bằng mean và std nếu được cung cấp: (x - mean) / std.
        - Đổi định dạng từ OpenCV HWC (Height, Width, Channel) sang PyTorch/ONNX CHW (Channel, Height, Width).

    Tham số:
        img (np.ndarray): Ảnh crop khuôn mặt (RGB/BGR).
        model_img_size (int): Kích thước cạnh ảnh vuông đầu vào của mô hình (ví dụ: 128).
        mean (Optional[List[float]]): Giá trị trung bình để chuẩn hóa kênh màu [R, G, B].
        std (Optional[List[float]]): Độ lệch chuẩn để chuẩn hóa kênh màu [R, G, B].

    Trả về:
        np.ndarray: Mảng 3D float32 kích thước (3, model_img_size, model_img_size).
    """
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
) -> np.ndarray:
    """
    Tiền xử lý đồng thời một danh sách nhiều ảnh crop khuôn mặt (Batching).

    Tham số:
        face_crops (List[np.ndarray]): Danh sách các ảnh khuôn mặt đã crop.
        model_img_size (int): Kích thước đầu vào mô hình (ví dụ: 128).
        mean (Optional[List[float]]): Giá trị mean chuẩn hóa.
        std (Optional[List[float]]): Giá trị std chuẩn hóa.

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
        batch[i] = preprocess(face_crop, model_img_size, mean=mean, std=std)

    return batch


def crop(img: np.ndarray, bbox: Tuple[int, int, int, int], bbox_expansion_factor: float = 1.5) -> np.ndarray:
    """
    Cắt vùng khuôn mặt từ khung hình gốc dựa theo Bounding Box (bbox),
    tự động mở rộng vùng cắt theo bbox_expansion_factor (mặc định 1.5x)
    để lấy thêm ngữ cảnh (tóc, tai, viền màn hình) phục vụ nhận diện giả mạo.

    Tham số:
        img (np.ndarray): Khung hình ảnh gốc (H, W, C).
        bbox (Tuple[int, int, int, int]): Tọa độ (x, y, w, h) hoặc (x1, y1, x2, y2).
        bbox_expansion_factor (float): Tỷ lệ mở rộng khung bao (mặc định 1.5 = mở rộng thêm 50%).

    Trả về:
        np.ndarray: Ảnh khuôn mặt đã cắt vuông.
    """
    original_height, original_width = img.shape[:2]
    x, y, w, h = bbox

    # Hỗ trợ tự động chuyển đổi nếu đầu vào là format (x1, y1, x2, y2) thay vì (x, y, w, h)
    if w > x and h > y and w > 0 and h > 0 and x < original_width and y < original_height:
        if w <= original_width and h <= original_height:
            w_dim = w - x
            h_dim = h - y
            if w_dim > 0 and h_dim > 0:
                w, h = w_dim, h_dim

    if w <= 0 or h <= 0:
        raise ValueError("Kích thước bounding box không hợp lệ!")

    # Chọn cạnh lớn nhất để tạo khung vuông
    max_dim = max(w, h)
    center_x = x + w / 2
    center_y = y + h / 2

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
