#!/usr/bin/env python3
"""
Script kiểm tra nhanh Face Detection trên các ảnh trong thư mục gallery.

Tính năng:
    - Quét tự động toàn bộ ảnh trong thư mục gallery (hỗ trợ thư mục con theo tên người).
    - Hỗ trợ cả 2 bộ detector: YunNet (mặc định siêu nhẹ ~122KB) và SCRFD (insightface).
    - Đo lường và thống kê thời gian inference (ms) từng ảnh và trung bình.
    - Báo cáo chi tiết vị trí bbox, kích thước (WxH), độ tin cậy (score), landmark 5 điểm.
    - Cảnh báo các ảnh không tìm thấy khuôn mặt (0 face) hoặc có nhiều hơn 1 khuôn mặt (>1 faces).
    - Tự động vẽ bounding box, điểm landmark và lưu ảnh kết quả vào thư mục output.
    - Hỗ trợ tùy chọn cắt khuôn mặt (crop) và xem trực tiếp qua cửa sổ OpenCV (--show).

Cách dùng:
    python scripts/test_detection.py
    python scripts/test_detection.py --conf-thresh 0.6
    python scripts/test_detection.py --detector scrfd
    python scripts/test_detection.py --image "gallery/Nguyen Quang Anh/48d43963b624367a6f352.jpg"
    python scripts/test_detection.py --save-dir output/test_detection
    python scripts/test_detection.py --show
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Màu sắc BGR cho visualization
COLOR_PRIMARY_BOX = (0, 220, 80)      # Xanh lá - khuôn mặt lớn nhất / chính
COLOR_SECONDARY_BOX = (0, 165, 255)   # Cam - khuôn mặt phụ / thêm
COLOR_LANDMARK = (255, 255, 0)        # Vàng / Cyan
COLOR_TEXT_BG = (30, 30, 30)          # Xám đậm
COLOR_TEXT = (255, 255, 255)         # Trắng


def init_detector(detector_type: str, conf_thresh: float, nms_thresh: float, min_face_size: int, margin: int):
    """Khởi tạo detector theo lựa chọn (yunnet hoặc scrfd)."""
    if detector_type == "yunnet":
        from detection.yunnet_detector import FaceDetector
        return FaceDetector(
            model_path=config.DETECTOR_MODEL_PATH,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            top_k=config.DETECTOR_TOP_K,
            min_face_size=min_face_size,
            margin=margin,
        )
    elif detector_type == "scrfd":
        from detection.detector import FaceDetector
        return FaceDetector(
            model_name=config.MODEL_PACK_NAME,
            ctx_id=config.MODEL_CTX_ID,
            det_size=config.DETECTOR_DET_SIZE,
            conf_thresh=conf_thresh,
        )
    else:
        raise ValueError(f"Không hỗ trợ detector: '{detector_type}'. Chỉ chấp nhận 'yunnet' hoặc 'scrfd'.")


def collect_images(gallery_dir: Path, single_image: str = None):
    """Thu thập danh sách ảnh cần kiểm tra."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    if single_image:
        p = Path(single_image)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {p}")
        return [p]

    if not gallery_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục gallery tại: {gallery_dir}")

    images = []
    for root, _, files in os.walk(gallery_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                images.append(Path(root) / f)

    return images


def draw_detections(image: np.ndarray, detections: list, filename: str, latency_ms: float, detector_name: str):
    """Vẽ bounding box, score, landmarks và header thông tin lên ảnh."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Header bar phía trên
    header_h = 36
    cv2.rectangle(vis, (0, 0), (w, header_h), (25, 25, 25), -1)
    header_text = f"[{detector_name.upper()}] {filename} | {w}x{h} | {len(detections)} face(s) | {latency_ms:.1f}ms"
    cv2.putText(vis, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        box_w = x2 - x1
        box_h = y2 - y1

        # Màu sắc: khuôn mặt đầu tiên là xanh lá, các khuôn mặt sau là cam
        box_color = COLOR_PRIMARY_BOX if i == 0 else COLOR_SECONDARY_BOX

        # Vẽ bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

        # Label thông tin khuôn mặt
        label = f"#{i+1}: {score:.2f} ({box_w}x{box_h})"
        (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        label_y1 = max(header_h + 5, y1 - txt_h - 8)
        label_y2 = label_y1 + txt_h + 6
        label_x2 = min(w, x1 + txt_w + 10)

        cv2.rectangle(vis, (x1, label_y1), (label_x2, label_y2), box_color, -1)
        cv2.putText(vis, label, (x1 + 4, label_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Vẽ 5 điểm landmark
        landmarks = det.get("landmarks")
        if landmarks:
            landmark_colors = [
                (255, 0, 0),    # Điểm 1 (mắt phải người xem) - Xanh dương
                (0, 0, 255),    # Điểm 2 (mắt trái người xem) - Đỏ
                (0, 255, 255),  # Điểm 3 (mũi) - Vàng
                (255, 0, 255),  # Điểm 4 (khóe miệng trái) - Hồng
                (0, 255, 0),    # Điểm 5 (khóe miệng phải) - Xanh lá
            ]
            for pt_idx, pt in enumerate(landmarks):
                px, py = int(pt[0]), int(pt[1])
                pt_color = landmark_colors[pt_idx % len(landmark_colors)]
                cv2.circle(vis, (px, py), 4, pt_color, -1, cv2.LINE_AA)
                cv2.circle(vis, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh Face Detection trên ảnh trong thư mục gallery")
    parser.add_argument("--gallery", type=str, default="gallery", help="Đường dẫn tới thư mục gallery (mặc định: gallery)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn tới 1 ảnh duy nhất để test")
    parser.add_argument("--detector", type=str, choices=["yunnet", "scrfd"], default="yunnet", help="Chọn model detector (yunnet hoặc scrfd, mặc định: yunnet)")
    parser.add_argument("--conf-thresh", type=float, default=config.DETECTOR_CONF_THRESH, help=f"Ngưỡng confidence (mặc định: {config.DETECTOR_CONF_THRESH})")
    parser.add_argument("--nms-thresh", type=float, default=config.DETECTOR_NMS_THRESH, help=f"Ngưỡng NMS IoU (mặc định: {config.DETECTOR_NMS_THRESH})")
    parser.add_argument("--min-face-size", type=int, default=config.DETECTOR_MIN_FACE_SIZE, help=f"Kích thước khuôn mặt tối thiểu tính bằng px (mặc định: {config.DETECTOR_MIN_FACE_SIZE})")
    parser.add_argument("--margin", type=int, default=config.DETECTOR_MARGIN, help=f"Khoảng cách tối thiểu từ bbox tới mép ảnh (mặc định: {config.DETECTOR_MARGIN})")
    parser.add_argument("--save-dir", type=str, default="output/test_detection", help="Thư mục lưu ảnh đã vẽ bbox (mặc định: output/test_detection)")
    parser.add_argument("--no-save", action="store_true", help="Không lưu ảnh kết quả vào ổ đĩa")
    parser.add_argument("--crop", action="store_true", help="Cắt và lưu riêng các khuôn mặt phát hiện được")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ OpenCV tương tác (bấm phím bất kỳ để qua ảnh tiếp, Esc/q để thoát)")
    args = parser.parse_args()

    gallery_path = Path(args.gallery)
    if not gallery_path.is_absolute():
        gallery_path = PROJECT_ROOT / gallery_path

    try:
        image_paths = collect_images(gallery_path, args.image)
    except Exception as e:
        print(f"[!] Lỗi thu thập ảnh: {e}")
        return

    if not image_paths:
        print(f"[!] Không tìm thấy ảnh nào trong: {gallery_path}")
        return

    print("=" * 70)
    print("           FACE DETECTION TEST RUNNER")
    print("=" * 70)
    print(f" * Detector        : {args.detector.upper()}")
    print(f" * Confidence      : {args.conf_thresh}")
    print(f" * Min Face Size   : {args.min_face_size} px")
    print(f" * Margin to Edge  : {args.margin} px")
    print(f" * Số lượng ảnh    : {len(image_paths)}")
    print(f" * Lưu kết quả     : {'Không' if args.no_save else args.save_dir}")
    print("-" * 70)

    print(f"\n[+] Đang khởi tạo detector '{args.detector}'...")
    try:
        detector = init_detector(
            detector_type=args.detector,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            min_face_size=args.min_face_size,
            margin=args.margin,
        )
    except Exception as e:
        print(f"[!] Lỗi khởi tạo detector: {e}")
        return

    save_dir = PROJECT_ROOT / args.save_dir
    crops_dir = save_dir / "crops"
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.crop:
            crops_dir.mkdir(parents=True, exist_ok=True)

    latencies = []
    images_with_no_faces = []
    images_with_multi_faces = []
    total_faces = 0

    print("\n[+] Bắt đầu xử lý từng ảnh:")

    for idx, img_path in enumerate(image_paths, start=1):
        rel_path = img_path.relative_to(PROJECT_ROOT) if img_path.is_relative_to(PROJECT_ROOT) else img_path
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"\n[{idx}/{len(image_paths)}] [!] Lỗi: Không thể đọc file ảnh: {rel_path}")
            continue

        h, w = frame.shape[:2]

        # Đo thời gian inference
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        num_faces = len(detections)
        total_faces += num_faces

        status_tag = "[OK]"
        if num_faces == 0:
            status_tag = "[!] 0 FACE"
            images_with_no_faces.append(str(rel_path))
        elif num_faces > 1:
            status_tag = f"[!] {num_faces} FACES"
            images_with_multi_faces.append((str(rel_path), num_faces))

        print(f"\n[{idx}/{len(image_paths)}] {status_tag} {rel_path} ({w}x{h}, {latency:.1f}ms) -> {num_faces} mặt")

        for f_idx, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det["bbox"]
            fw, fh = x2 - x1, y2 - y1
            conf = det["score"]
            lm_info = f"{len(det['landmarks'])} điểm" if det.get("landmarks") else "Không có"
            print(f"      Face #{f_idx}: bbox=({x1}, {y1}, {x2}, {y2}), size={fw}x{fh}px, conf={conf:.3f}, landmarks={lm_info}")

        # Vẽ và lưu ảnh
        if not args.no_save:
            annotated = draw_detections(frame, detections, img_path.name, latency, args.detector)
            # Giữ cấu trúc thư mục con hoặc đặt tên tương ứng
            safe_name = f"{img_path.parent.name}_{img_path.name}" if img_path.parent != gallery_path else img_path.name
            out_file = save_dir / f"det_{safe_name}"
            cv2.imwrite(str(out_file), annotated)

            # Nếu bật tùy chọn crop
            if args.crop:
                for f_idx, det in enumerate(detections, start=1):
                    x1, y1, x2, y2 = det["bbox"]
                    crop_img = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop_img.size > 0:
                        crop_file = crops_dir / f"crop_{safe_name}_f{f_idx}.jpg"
                        cv2.imwrite(str(crop_file), crop_img)

        # Hiển thị nếu bật cờ --show
        if args.show:
            annotated = draw_detections(frame, detections, img_path.name, latency, args.detector)
            # Resize hiển thị nếu ảnh quá to để xem vừa màn hình
            display_img = annotated
            max_disp_h, max_disp_w = 900, 1200
            if h > max_disp_h or w > max_disp_w:
                scale = min(max_disp_h / h, max_disp_w / w)
                display_img = cv2.resize(annotated, (int(w * scale), int(h * scale)))

            cv2.imshow("Face Detection Test (Press any key for next, Esc/q to exit)", display_img)
            key = cv2.waitKey(0) & 0xFF
            if key in [27, ord('q')]:
                print("\n[!] Người dùng hủy xem trước (Quit).")
                break

    if args.show:
        cv2.destroyAllWindows()

    # Thống kê tổng hợp
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print("\n" + "=" * 70)
    print("                    BÁO CÁO TỔNG HỢP")
    print("=" * 70)
    print(f" * Tổng số ảnh đã test    : {len(image_paths)}")
    print(f" * Tổng khuôn mặt tìm thấy: {total_faces}")
    print(f" * Thời gian xử lý TB     : {avg_latency:.2f} ms/ảnh")

    if images_with_no_faces:
        print(f"\n [!] Cảnh báo: {len(images_with_no_faces)} ảnh KHÔNG tìm thấy khuôn mặt:")
        for path_str in images_with_no_faces:
            print(f"     - {path_str}")

    if images_with_multi_faces:
        print(f"\n [!] Cảnh báo: {len(images_with_multi_faces)} ảnh tìm thấy NHIỀU HƠN 1 khuôn mặt:")
        for path_str, n_faces in images_with_multi_faces:
            print(f"     - {path_str} ({n_faces} mặt)")

    if not args.no_save:
        print(f"\n [v] Ảnh đã được vẽ bbox và lưu tại: {save_dir}")
        if args.crop:
            print(f" [v] Các khuôn mặt crop đã được lưu tại: {crops_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
