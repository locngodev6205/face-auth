import os
import cv2
import sys
import argparse

# Thêm thư mục gốc vào sys.path để có thể import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from detection.yunnet_detector import FaceDetector
from recognition.embedder import FaceEmbedder
from alignment.aligner import get_input_face
from database.vector_db import VectorDB
from enrollment.enroll import enroll_person


def main():
    parser = argparse.ArgumentParser(description="Đăng ký dữ liệu khuôn mặt vào Database từ thư mục gallery")
    parser.add_argument("--gallery", type=str, default="gallery", help="Đường dẫn đến thư mục chứa ảnh (mặc định: gallery)")
    parser.add_argument("--ignore-duplicate", action="store_true", help="Bỏ qua không ghi đè nếu người dùng đã tồn tại (mặc định: ghi đè)")
    parser.add_argument("--no-individuals", action="store_true", help="Không lưu các vector thành phần (mặc định: có lưu)")
    args = parser.parse_args()

    gallery_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.gallery)

    if not os.path.exists(gallery_dir):
        print(f"Thư mục '{gallery_dir}' không tồn tại. Đang tiến hành tạo thư mục...")
        os.makedirs(gallery_dir)
        print("Vui lòng thêm dữ liệu vào thư mục này rồi chạy lại script.")
        return

    print("Đang khởi tạo model và kết nối database...")
    detector = FaceDetector(
        model_path=config.DETECTOR_MODEL_PATH,
        conf_thresh=config.DETECTOR_CONF_THRESH,
        nms_thresh=config.DETECTOR_NMS_THRESH,
        top_k=config.DETECTOR_TOP_K,
        min_face_size=config.DETECTOR_MIN_FACE_SIZE,
        margin=config.DETECTOR_MARGIN,
    )
    embedder = FaceEmbedder(config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID)
    db = VectorDB(
        conninfo=config.DB_CONN_INFO,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
    )

    print(f"Bắt đầu quét thư mục gallery: {gallery_dir}")

    for person_name in os.listdir(gallery_dir):
        person_dir = os.path.join(gallery_dir, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"\nĐang xử lý dữ liệu cho: {person_name}")
        embeddings = []

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(person_dir, filename)
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"  [!] Lỗi: Không thể đọc ảnh {filename}")
                continue

            detections = detector.detect(frame)

            if not detections:
                print(f"  [!] Lỗi: Không tìm thấy khuôn mặt nào trong {filename}")
                continue

            if len(detections) > 1:
                det = max(detections, key=lambda d: d.get("score", 0.0))
                print(f"  [!] Cảnh báo: Tìm thấy {len(detections)} khuôn mặt trong {filename}. Đã chọn khuôn mặt có confidence cao nhất ({det['score']:.2f}).")
            else:
                det = detections[0]
            aligned_face = get_input_face(frame, det["bbox"], det.get("landmarks"), embedder.input_size)

            if aligned_face is None:
                print(f"  [!] Lỗi: Không tạo được ảnh align từ {filename}")
                continue

            embedding = embedder.embed_aligned(aligned_face)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"  [v] Trích xuất thành công đặc trưng từ {filename}")
            else:
                print(f"  [!] Lỗi: Không thể tạo embedding từ {filename}")

        if not embeddings:
            print(f"==> Bỏ qua '{person_name}' - Không có ảnh hợp lệ nào.")
            continue

        try:
            overwrite = not args.ignore_duplicate
            save_individuals = not args.no_individuals
            new_id, is_new, is_ignored, warnings = enroll_person(
                db, user_id=person_name, name=person_name, embeddings=embeddings, overwrite=overwrite, save_individuals=save_individuals
            )

            for w in warnings:
                print(f"  [!] {w}")

            if is_ignored:
                action = "Bỏ qua (đã tồn tại)"
            else:
                action = "Đăng ký mới" if is_new else "Cập nhật (ghi đè)"

            print(f"==> {action} '{person_name}' với ID: {new_id} (Dựa trên {len(embeddings)} ảnh)")
        except Exception as e:
            print(f"==> Lỗi khi đăng ký '{person_name}': {e}")

    db.close()


if __name__ == "__main__":
    main()