import os
import cv2
import sys
import argparse
import random
from uuid import uuid4

# Thêm thư mục gốc vào sys.path để có thể import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def ensure_test_employee(db, folder_name, create_missing=True):
    """Resolve a code/name, or create a test-only PBL6 employee with a real name."""
    with db.pool.connection() as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT employee_code, full_name FROM employees WHERE employee_code = %s",
                (folder_name,),
            ).fetchone()
            if row:
                return row[0], row[1], False
            rows = conn.execute(
                "SELECT employee_code, full_name FROM employees WHERE full_name = %s LIMIT 2",
                (folder_name,),
            ).fetchall()
            if len(rows) > 1:
                raise ValueError("Tên trùng nhiều nhân viên; hãy đặt tên thư mục bằng employee_code")
            if rows:
                return rows[0][0], rows[0][1], False
            if not create_missing:
                raise ValueError(f"Không tìm thấy nhân viên '{folder_name}' (--existing-only)")
            if not folder_name.strip() or len(folder_name) > 100:
                raise ValueError("Tên nhân viên phải có từ 1 đến 100 ký tự")

            token = uuid4().hex
            rng = random.SystemRandom()
            department = conn.execute("SELECT id FROM departments ORDER BY random() LIMIT 1").fetchone()
            row = conn.execute("""
                INSERT INTO employees (
                    id, employee_code, full_name, department_id,
                    position, phone, email, hourly_rate, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE')
                RETURNING employee_code, full_name
            """, (
                str(uuid4()), f"TEST-{token}", folder_name,
                department[0] if department else None,
                rng.choice(["Nhân viên", "Thu ngân", "Bảo vệ"]),
                f"09{rng.randrange(100000000):08d}",
                f"test-{token}@example.invalid", rng.randint(20, 80) * 1000,
            )).fetchone()
            return row[0], row[1], True


def main():
    parser = argparse.ArgumentParser(description="Đăng ký dữ liệu khuôn mặt vào Database từ thư mục gallery")
    parser.add_argument("--gallery", type=str, default="gallery", help="Đường dẫn đến thư mục chứa ảnh (mặc định: gallery)")
    parser.add_argument("--ignore-duplicate", action="store_true", help="Bỏ qua nếu khuôn mặt đã đăng ký cùng model (mặc định: ghi đè)")
    parser.add_argument("--no-individuals", action="store_true", help="Không lưu các vector thành phần (mặc định: có lưu)")
    parser.add_argument("--existing-only", action="store_true", help="Chỉ enroll nhân viên có sẵn; mặc định tạo hồ sơ TEST khi thiếu")
    args = parser.parse_args()

    # Load model dependencies only after parsing CLI (including --help).
    from detection.detector import FaceDetector
    from recognition.embedder import FaceEmbedder
    from alignment.aligner import get_input_face
    from database.vector_db import VectorDB
    from enrollment.enroll import enroll_person

    gallery_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.gallery)

    if not os.path.exists(gallery_dir):
        print(f"Thư mục '{gallery_dir}' không tồn tại. Đang tiến hành tạo thư mục...")
        os.makedirs(gallery_dir)
        print("Vui lòng thêm dữ liệu vào thư mục này rồi chạy lại script.")
        return

    print("Đang khởi tạo model và kết nối database...")
    db = VectorDB(
        conninfo=config.DB_CONN_INFO,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
    )

    try:
        # SCRFD: det_size kiểm soát độ phân giải inference nội bộ; conf_thresh tương đương DETECTOR_CONF_THRESH
        detector = FaceDetector(
            model_name=config.MODEL_PACK_NAME,
            ctx_id=config.MODEL_CTX_ID,
            det_size=config.DETECTOR_DET_SIZE,
            conf_thresh=config.DETECTOR_CONF_THRESH,
        )
        embedder = FaceEmbedder(config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID)
        print(f"Bắt đầu quét thư mục gallery: {gallery_dir}")
        failures = 0
        completed = 0

        for person_name in sorted(os.listdir(gallery_dir)):
            person_dir = os.path.join(gallery_dir, person_name)

            if not os.path.isdir(person_dir):
                continue

            print(f"\nĐang xử lý dữ liệu cho nhân viên: {person_name}")
            embeddings = []

            for filename in sorted(os.listdir(person_dir)):
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
                    print(f"  [!] Bỏ qua {filename}: có {len(detections)} khuôn mặt, cần ảnh chỉ có một người.")
                    continue
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
                print(f"==> Bỏ qua nhân viên '{person_name}' - Không có ảnh hợp lệ nào.")
                failures += 1
                continue

            try:
                employee_code, full_name, created = ensure_test_employee(
                    db, person_name, create_missing=not args.existing_only,
                )
                if created:
                    print(f"  [TEST] Đã tạo nhân viên '{full_name}' ({employee_code}); các trường khác là dữ liệu mẫu.")
                overwrite = not args.ignore_duplicate
                save_individuals = not args.no_individuals
                new_id, is_new, is_ignored, warnings = enroll_person(
                    db,
                    employee_code=employee_code,
                    full_name=full_name,
                    embeddings=embeddings,
                    overwrite=overwrite,
                    save_individuals=save_individuals,
                    model_version=config.EMBEDDING_MODEL_VERSION,
                )

                for w in warnings:
                    print(f"  [!] {w}")

                if is_ignored:
                    action = "Bỏ qua (khuôn mặt đã đăng ký)"
                else:
                    action = "Đăng ký khuôn mặt mới" if is_new else "Cập nhật khuôn mặt (ghi đè)"

                print(f"==> {action} '{person_name}' với Employee ID: {new_id} (Dựa trên {len(embeddings)} ảnh)")
                completed += 1
            except Exception as e:
                failures += 1
                print(f"==> Lỗi khi đăng ký nhân viên '{person_name}': {e}")

        print(f"Hoàn tất: {completed} danh tính; lỗi/bỏ qua do ảnh không hợp lệ: {failures}.")
        return 1 if failures or not completed else 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
