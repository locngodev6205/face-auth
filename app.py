import cv2
import config
from detection.detector import FaceDetector
from recognition.embedder import FaceEmbedder
from alignment.aligner import get_input_face
from database.vector_db import VectorDB, decide_identity
from tracking.tracker import FaceTracker


# ── Màu sắc cho 3 trạng thái track ─────────────────────────────────────────
COLOR_PENDING    = (180, 180, 180)   # Xám — chưa nhận diện lần nào
COLOR_LOCKED     = (0, 210, 80)      # Xanh lá — đã nhận diện & đang locked
COLOR_REVERIFY   = (0, 165, 255)     # Cam — đang re-verify (trong khoảng cách frame)
COLOR_UNKNOWN    = (0, 40, 220)      # Đỏ — UNKNOWN


def draw_track(frame, bbox, name, score, is_pending, is_reverifying):
    """
    Vẽ bounding box và label lên frame theo trạng thái track.

    Trạng thái:
        - Pending (xám)   : track vừa được tạo, chưa có kết quả nhận diện lần nào.
        - Locked (xanh)   : đã nhận diện, đang trong khoảng cách frame (confident).
        - Re-verify (cam) : đang chờ re-verify nhưng giữ nguyên tên cũ hiển thị.
        - Unknown (đỏ)    : không khớp với ai trong DB.
    """
    x1, y1, x2, y2 = bbox

    if is_pending:
        color = COLOR_PENDING
        label = "Nhan dien..."
    elif name == "UNKNOWN":
        color = COLOR_UNKNOWN
        label = f"UNKNOWN  {score:.2f}"
    elif is_reverifying:
        color = COLOR_REVERIFY
        label = f"{name}  {score:.2f}"
    else:
        # Locked — đã nhận diện & chưa đến frame re-verify
        color = COLOR_LOCKED
        label = f"[OK] {name}  {score:.2f}"

    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Background rect cho label để dễ đọc
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thick = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y1 = max(y1 - text_h - baseline - 4, 0)
    label_y2 = max(y1 - 2, text_h + baseline + 4)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, label_y2), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 3, label_y2 - baseline - 2),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def main():
    # ── LOAD ONCE ────────────────────────────────────────────────────────────
    detector = FaceDetector(
        config.MODEL_PACK_NAME,
        ctx_id=config.MODEL_CTX_ID,
        det_size=config.DETECTOR_DET_SIZE,
        conf_thresh=config.DETECTOR_CONF_THRESH,
    )
    embedder = FaceEmbedder(config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID)
    db = VectorDB(
        conninfo=config.DB_CONN_INFO,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
    )
    tracker = FaceTracker(
        recognize_interval_seconds=config.RECOGNIZE_INTERVAL_SECONDS,
        iou_threshold=config.TRACK_IOU_THRESHOLD,
        max_missing_frames=config.TRACK_MAX_MISSING_FRAMES,
    )

    cap = cv2.VideoCapture(0)

    print("Starting face authentication system...")
    print(f"  MATCH_THRESHOLD    = {config.MATCH_THRESHOLD}")
    print(f"  RECOGNIZE_INTERVAL = {config.RECOGNIZE_INTERVAL_SECONDS}s (time-based)")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            try:
                detections = detector.detect(frame)
            except Exception as e:
                print(f"[Detector error] {e}")
                detections = []

            tracked = tracker.update(detections)

            for detection, track in tracked:
                bbox      = detection["bbox"]
                landmarks = detection.get("landmarks")

                if track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS):
                    try:
                        aligned_face = get_input_face(
                            frame, bbox, landmarks, embedder.input_size
                        )
                        embedding = (
                            embedder.embed_aligned(aligned_face)
                            if aligned_face is not None
                            else None
                        )
                        if embedding is not None:
                            rows = db.search(embedding, top_k=5)
                            name, score = decide_identity(rows, threshold=config.MATCH_THRESHOLD)
                            track.update_result(name, score)
                        # Nếu align/embed thất bại: giữ kết quả cũ, thử lại frame sau
                    except Exception as e:
                        print(f"[Pipeline error] {e}")

                # Xác định trạng thái hiển thị
                name, score = track.name, track.score
                is_pending    = track.is_pending
                is_reverifying = (
                    not is_pending
                    and track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS)
                )

                draw_track(frame, bbox, name, score, is_pending, is_reverifying)

            cv2.imshow("Face Auth Prototype V1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()