import base64
import sys
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from alignment.aligner import get_input_face
from database.vector_db import VectorDB
from detection.detector import FaceDetector
from enrollment.enroll import build_identity_embedding
from recognition.embedder import FaceEmbedder

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI(
    title="SmartFace AIoT — Internal Face Auth Service",
    description="Dịch vụ trích xuất vector khuôn mặt và quản lý sinh trắc học nội bộ",
    version="1.0.0",
)

# Global model & DB singletons
detector: Optional[FaceDetector] = None
embedder: Optional[FaceEmbedder] = None
db: Optional[VectorDB] = None


@app.on_event("startup")
def startup_event():
    global detector, embedder, db
    print("[api_service] Loading models...")
    detector = FaceDetector(
        model_name=config.MODEL_PACK_NAME,
        ctx_id=config.MODEL_CTX_ID,
        det_size=config.DETECTOR_DET_SIZE,
        conf_thresh=config.DETECTOR_CONF_THRESH,
        min_face_size=config.DETECTOR_MIN_FACE_SIZE,
    )
    print(f"[api_service] Detector loaded: {config.APP_DETECTOR}")
    embedder = FaceEmbedder(
        model_name=config.MODEL_PACK_NAME,
        ctx_id=config.MODEL_CTX_ID,
    )
    print(f"[api_service] Embedder loaded: {config.MODEL_PACK_NAME}")
    db = VectorDB(
        conninfo=config.DB_CONN_INFO,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
    )
    print(f"[api_service] VectorDB pool ready: min={config.DB_POOL_MIN_SIZE}, max={config.DB_POOL_MAX_SIZE}")
    print("[api_service] Ready on :5000")


@app.on_event("shutdown")
def shutdown_event():
    global db
    if db is not None:
        db.close()
        print("[CLEANUP] Đã đóng kết nối VectorDB pool.")


class EnrollRequest(BaseModel):
    employee_code: str = Field(..., min_length=1, max_length=50)
    images_base64: List[str] = Field(..., min_length=1, max_length=10)


@app.get("/health")
def health_check():
    """Health check endpoint để Backend kiểm tra trạng thái hoạt động."""
    if detector is None or embedder is None or db is None:
        raise HTTPException(status_code=503, detail="Face Auth chưa sẵn sàng")
    try:
        db.check_connection()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Face Auth database is not ready: {type(exc).__name__}",
        ) from exc
    return {
        "status": "ready",
        "model_version": config.EMBEDDING_MODEL_VERSION,
    }


@app.post("/internal/enroll")
def enroll_faces(req: EnrollRequest):
    """
    Trích xuất vector khuôn mặt từ danh sách ảnh base64 và lưu vào database.
    """
    if detector is None or embedder is None or db is None:
        raise HTTPException(status_code=503, detail="Dịch vụ chưa sẵn sàng, đang khởi tạo model...")

    embeddings = []
    failed_count = 0

    for b64_str in req.images_base64:
        try:
            # Xử lý Data URI prefix nếu có (ví dụ: data:image/jpeg;base64,...)
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                failed_count += 1
                continue
        except Exception:
            failed_count += 1
            continue

        detections = detector.detect(img)
        if len(detections) == 0:
            failed_count += 1
            continue

        # Nếu có nhiều hơn 1 khuôn mặt, bỏ qua ảnh này
        if len(detections) > 1:
            failed_count += 1
            continue

        best = max(detections, key=lambda d: d.get("score", 0.0))
        aligned = get_input_face(img, best["bbox"], best.get("landmarks"), embedder.input_size)
        if aligned is None:
            failed_count += 1
            continue

        emb = embedder.embed_aligned(aligned)
        if emb is not None:
            embeddings.append(emb)
        else:
            failed_count += 1

    if len(embeddings) == 0:
        raise HTTPException(status_code=400, detail="Không phát hiện được khuôn mặt hợp lệ nào trong các ảnh đã gửi")

    centroid, warnings, valid_embeddings = build_identity_embedding(embeddings, outlier_threshold=0.35)

    total_images = len(req.images_base64)
    detected_faces = len(embeddings)
    n_samples_used = len(valid_embeddings)
    n_outliers_removed = len(embeddings) - len(valid_embeddings)
    extraction_rate = round((detected_faces / total_images) * 100, 1) if total_images > 0 else 0.0
    valid_rate = round((n_samples_used / total_images) * 100, 1) if total_images > 0 else 0.0

    if failed_count > 0:
        warnings.append(f"{failed_count}/{total_images} ảnh không trích xuất được khuôn mặt")

    try:
        db.upsert(
            employee_code=req.employee_code,
            individual_embeddings=valid_embeddings,
            mean_embedding=centroid,
            model_version=config.EMBEDDING_MODEL_VERSION,
            overwrite=True,
            save_individuals=True,
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu embeddings: {e}")

    return {
        "status": "ok",
        "employee_code": req.employee_code,
        "total_images": total_images,
        "detected_faces": detected_faces,
        "n_samples_used": n_samples_used,
        "n_outliers_removed": n_outliers_removed,
        "extraction_rate": extraction_rate,
        "valid_rate": valid_rate,
        "message": f"Trích xuất thành công {detected_faces}/{total_images} khung hình khuôn mặt (tỉ lệ: {extraction_rate}%).",
        "warnings": warnings,
    }



@app.delete("/internal/enroll/{employee_code}")
def delete_employee_faces(employee_code: str):
    """
    Xóa toàn bộ vector sinh trắc học của nhân viên theo employee_code.
    KHÔNG đổi employees.status — Backend quản lý việc này.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Dịch vụ chưa sẵn sàng")

    with db.pool.connection() as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT id FROM employees WHERE employee_code = %s;",
                (employee_code,)
            ).fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Employee not found")

            emp_uuid = row[0]
            conn.execute(
                "DELETE FROM face_embeddings WHERE employee_id = %s;",
                (emp_uuid,)
            )

    return {"status": "ok", "employee_code": employee_code}
