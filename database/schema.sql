-- =============================================================================
-- SmartFace AIoT — Clean Core Schema (3 Tables)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Bảng nhân viên
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Bảng vector đặc trưng khuôn mặt (Face Embeddings)
CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id BIGSERIAL PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
    embedding VECTOR(512) NOT NULL,
    embedding_type TEXT NOT NULL CHECK (embedding_type IN ('SAMPLE', 'CENTROID')),
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Bảng lịch sử điểm danh (Attendance Logs)
CREATE TABLE IF NOT EXISTS attendance_logs (
    log_id BIGSERIAL PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id) ON DELETE RESTRICT,
    action TEXT NOT NULL CHECK (action IN ('CHECKIN', 'CHECKOUT')),
    face_similarity REAL,
    liveness_score REAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index tối ưu truy vấn cooldown điểm danh
CREATE INDEX IF NOT EXISTS idx_attendance_lookup 
ON attendance_logs (employee_id, action, timestamp DESC);
