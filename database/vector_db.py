from pathlib import Path
from typing import Optional, Tuple, List, Union
from datetime import datetime, timezone
import numpy as np
from psycopg_pool import ConnectionPool

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class VectorDB:
    """
    Module quản lý kết nối PostgreSQL + pgvector với Connection Pooling.
    Quản lý 3 bảng cốt lõi theo schema tối giản:
        1. employees
        2. face_embeddings (SAMPLE & CENTROID)
        3. attendance_logs (cooldown check)
    """

    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 4):
        self.pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size, open=True)
        self.init_schema()

    def init_schema(self):
        """Khởi tạo schema sạch từ database/schema.sql."""
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.pool.connection() as conn:
            # Nếu phát hiện schema prototype cũ (chứa cột user_id thay vì employee_id, hoặc bảng users)
            # tự động dọn dẹp các bảng cũ để tái tạo schema sạch
            legacy = conn.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND ((table_name = 'attendance_logs' AND column_name = 'user_id')
                    OR (table_name = 'face_embeddings' AND column_name = 'user_id')
                    OR (table_name = 'users'))
                LIMIT 1;
            """).fetchone()
            if legacy:
                conn.execute("DROP TABLE IF EXISTS attendance_logs;")
                conn.execute("DROP TABLE IF EXISTS face_embeddings;")
                conn.execute("DROP TABLE IF EXISTS employees;")
                conn.execute("DROP TABLE IF EXISTS users;")

            conn.execute(schema_sql)
            conn.commit()

    @staticmethod
    def _to_vector_literal(embedding) -> str:
        return "[" + ",".join(f"{float(x):.8f}" for x in np.asarray(embedding).flatten()) + "]"

    def search(
        self,
        embedding: Union[np.ndarray, list],
        top_k: int = 5,
        model_version: str = "buffalo_s",
    ) -> List[Tuple[str, str, float]]:
        """
        Tìm kiếm 1:N nhận diện khuôn mặt.
        Chỉ tìm kiếm trên các vector thỏa mãn:
            - embedding_type = 'CENTROID'
            - model_version = model đang chạy
            - employee.status = 'ACTIVE'
        """
        vec = self._to_vector_literal(embedding)
        with self.pool.connection() as conn:
            query = """
                SELECT
                    e.employee_id,
                    e.full_name,
                    1 - (f.embedding <=> %s::vector) AS similarity
                FROM face_embeddings f
                JOIN employees e ON f.employee_id = e.employee_id
                WHERE f.embedding_type = 'CENTROID'
                  AND f.model_version = %s
                  AND e.status = 'ACTIVE'
                ORDER BY f.embedding <=> %s::vector
                LIMIT %s;
            """
            rows = conn.execute(query, (vec, model_version, vec, top_k)).fetchall()
        return rows

    def upsert(
        self,
        employee_id: str,
        full_name: str,
        individual_embeddings: Optional[List[np.ndarray]] = None,
        mean_embedding: Optional[np.ndarray] = None,
        overwrite: bool = True,
        save_individuals: bool = True,
        model_version: str = "buffalo_s",
    ) -> Tuple[str, bool, bool]:
        """
        Đăng ký hoặc cập nhật dữ liệu sinh trắc học khuôn mặt cho nhân viên.
        Thực hiện trong 1 transaction duy nhất:
            1. Upsert bảng employees (ACTIVE)
            2. Xoá toàn bộ embeddings cũ của employee với cùng model_version
            3. Insert các embedding SAMPLE mới
            4. Insert 1 embedding CENTROID mới
        Trả về tuple: (employee_id, is_new, is_ignored)
        """
        if not employee_id or not full_name:
            raise ValueError("Thiếu employee_id hoặc full_name khi đăng ký nhân viên.")

        if mean_embedding is None:
            raise ValueError("Thiếu mean_embedding khi đăng ký khuôn mặt.")

        with self.pool.connection() as conn:
            with conn.transaction():
                # 1. Upsert employee
                conn.execute("""
                    INSERT INTO employees (employee_id, full_name, status, updated_at)
                    VALUES (%s, %s, 'ACTIVE', NOW())
                    ON CONFLICT (employee_id) DO UPDATE SET
                        full_name = EXCLUDED.full_name,
                        status = 'ACTIVE',
                        updated_at = NOW();
                """, (employee_id, full_name))

                # Kiểm tra nhân viên đã có embedding cùng model_version trước đó chưa
                existing = conn.execute("""
                    SELECT 1 FROM face_embeddings
                    WHERE employee_id = %s AND model_version = %s
                    LIMIT 1;
                """, (employee_id, model_version)).fetchone()
                is_new = not bool(existing)

                if existing and not overwrite:
                    return employee_id, is_new, True

                # 2. Xoá embeddings cũ của nhân viên với cùng model_version
                conn.execute("""
                    DELETE FROM face_embeddings
                    WHERE employee_id = %s AND model_version = %s;
                """, (employee_id, model_version))

                # 3. Insert các individual sample embeddings
                if save_individuals and individual_embeddings:
                    for emb in individual_embeddings:
                        vec = self._to_vector_literal(emb)
                        conn.execute("""
                            INSERT INTO face_embeddings (
                                employee_id, embedding, embedding_type, model_version
                            )
                            VALUES (%s, %s::vector, 'SAMPLE', %s);
                        """, (employee_id, vec, model_version))

                # 4. Insert mean centroid embedding
                mean_vec = self._to_vector_literal(mean_embedding)
                conn.execute("""
                    INSERT INTO face_embeddings (
                        employee_id, embedding, embedding_type, model_version
                    )
                    VALUES (%s, %s::vector, 'CENTROID', %s);
                """, (employee_id, mean_vec, model_version))

        return employee_id, is_new, False

    def log_attendance(
        self,
        employee_id: str,
        action: str = "CHECKIN",
        gap_minutes: int = 15,
        face_similarity: Optional[float] = None,
        liveness_score: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[datetime]]:
        """
        Ghi nhận lịch sử điểm danh với kiểm tra cooldown cho demo 1 tiến trình.
        Trả về tuple: (success: bool, reason: str, timestamp: Optional[datetime])
        """
        if not employee_id:
            return False, "Thiếu employee_id", None

        action = action.upper()
        if action not in ("CHECKIN", "CHECKOUT"):
            return False, f"Action không hợp lệ: {action}", None

        try:
            with self.pool.connection() as conn:
                # 1. Kiểm tra Cooldown khoảng thời gian gap_minutes
                if gap_minutes > 0:
                    existing = conn.execute("""
                        SELECT timestamp FROM attendance_logs
                        WHERE employee_id = %s AND action = %s
                          AND timestamp > NOW() - make_interval(mins => %s)
                        ORDER BY timestamp DESC
                        LIMIT 1;
                    """, (employee_id, action, gap_minutes)).fetchone()

                    if existing:
                        last_ts = existing[0]
                        if last_ts and last_ts.tzinfo is None:
                            last_ts = last_ts.replace(tzinfo=timezone.utc)
                        return False, f"Đã {action} gần đây. Vui lòng đợi {gap_minutes} phút.", last_ts

                # 2. Ghi nhận log mới
                inserted = conn.execute("""
                    INSERT INTO attendance_logs (
                        employee_id, action, face_similarity, liveness_score
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING timestamp;
                """, (employee_id, action, face_similarity, liveness_score)).fetchone()

                last_ts = inserted[0]
                if last_ts and last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)

                return True, "Thành công", last_ts
        except Exception as e:
            return False, f"Lỗi database: {e}", None

    def close(self):
        self.pool.close()


def decide_identity(rows: List[Tuple], threshold: float) -> Tuple[Optional[str], str, float]:
    """
    Quyết định danh tính người dùng dựa trên kết quả tìm kiếm vector và ngưỡng so khớp.
    rows: danh sách các hàng (employee_id, full_name, similarity) sắp xếp giảm dần theo similarity.
    """
    if not rows:
        return None, "UNKNOWN", 0.0

    best_id, best_name, best_similarity = rows[0]

    if best_similarity >= threshold:
        return str(best_id), str(best_name), float(best_similarity)

    return None, "UNKNOWN", float(best_similarity)