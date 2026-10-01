import math
from typing import Optional, Tuple, List, Union
import numpy as np
from psycopg_pool import ConnectionPool


class VectorDB:
    """
    Module quản lý kết nối PostgreSQL + pgvector với Connection Pooling.
    Đọc/ghi face_embeddings theo schema Backend (Prisma).
    KHÔNG tạo bảng, KHÔNG ghi attendance_logs.
    """

    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 4):
        self.pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size, open=True)
        try:
            self.validate_schema()
        except Exception:
            self.pool.close()
            raise

    @staticmethod
    def _to_vector_literal(embedding) -> str:
        values = np.asarray(embedding, dtype=np.float64).flatten()
        if values.size != 512 or not np.isfinite(values).all():
            raise ValueError("Face embedding phải có đúng 512 giá trị hữu hạn")
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Face embedding phải có norm hữu hạn và khác 0")
        values /= norm
        return "[" + ",".join(f"{x:.8f}" for x in values) + "]"

    def validate_schema(self) -> None:
        """Fail at startup when this service is connected to the legacy/wrong DB."""
        expected = {
            ("employees", "id"): "uuid",
            ("employees", "employee_code"): "character varying(50)",
            ("employees", "full_name"): "character varying(100)",
            ("employees", "status"): "character varying(20)",
            ("face_embeddings", "employee_id"): "uuid",
            ("face_embeddings", "embedding"): "vector(512)",
            ("face_embeddings", "model_version"): "character varying(50)",
            ("face_embeddings", "is_active"): "boolean",
            ("face_embeddings", "embedding_type"): "character varying(20)",
        }
        with self.pool.connection() as conn:
            rows = conn.execute("""
                SELECT c.relname, a.attname,
                       format_type(a.atttypid, a.atttypmod)
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_attribute a ON a.attrelid = c.oid
                WHERE n.nspname = 'public'
                  AND c.relname IN ('employees', 'face_embeddings')
                  AND a.attnum > 0 AND NOT a.attisdropped;
            """).fetchall()
        actual = {(table, column): data_type for table, column, data_type in rows}
        problems = [
            f"{table}.{column} expected {data_type}, got {actual.get((table, column), 'missing')}"
            for (table, column), data_type in expected.items()
            if actual.get((table, column)) != data_type
        ]
        if problems:
            raise RuntimeError(
                "Face Auth requires the PBL6 Prisma schema; "
                "apply the face_embeddings repair migration: " + "; ".join(problems)
            )

    def check_connection(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT 1")

    def search(
        self,
        embedding: Union[np.ndarray, list],
        top_k: int = 5,
        model_version: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Tìm kiếm 1:N nhận diện khuôn mặt.
        Chỉ tìm kiếm trên các vector thỏa mãn:
            - embedding_type = 'CENTROID'
            - model_version = model đang chạy
            - is_active = true
            - employee.status = 'ACTIVE'

        Trả về: List[(employee_code, full_name, similarity)]
        """
        if model_version is None:
            import config
            model_version = config.EMBEDDING_MODEL_VERSION
        vec = self._to_vector_literal(embedding)
        with self.pool.connection() as conn:
            query = """
                SELECT
                    e.employee_code,
                    e.full_name,
                    1 - (f.embedding <=> %s::vector) AS similarity
                FROM face_embeddings f
                JOIN employees e ON e.id = f.employee_id
                WHERE f.embedding_type = 'CENTROID'
                  AND f.model_version = %s
                  AND f.is_active = true
                  AND e.status = 'ACTIVE'
                  AND f.embedding IS NOT NULL
                ORDER BY f.embedding <=> %s::vector
                LIMIT %s;
            """
            rows = conn.execute(query, (vec, model_version, vec, top_k)).fetchall()
        return rows

    def upsert(
        self,
        employee_code: str,
        individual_embeddings: Optional[List[np.ndarray]] = None,
        mean_embedding: Optional[np.ndarray] = None,
        model_version: Optional[str] = None,
        overwrite: bool = True,
        save_individuals: bool = True,
    ) -> Tuple[str, bool, bool]:
        """
        Đăng ký hoặc cập nhật dữ liệu sinh trắc học khuôn mặt cho nhân viên.
        KHÔNG tạo employee — employee phải đã tồn tại trong DB (do Backend quản lý).

        Thực hiện trong 1 transaction duy nhất:
            1. Lookup employee UUID từ employee_code
            2. Kiểm tra existing embeddings
            3. Xoá embeddings cũ (nếu overwrite)
            4. Insert các embedding SAMPLE mới
            5. Insert 1 embedding CENTROID mới

        Trả về tuple: (employee_code, is_new, is_ignored)
        """
        if model_version is None:
            import config
            model_version = config.EMBEDDING_MODEL_VERSION
        if not employee_code:
            raise ValueError("Thiếu employee_code khi đăng ký khuôn mặt.")

        if mean_embedding is None:
            raise ValueError("Thiếu mean_embedding khi đăng ký khuôn mặt.")

        with self.pool.connection() as conn:
            with conn.transaction():
                # 1. Lookup UUID từ employee_code
                row = conn.execute("""
                    SELECT id FROM employees WHERE employee_code = %s FOR UPDATE;
                """, (employee_code,)).fetchone()

                if not row:
                    raise ValueError(
                        f"Employee '{employee_code}' không tồn tại trong DB. "
                        "Hãy tạo nhân viên qua Backend trước."
                    )
                emp_uuid = row[0]

                # 2. Kiểm tra nhân viên đã có embedding cùng model_version chưa
                existing = conn.execute("""
                    SELECT 1 FROM face_embeddings
                    WHERE employee_id = %s AND model_version = %s
                    LIMIT 1;
                """, (emp_uuid, model_version)).fetchone()
                is_new = not bool(existing)

                if existing and not overwrite:
                    return employee_code, is_new, True

                # 3. Xoá embeddings cũ của nhân viên với cùng model_version
                conn.execute("""
                    DELETE FROM face_embeddings
                    WHERE employee_id = %s AND model_version = %s;
                """, (emp_uuid, model_version))

                # 4. Insert các individual sample embeddings
                if save_individuals and individual_embeddings:
                    for emb in individual_embeddings:
                        vec = self._to_vector_literal(emb)
                        conn.execute("""
                            INSERT INTO face_embeddings (
                                id, employee_id, embedding, embedding_type,
                                model_version, is_active, created_at
                            )
                            VALUES (
                                gen_random_uuid(), %s, %s::vector, 'SAMPLE',
                                %s, true, CURRENT_TIMESTAMP
                            );
                        """, (emp_uuid, vec, model_version))

                # 5. Insert mean centroid embedding
                mean_vec = self._to_vector_literal(mean_embedding)
                conn.execute("""
                    INSERT INTO face_embeddings (
                        id, employee_id, embedding, embedding_type,
                        model_version, is_active, created_at
                    )
                    VALUES (
                        gen_random_uuid(), %s, %s::vector, 'CENTROID',
                        %s, true, CURRENT_TIMESTAMP
                    );
                """, (emp_uuid, mean_vec, model_version))

        return employee_code, is_new, False

    def close(self):
        self.pool.close()


def decide_identity(rows: List[Tuple], threshold: float) -> Tuple[Optional[str], str, float]:
    """
    Quyết định danh tính người dùng dựa trên kết quả tìm kiếm vector và ngưỡng so khớp.
    rows: danh sách các hàng (employee_code, full_name, similarity) sắp xếp giảm dần theo similarity.
    """
    if not rows:
        return None, "UNKNOWN", 0.0

    best_code, best_name, best_similarity = rows[0]
    try:
        best_similarity = float(best_similarity)
    except (TypeError, ValueError):
        return None, "UNKNOWN", 0.0
    if not math.isfinite(best_similarity):
        return None, "UNKNOWN", 0.0

    if best_similarity >= threshold:
        return str(best_code), str(best_name), best_similarity

    return None, "UNKNOWN", best_similarity
