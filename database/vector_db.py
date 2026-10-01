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

    @staticmethod
    def _to_vector_literal(embedding) -> str:
        return "[" + ",".join(f"{float(x):.8f}" for x in np.asarray(embedding).flatten()) + "]"

    def search(
        self,
        embedding: Union[np.ndarray, list],
        top_k: int = 5,
        model_version: str = "arcface_v1",
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
        model_version: str = "arcface_v1",
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
        if not employee_code:
            raise ValueError("Thiếu employee_code khi đăng ký khuôn mặt.")

        if mean_embedding is None:
            raise ValueError("Thiếu mean_embedding khi đăng ký khuôn mặt.")

        with self.pool.connection() as conn:
            with conn.transaction():
                # 1. Lookup UUID từ employee_code
                row = conn.execute("""
                    SELECT id FROM employees WHERE employee_code = %s;
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

    if best_similarity >= threshold:
        return str(best_code), str(best_name), float(best_similarity)

    return None, "UNKNOWN", float(best_similarity)