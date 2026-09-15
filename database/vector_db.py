from typing import Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from psycopg_pool import ConnectionPool


class VectorDB:
    """
    Dùng connection pool thay vì psycopg.connect() mới mỗi lần search/upsert.
    Mở connection mới trong vòng lặp webcam (mỗi frame) sẽ tốn TCP handshake +
    auth mỗi lần — đủ để giết FPS, đặc biệt khi DB nằm trên cloud (Supabase/
    Neon/RDS) có latency mạng cao hơn local.
    """

    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 4):
        self.pool = ConnectionPool(conninfo, min_size=min_size, max_size=max_size, open=True)

        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    embedding VECTOR(512) NOT NULL,
                    is_mean BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    log_id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    action TEXT NOT NULL CHECK (action IN ('CHECKIN', 'CHECKOUT')),
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_attendance_lookup 
                ON attendance_logs (user_id, action, timestamp DESC);
            """)
            conn.commit()

    @staticmethod
    def _to_vector_literal(embedding):
        return "[" + ",".join(f"{float(x):.8f}" for x in np.asarray(embedding).flatten()) + "]"

    def search(self, embedding, top_k=5, mean_only=True):
        vec = self._to_vector_literal(embedding)
        with self.pool.connection() as conn:
            query = """
                SELECT
                    u.user_id,
                    u.name,
                    1 - (f.embedding <=> %s::vector) AS similarity
                FROM face_embeddings f
                JOIN users u ON f.user_id = u.user_id
            """
            if mean_only:
                query += " WHERE f.is_mean = TRUE"
                
            query += """
                ORDER BY f.embedding <=> %s::vector
                LIMIT %s;
            """
            rows = conn.execute(query, (vec, vec, top_k)).fetchall()
        return rows

    def upsert(self, user_id, name, individual_embeddings, mean_embedding, overwrite=True, save_individuals=True):
        with self.pool.connection() as conn:
            # Upsert user
            conn.execute("""
                INSERT INTO users (user_id, name)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET name = EXCLUDED.name;
            """, (user_id, name))
            
            # Check existing embeddings
            existing = conn.execute(
                "SELECT 1 FROM face_embeddings WHERE user_id = %s LIMIT 1", (user_id,)
            ).fetchone()
            
            is_new = not bool(existing)
            is_ignored = False
            
            if existing:
                if not overwrite:
                    is_ignored = True
                    conn.commit()
                    return user_id, is_new, is_ignored
                
                # Delete old embeddings if overwrite
                conn.execute("DELETE FROM face_embeddings WHERE user_id = %s", (user_id,))
            
            # Insert individual embeddings
            if save_individuals:
                for i, emb in enumerate(individual_embeddings, start=1):
                    vec = self._to_vector_literal(emb)
                    emb_id = f"{user_id}_{i:04d}"
                    conn.execute("""
                        INSERT INTO face_embeddings (embedding_id, user_id, embedding, is_mean)
                        VALUES (%s, %s, %s::vector, FALSE)
                    """, (emb_id, user_id, vec))
                
            # Insert mean embedding
            mean_vec = self._to_vector_literal(mean_embedding)
            mean_id = f"{user_id}_0000"
            conn.execute("""
                INSERT INTO face_embeddings (embedding_id, user_id, embedding, is_mean)
                VALUES (%s, %s, %s::vector, TRUE)
            """, (mean_id, user_id, mean_vec))
            
            conn.commit()
            return user_id, is_new, is_ignored

    def log_attendance(
        self, user_id: Optional[str], action: str, gap_minutes: int
    ) -> Tuple[bool, str, Optional[datetime]]:
        """
        Ghi nhận lịch sử điểm danh.
        Trả về tuple: (success: bool, reason: str, timestamp: Optional[datetime])
          - Thành công: (True, "Thành công", timestamp_moi_ghi)
          - Bị chặn do cooldown: (False, "Đã ... gần đây...", timestamp_gan_nhat) -> dùng cho local timer
          - Lỗi hoặc thiếu tham số: (False, reason, None)
        """
        if not user_id:
            return False, "Thiếu user_id", None

        action = action.upper()
        if action not in ("CHECKIN", "CHECKOUT"):
            return False, f"Action không hợp lệ: {action}", None

        try:
            with self.pool.connection() as conn:
                # Check for recent identical action
                existing = conn.execute("""
                    SELECT timestamp FROM attendance_logs 
                    WHERE user_id = %s AND action = %s AND timestamp > NOW() - make_interval(mins => %s)
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (user_id, action, gap_minutes)).fetchone()
                
                if existing:
                    last_ts = existing[0]
                    if last_ts and last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    return False, f"Đã {action} gần đây. Vui lòng đợi {gap_minutes} phút.", last_ts
                    
                inserted = conn.execute("""
                    INSERT INTO attendance_logs (user_id, action)
                    VALUES (%s, %s)
                    RETURNING timestamp
                """, (user_id, action)).fetchone()
                conn.commit()
                last_ts = inserted[0]
                if last_ts and last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                return True, "Thành công", last_ts
        except Exception as e:
            return False, f"Lỗi database: {e}", None

    def close(self):
        self.pool.close()


def decide_identity(rows, threshold: float):
    if not rows:
        return None, "UNKNOWN", 0.0

    best_id, best_name, best_similarity = rows[0]

    if best_similarity >= threshold:
        return best_id, best_name, float(best_similarity)

    return None, "UNKNOWN", float(best_similarity)