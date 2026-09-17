#!/usr/bin/env python3
"""
Script reset database cho module face_auth.

Thao tác thực hiện:
    1. Kết nối và hiển thị database hiện tại.
    2. Yêu cầu xác nhận (trừ khi có cờ --yes).
    3. Drop đúng 4 bảng theo thứ tự dependency (không dùng CASCADE rộng):
       - attendance_logs
       - face_embeddings
       - employees
       - users (bảng prototype cũ nếu còn)
    4. Tái tạo 3 bảng sạch từ database/schema.sql.
"""

import os
import sys
import argparse
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from psycopg_pool import ConnectionPool

SCHEMA_PATH = Path(__file__).parent.parent / "database" / "schema.sql"


def reset_database(pool: ConnectionPool, skip_confirm: bool = False) -> bool:
    with pool.connection() as conn:
        curr_db = conn.execute("SELECT current_database();").fetchone()[0]
        print(f"[INFO] Kết nối tới database: '{curr_db}'")

        if not skip_confirm:
            ans = input(f"CẢNH BÁO: Bạn có chắc chắn muốn xoá sạch dữ liệu face_auth trong '{curr_db}' và tạo lại 3 bảng? (y/N): ")
            if ans.strip().lower() != "y":
                print("[INFO] Đã huỷ thao tác.")
                return False

        print("[INFO] Đang drop các bảng theo thứ tự dependency...")
        # Drop tuần tự, không dùng CASCADE rộng
        conn.execute("DROP TABLE IF EXISTS attendance_logs;")
        conn.execute("DROP TABLE IF EXISTS face_embeddings;")
        conn.execute("DROP TABLE IF EXISTS employees;")
        conn.execute("DROP TABLE IF EXISTS users;")

        print("[INFO] Đang tạo lại schema sạch từ database/schema.sql...")
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.execute(schema_sql)
        conn.commit()

        print("[SUCCESS] Đã reset database thành công với đúng 3 bảng sạch:")
        print("  - employees")
        print("  - face_embeddings")
        print("  - attendance_logs")
        return True


def main():
    parser = argparse.ArgumentParser(description="Reset dữ liệu face_auth và khởi tạo 3 bảng sạch.")
    parser.add_argument("--reset", action="store_true", help="Bắt buộc: xác nhận thực hiện reset database")
    parser.add_argument("--yes", "-y", action="store_true", help="Bỏ qua câu hỏi xác nhận (non-interactive)")
    args = parser.parse_args()

    if not args.reset:
        print("[ERROR] Bắt buộc truyền cờ --reset để thực hiện reset database.", file=sys.stderr)
        sys.exit(1)

    pool = ConnectionPool(config.DB_CONN_INFO, min_size=1, max_size=1, open=True)
    try:
        success = reset_database(pool, skip_confirm=args.yes)
        if not success:
            sys.exit(1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
