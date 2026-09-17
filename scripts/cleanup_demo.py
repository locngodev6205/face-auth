#!/usr/bin/env python3
"""
Dọn dẹp / xoá demo identities theo namespace khỏi database.

Luồng xử lý:
    1. Tìm dữ liệu thuộc namespace (<namespace>_*).
    2. Hiển thị số employee, embeddings và attendance.
    3. Yêu cầu xác nhận nếu không có --yes.
    4. Trong 1 transaction:
       - Xóa attendance_logs thuộc namespace
       - Xóa employees thuộc namespace (ON DELETE CASCADE tự xóa face_embeddings)

Usage:
    python scripts/cleanup_demo.py --namespace demo_lfw
    python scripts/cleanup_demo.py --namespace demo_lfw --dry-run
    python scripts/cleanup_demo.py --namespace demo_lfw --yes
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database.vector_db import VectorDB

DEFAULT_NAMESPACE = "demo_lfw"


def main():
    parser = argparse.ArgumentParser(
        description="Dọn dẹp demo identities theo namespace khỏi database."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=DEFAULT_NAMESPACE,
        help=f"Prefix namespace cần dọn dẹp (mặc định: {DEFAULT_NAMESPACE}). "
             "Sẽ lọc tất cả employee_id có dạng <namespace>_*.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ hiển thị danh sách sẽ xử lý, không thực hiện thay đổi.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Bỏ qua bước xác nhận interactive.",
    )
    args = parser.parse_args()

    namespace = args.namespace
    pattern = f"{namespace}_%"

    db = VectorDB(conninfo=config.DB_CONN_INFO, min_size=1, max_size=2)

    with db.pool.connection() as conn:
        rows = conn.execute(
            "SELECT employee_id, full_name FROM employees WHERE employee_id LIKE %s ORDER BY full_name",
            (pattern,),
        ).fetchall()
        emb_count = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings f "
            "JOIN employees e ON f.employee_id = e.employee_id "
            "WHERE e.employee_id LIKE %s",
            (pattern,),
        ).fetchone()[0]
        att_count = conn.execute(
            "SELECT COUNT(*) FROM attendance_logs WHERE employee_id LIKE %s",
            (pattern,),
        ).fetchone()[0]

    if not rows:
        print(f"[OK] Không tìm thấy dữ liệu nào với prefix '{namespace}_'.")
        db.close()
        return

    print(f"Tìm thấy dữ liệu thuộc namespace '{namespace}':")
    print(f"  - Employees   : {len(rows)}")
    print(f"  - Embeddings  : {emb_count}")
    print(f"  - Attendance  : {att_count}")

    if args.dry_run:
        print("\n[dry-run] Không thực hiện thay đổi. Bỏ --dry-run để xóa thật.")
        db.close()
        return

    if not args.yes:
        confirm = input(f"\nXác nhận xóa dữ liệu của {len(rows)} nhân viên namespace '{namespace}'? [y/N] ").strip().lower()
        if confirm != "y":
            print("Đã huỷ thao tác.")
            db.close()
            return

    with db.pool.connection() as conn:
        with conn.transaction():
            if att_count > 0:
                conn.execute("DELETE FROM attendance_logs WHERE employee_id LIKE %s", (pattern,))
                print(f"  ✓ Đã xoá {att_count} bản ghi attendance_logs.")
            conn.execute("DELETE FROM employees WHERE employee_id LIKE %s", (pattern,))
            print(f"  ✓ Đã xoá {len(rows)} nhân viên (CASCADE tự xoá {emb_count} embeddings).")

    db.close()
    print("[SUCCESS] Hoàn tất dọn dẹp namespace.")


if __name__ == "__main__":
    main()
