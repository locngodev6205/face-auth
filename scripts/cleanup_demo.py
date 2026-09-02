#!/usr/bin/env python3
"""
Xoá nhanh toàn bộ demo LFW identities (namespace demo_lfw_) khỏi VectorDB.

Dùng sau khi hoàn thành demo hoặc muốn reset dữ liệu test.

Usage:
    python scripts/cleanup_demo.py                          # xoá demo_lfw_* (có confirm)
    python scripts/cleanup_demo.py --dry-run               # xem trước, không xoá
    python scripts/cleanup_demo.py --namespace benchmark_lfw  # xoá namespace khác
    python scripts/cleanup_demo.py --yes                   # bỏ qua confirm (dùng cho script)
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
        description="Xoá toàn bộ demo LFW identities khỏi VectorDB."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=DEFAULT_NAMESPACE,
        help=f"Prefix namespace cần xoá (mặc định: {DEFAULT_NAMESPACE}). "
             "Sẽ xoá tất cả user_id có dạng <namespace>_*.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ hiển thị danh sách sẽ xoá, KHÔNG thực sự xoá.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Bỏ qua bước xác nhận (dùng khi gọi từ script tự động).",
    )
    args = parser.parse_args()

    namespace = args.namespace
    pattern   = f"{namespace}_%"

    db = VectorDB(conninfo=config.DB_CONN_INFO, min_size=1, max_size=2)

    # Đếm trước
    with db.pool.connection() as conn:
        rows = conn.execute(
            "SELECT user_id, name FROM users WHERE user_id LIKE %s ORDER BY name",
            (pattern,),
        ).fetchall()
        emb_count = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings f "
            "JOIN users u ON f.user_id = u.user_id "
            "WHERE u.user_id LIKE %s",
            (pattern,),
        ).fetchone()[0]

    if not rows:
        print(f"[OK] Không tìm thấy identity nào với prefix '{namespace}_'.")
        db.close()
        return

    print(f"Tìm thấy {len(rows)} identity và {emb_count} embedding vectors với prefix '{namespace}_':")
    for user_id, name in rows:
        print(f"  {user_id:45s}  ({name})")

    if args.dry_run:
        print(f"\n[dry-run] Không xoá. Bỏ --dry-run để thực sự xoá.")
        db.close()
        return

    if not args.yes:
        confirm = input(f"\nXác nhận xoá {len(rows)} identity và {emb_count} vectors? [y/N] ").strip().lower()
        if confirm != "y":
            print("Đã huỷ.")
            db.close()
            return

    with db.pool.connection() as conn:
        conn.execute("DELETE FROM users WHERE user_id LIKE %s", (pattern,))
        conn.commit()

    print(f"\n✓ Đã xoá {len(rows)} identity.")
    print("  (ON DELETE CASCADE tự xoá toàn bộ face_embeddings liên quan.)")
    db.close()


if __name__ == "__main__":
    main()
