"""Preview/remove only test employees created by run_enroll.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import psycopg


def delete_test_identities(conn, apply=False):
    with conn.transaction():
        rows = conn.execute("""
            SELECT e.id, e.employee_code, e.full_name,
                EXISTS (SELECT 1 FROM attendance_logs a WHERE a.employee_id=e.id)
                OR EXISTS (SELECT 1 FROM daily_attendance_summary a WHERE a.employee_id=e.id)
                OR EXISTS (SELECT 1 FROM payroll_records p WHERE p.employee_id=e.id)
                OR EXISTS (SELECT 1 FROM employee_shifts s WHERE s.employee_id=e.id)
                AS has_business_data
            FROM employees e
            WHERE e.employee_code ~ '^TEST-[0-9a-f]{32}$'
              AND e.email = 'test-' || substring(e.employee_code FROM 6) || '@example.invalid'
              AND e.user_id IS NULL
            ORDER BY e.employee_code
            FOR UPDATE OF e
        """).fetchall()
        removable = [row for row in rows if not row[3]]
        if apply:
            for employee_id, _, _, _ in removable:
                # The employee FK cascades deletion to SAMPLE/CENTROID embeddings.
                conn.execute('DELETE FROM employees WHERE id = %s', (employee_id,))
        return rows, len(removable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes', action='store_true', help='Thực hiện xóa; mặc định chỉ xem danh sách')
    args = parser.parse_args()
    with psycopg.connect(config.DB_CONN_INFO, connect_timeout=5) as conn:
        rows, count = delete_test_identities(conn, apply=args.yes)
    for _, code, name, protected in rows:
        action = 'GIỮ: có dữ liệu nghiệp vụ' if protected else ('ĐÃ XÓA' if args.yes else 'SẼ XÓA')
        print(f'{action}: {code} — {name}')
    print(f'{"Đã xóa" if args.yes else "Có thể xóa"}: {count}; giữ lại: {len(rows)-count}')
    if not args.yes:
        print('Chạy lại với --yes để xóa cả hồ sơ test và các embeddings liên quan.')


if __name__ == '__main__':
    main()
