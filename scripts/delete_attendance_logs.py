import os
import sys
import argparse

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from psycopg_pool import ConnectionPool

# Mặc định số phút để xoá (bạn có thể sửa trực tiếp biến này)
DEFAULT_MINUTES = 1

def main():
    parser = argparse.ArgumentParser(description="Xoá lịch sử check-in / check-out trong khoảng thời gian gần nhất")
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES, help=f"Số phút gần nhất để xoá bản ghi (mặc định: {DEFAULT_MINUTES})")
    parser.add_argument("--yes", action="store_true", help="Bỏ qua xác nhận y/n")
    args = parser.parse_args()

    if args.minutes <= 0:
        print("Số phút phải lớn hơn 0.")
        return

    print(f"Connecting to database...")
    pool = ConnectionPool(config.DB_CONN_INFO, min_size=1, max_size=1)
    
    with pool.connection() as conn:
        try:
            # Count how many logs will be deleted
            count_query = "SELECT COUNT(*) FROM attendance_logs WHERE timestamp > NOW() - make_interval(mins => %s)"
            count = conn.execute(count_query, (args.minutes,)).fetchone()[0]
            
            if count == 0:
                print(f"Không có bản ghi nào trong {args.minutes} phút qua.")
                return

            print(f"CẢNH BÁO: Bạn sắp xoá {count} bản ghi attendance trong {args.minutes} phút gần nhất.")
            if not args.yes:
                confirm = input("Tiếp tục? (y/N): ")
                if confirm.lower() != 'y':
                    print("Đã huỷ.")
                    return
            
            # Execute deletion
            delete_query = "DELETE FROM attendance_logs WHERE timestamp > NOW() - make_interval(mins => %s)"
            conn.execute(delete_query, (args.minutes,))
            conn.commit()
            print(f"Đã xoá thành công {count} bản ghi.")
            
        except Exception as e:
            print(f"Lỗi: {e}")
            conn.rollback()
        finally:
            pool.close()

if __name__ == "__main__":
    main()
