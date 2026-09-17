import os
import sys
import argparse

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import timezone
import config
from psycopg_pool import ConnectionPool

def main():
    parser = argparse.ArgumentParser(description="Xem lịch sử check-in / check-out")
    parser.add_argument("--limit", type=int, default=50, help="Số lượng bản ghi hiển thị (mặc định: 50)")
    args = parser.parse_args()

    print(f"Connecting to database...")
    pool = ConnectionPool(config.DB_CONN_INFO, min_size=1, max_size=1)
    
    with pool.connection() as conn:
        try:
            # Query attendance logs
            query = """
                SELECT 
                    a.log_id,
                    a.timestamp,
                    e.employee_id,
                    e.full_name,
                    a.action,
                    a.face_similarity,
                    a.liveness_score
                FROM attendance_logs a
                JOIN employees e ON a.employee_id = e.employee_id
                ORDER BY a.timestamp DESC
                LIMIT %s
            """
            rows = conn.execute(query, (args.limit,)).fetchall()
            
            if not rows:
                print("Chưa có lịch sử check-in/check-out nào.")
                return

            # Print beautifully
            print(f"\n{'-'*95}")
            print(f"{'TIME':<20} | {'EMPLOYEE ID':<18} | {'NAME':<20} | {'ACTION':<9} | {'SIM':<6} | {'PAD SCORE':<10}")
            print(f"{'-'*95}")
            
            for row in rows:
                log_id, ts, emp_id, name, action, sim, liveness = row
                if ts and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts_str = ts.astimezone().strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                
                # Colors
                action_colored = f"\033[92m{action:<9}\033[0m" if action == 'CHECKIN' else f"\033[93m{action:<9}\033[0m"
                sim_str = f"{sim:.2f}" if sim is not None else "-"
                live_str = f"{liveness:.2f}" if liveness is not None else "-"
                
                print(f"{ts_str:<20} | {emp_id:<18} | {name:<20} | {action_colored} | {sim_str:<6} | {live_str:<10}")
            
            print(f"{'-'*95}\n")
            print(f"Total: {len(rows)} record(s) shown.")
            
        except Exception as e:
            print(f"Lỗi truy vấn: {e}")
        finally:
            pool.close()


if __name__ == "__main__":
    main()
