"""Dò các camera OpenCV có thể mở và đọc được frame."""

import argparse
import time

import cv2


def find_cameras(max_index):
    cameras = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue

            frame = None
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.15)
            if frame is None or not ok:
                continue

            try:
                backend = cap.getBackendName()
            except cv2.error:
                backend = "unknown"
            height, width = frame.shape[:2]
            cameras.append((index, backend, width, height))
        finally:
            cap.release()
    return cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-index", type=int, default=10,
        help="Chỉ số camera cao nhất cần thử (mặc định: 10, bao gồm cả số này)",
    )
    args = parser.parse_args()
    if args.max_index < 0:
        parser.error("--max-index phải từ 0 trở lên")

    print(f"Đang dò camera từ index 0 đến {args.max_index}...")
    cameras = find_cameras(args.max_index)
    if not cameras:
        print("Không tìm thấy camera đọc được frame.")
        print("Kiểm tra kết nối, quyền camera và thử tăng --max-index.")
        return 1

    print("Camera tìm thấy:")
    for index, backend, width, height in cameras:
        print(f"  index {index}: {backend}, {width}x{height}")
    print("Dùng camera mong muốn bằng: python3 app.py --camera INDEX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
