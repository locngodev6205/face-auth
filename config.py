import os
from dotenv import load_dotenv

load_dotenv()

# Database Config
DATABASE_URL = os.getenv("DATABASE_URL")

POSTGRES_DB = os.getenv("POSTGRES_DB", "face_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Connection info string fallback
if DATABASE_URL:
    DB_CONN_INFO = DATABASE_URL
else:
    DB_CONN_INFO = f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD}"

# Connection pool (tránh mở connection mới mỗi lần search/upsert)
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "4"))

# Model Config
# Detection (SCRFD) và Recognition (ArcFace) đều lấy từ cùng một insightface
# model pack — không cần quản lý riêng file weight YOLOv8-face .pt/.onnx nữa.
MODEL_PACK_NAME = os.getenv("MODEL_PACK_NAME", "buffalo_s")

# ctx_id: -1 = CPU, >=0 = GPU device index (chỉ có ý nghĩa khi dùng CUDAExecutionProvider)
MODEL_CTX_ID = int(os.getenv("MODEL_CTX_ID", "-1"))

DETECTOR_DET_SIZE = (640, 640)
DETECTOR_CONF_THRESH = float(os.getenv("DETECTOR_CONF_THRESH", "0.5"))

# Tracking Config
# Khuôn mặt được track bằng IOU giữa các frame; danh tính chỉ được re-verify
# (chạy lại align+embed+DB search) sau mỗi RECOGNIZE_INTERVAL_SECONDS giây
# thay vì đếm số frame. Đặt theo thời gian (1.0s) giúp hệ thống hoạt động
# hoàn toàn nhất quán bất kể chạy trên thiết bị có FPS cao (Mac 30-60 FPS)
# hay thiết bị Edge có FPS thấp (Raspberry Pi 5-15 FPS).
RECOGNIZE_INTERVAL_SECONDS = float(os.getenv("RECOGNIZE_INTERVAL_SECONDS", "1.0"))
RECOGNIZE_INTERVAL = RECOGNIZE_INTERVAL_SECONDS  # Alias tương thích ngược

# Ngưỡng IOU để coi 2 bbox ở 2 frame liên tiếp là cùng một track.
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

# Số frame liên tiếp không match được detection nào trước khi xóa track
# (coi như khuôn mặt đã rời khỏi khung hình).
TRACK_MAX_MISSING_FRAMES = int(os.getenv("TRACK_MAX_MISSING_FRAMES", "10"))

# Enrollment Config
# Kết quả thực nghiệm trên LFW Benchmark cho thấy gallery size = 3 là điểm cân bằng
# tối ưu giữa độ chính xác nhận diện và thời gian/công sức đăng ký của người dùng.
RECOMMENDED_GALLERY_SIZE = int(os.getenv("RECOMMENDED_GALLERY_SIZE", "3"))

# App Config
# MATCH_THRESHOLD đã được calibrate từ LFW Identity Recognition Benchmark:
# - Ngưỡng tối ưu thực nghiệm trên tập calibration LFW là ~0.31.
# - Nâng lên 0.35 cho môi trường production để siết chặt bảo mật (giảm FAR/chặn người lạ tốt hơn).
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))