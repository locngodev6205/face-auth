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

# Model Config — Recognition (ArcFace)
# ArcFace lấy từ insightface model pack buffalo_s.
MODEL_PACK_NAME = os.getenv("MODEL_PACK_NAME", "buffalo_s")

# ctx_id: -1 = CPU, >=0 = GPU device index (chỉ có ý nghĩa khi dùng CUDAExecutionProvider)
MODEL_CTX_ID = int(os.getenv("MODEL_CTX_ID", "-1"))

# Detector Config — YunNet (cv2.FaceDetectorYN, ONNX ~122 KB)
# DETECTOR_MODEL_PATH: None = dùng đường dẫn mặc định trong detection/models/
DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", None)
DETECTOR_CONF_THRESH = float(os.getenv("DETECTOR_CONF_THRESH", "0.5"))
DETECTOR_NMS_THRESH  = float(os.getenv("DETECTOR_NMS_THRESH",  "0.3"))
DETECTOR_TOP_K       = int(os.getenv("DETECTOR_TOP_K", "5000"))

# Kích thước khuôn mặt tối thiểu (pixel chiều rộng & chiều cao).
# - Tăng giá trị (ví dụ: 60 - 100) để tập trung nhận diện người đứng gần camera/kiosk,
#   tự động bỏ qua người đi lại ở hậu cảnh xa.
# - Giảm giá trị (ví dụ: 20 - 30) nếu muốn nhận diện người từ khoảng cách xa.
DETECTOR_MIN_FACE_SIZE = int(os.getenv("DETECTOR_MIN_FACE_SIZE", "60"))

# Khoảng cách tối thiểu (pixel) từ khuôn mặt tới 4 mép viền ảnh.
# Bỏ qua các khuôn mặt quá sát mép (< margin) để tránh cắt lẹm mặt gây sai lệch nhận diện & anti-spoofing.
DETECTOR_MARGIN = int(os.getenv("DETECTOR_MARGIN", "5"))

# Kích thước khung hình khởi tạo cho detector
DETECTOR_INPUT_WIDTH  = int(os.getenv("DETECTOR_INPUT_WIDTH", "320"))
DETECTOR_INPUT_HEIGHT = int(os.getenv("DETECTOR_INPUT_HEIGHT", "320"))
DETECTOR_INPUT_SIZE   = (DETECTOR_INPUT_WIDTH, DETECTOR_INPUT_HEIGHT)

# Alias tương thích ngược (từ bản SCRFD cũ, giữ để không vỡ script cũ)
DETECTOR_DET_SIZE = (640, 640)

# Camera Config (dùng cho app.py và các script demo webcam real-time)
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS    = int(os.getenv("CAMERA_FPS", "30"))

# Detector cho app.py (real-time inference).
# - "yunnet": cv2.FaceDetectorYN, ~122 KB, nhanh trên edge, nhưng cần resize frame nhỏ trước khi detect.
#             App.py sẽ set CAMERA_WIDTH × CAMERA_HEIGHT qua cap.set() để giới hạn frame size.
# - "scrfd" : InsightFace SCRFD (buffalo_s), xử lý ảnh bất kỳ kích thước via det_size nội bộ,
#             không cần giới hạn frame size — camera sẽ dùng full resolution nếu muốn.
APP_DETECTOR = os.getenv("APP_DETECTOR", "yunnet").lower().strip()

# Anti-Spoofing / PAD (Presentation Attack Detection) Config
# PAD_ENABLED: bật/tắt module kiểm tra liveness. Cũng có thể bật/tắt qua CLI --pad / --no-pad.
# Set "true"/"1" để bật mặc định, "false"/"0" để tắt mặc định.
PAD_ENABLED = os.getenv("PAD_ENABLED", "true").lower() in ("true", "1")

# PAD_MODEL_FILENAME: chỉ cần đặt tên file model (không cần full path).
# Các file model phải nằm trong thư mục antispoof/models/.
# File có sẵn: best_model_quantized.onnx (128px), mnv4_best_224.onnx (224px), mnv3_large_3class_best.onnx (224px)
PAD_MODEL_FILENAME  = os.getenv("PAD_MODEL_FILENAME",  "mnv3_large_3class_best.onnx")
PAD_THRESHOLD       = float(os.getenv("PAD_THRESHOLD",  "0.5"))

# Adaptive Gamma Correction cho face crop của PAD.
# Bật để robust hơn với điều kiện ánh sáng khác nhau (tối / sáng quá).
# Gamma được tính động theo perceived luma (kênh V của HSV), target về ~110/255.
# Set "true"/"1" để bật, "false"/"0" để tắt.
PAD_GAMMA_ENABLED = os.getenv("PAD_GAMMA_ENABLED", "true").lower() in ("true", "1")

# Target luma (kênh V, [0-255]) mà adaptive gamma hướng đến.
# 110 ≈ 43% — đủ sáng để model học texture, không bị over-expose.
PAD_GAMMA_TARGET  = float(os.getenv("PAD_GAMMA_TARGET", "110.0"))

# Đường dẫn thư mục chứa tất cả model anti-spoofing
_ANTISPOOF_MODELS_DIR = os.path.join(os.path.dirname(__file__), "antispoof", "models")
PAD_MODEL_PATH = os.path.join(_ANTISPOOF_MODELS_DIR, PAD_MODEL_FILENAME)

# Alias tương thích ngược (từ biến LIVENESS_* cũ)
LIVENESS_MODEL_PATH = PAD_MODEL_PATH
LIVENESS_THRESHOLD  = PAD_THRESHOLD

# FPS overlay trên màn hình real-time
SHOW_FPS = os.getenv("SHOW_FPS", "true").lower() in ("true", "1")
# Số frame dùng để tính FPS trung bình (rolling window)
FPS_AVG_WINDOW = int(os.getenv("FPS_AVG_WINDOW", "30"))

# Tracking Config
# Khuôn mặt được track bằng IOU giữa các frame; danh tính chỉ được re-verify
# (chạy lại align+embed+DB search) sau mỗi RECOGNIZE_INTERVAL_SECONDS giây
# thay vì đếm số frame. Đặt theo thời gian (1.0s) giúp hệ thống hoạt động
# hoàn toàn nhất quán bất kể chạy trên thiết bị có FPS cao (Mac 30-60 FPS)
# hay thiết bị Edge có FPS thấp (Raspberry Pi 5-15 FPS).
RECOGNIZE_INTERVAL_SECONDS = float(os.getenv("RECOGNIZE_INTERVAL_SECONDS", "1.0"))
RECOGNIZE_INTERVAL = RECOGNIZE_INTERVAL_SECONDS  # Alias tương thích ngược

# PAD Temporal Smoothing — chống nhấp nháy SPOOF/REAL khi mặt di chuyển.
#
# PAD_SMOOTH_WINDOW: số lần inference PAD dùng để biểu quyết REAL/SPOOF (rolling window).
#   Mỗi lần inference = 1 "phiếu" bầu. Window tích lũy PAD_SMOOTH_WINDOW phiếu gần nhất.
#   Tăng → ổn định hơn, chậm phản ứng hơn.  Giảm → nhanh hơn nhưng dễ nhấp nháy.
PAD_SMOOTH_WINDOW = int(os.getenv("PAD_SMOOTH_WINDOW", "5"))

# PAD_SPOOF_MIN_RATIO: tỉ lệ tối thiểu phiếu SPOOF trong window để bị kết luận SPOOF.
#   0.6 = cần ≥ 60% phiếu SPOOF → mới hiện đỏ. Mặt thật di chuyển thường chỉ
#   bị nhiễu ~30-40% phiếu → không đủ ngưỡng → giữ xanh.
PAD_SPOOF_MIN_RATIO = float(os.getenv("PAD_SPOOF_MIN_RATIO", "0.6"))

# PAD_INTERVAL_SECONDS: khoảng cách giữa 2 lần chạy PAD inference cho cùng 1 track.
#
# KHÔNG ĐẶT THỦ CÔNG — được tự động tính từ:
#   PAD_INTERVAL = RECOGNIZE_INTERVAL / PAD_SMOOTH_WINDOW
#
# Ý nghĩa: PAD chạy đúng PAD_SMOOTH_WINDOW lần trong mỗi 1 chu kỳ recognition.
# → Khi recognition fire (mỗi 1s), rolling window đã đủ SMOOTH_WINDOW phiếu → verdict ổn định.
# → PAD và Recognition luôn đồng bộ, không cần chỉnh thêm.
#
# Muốn giảm tải trên Pi: tăng RECOGNIZE_INTERVAL_SECONDS (ví dụ 2.0s) hoặc
# giảm PAD_SMOOTH_WINDOW (ví dụ 4) — PAD_INTERVAL sẽ tự điều chỉnh.
# Override thủ công: đặt PAD_INTERVAL_SECONDS trong .env (không khuyến nghị).
PAD_INTERVAL_SECONDS = float(
    os.getenv(
        "PAD_INTERVAL_SECONDS",
        str(round(RECOGNIZE_INTERVAL_SECONDS / max(PAD_SMOOTH_WINDOW, 1), 4)),
    )
)

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