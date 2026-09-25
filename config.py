import math
import os
from dotenv import load_dotenv

load_dotenv()


def _clean_env_val(val: str | None) -> str | None:
    if val is None:
        return None
    # python-dotenv đã xử lý comment và quoted value khi load_dotenv().
    # ở đây chỉ cần strip khoảng trắng thừa ở đầu/cuối.
    return val.strip()


def _get_env_str(key: str, default: str | None = None) -> str | None:
    val = _clean_env_val(os.getenv(key))
    return val if val is not None and val != "" else default


def _get_env_int(key: str, default: int) -> int:
    val = _clean_env_val(os.getenv(key))
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = _clean_env_val(os.getenv(key))
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    val = _clean_env_val(os.getenv(key))
    if val is None or val == "":
        return default
    return val.lower() in ("true", "1", "yes", "y", "on")


# Database Config
DATABASE_URL = _get_env_str("DATABASE_URL", None)

POSTGRES_DB = _get_env_str("POSTGRES_DB", "face_db")
POSTGRES_USER = _get_env_str("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = _get_env_str("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = _get_env_str("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _get_env_str("POSTGRES_PORT", "5432")

# Connection info string fallback
if DATABASE_URL:
    DB_CONN_INFO = DATABASE_URL
else:
    DB_CONN_INFO = f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD}"

# Connection pool (tránh mở connection mới mỗi lần search/upsert)
DB_POOL_MIN_SIZE = _get_env_int("DB_POOL_MIN_SIZE", 1)
DB_POOL_MAX_SIZE = _get_env_int("DB_POOL_MAX_SIZE", 4)

# Model Config — Recognition (ArcFace)
# ArcFace lấy từ insightface model pack buffalo_s.
MODEL_PACK_NAME = _get_env_str("MODEL_PACK_NAME", "buffalo_s")
EMBEDDING_MODEL_VERSION = _get_env_str("EMBEDDING_MODEL_VERSION", MODEL_PACK_NAME)

# ctx_id: -1 = CPU, >=0 = GPU device index (chỉ có ý nghĩa khi dùng CUDAExecutionProvider)
MODEL_CTX_ID = _get_env_int("MODEL_CTX_ID", -1)

# Detector Config — YunNet / SCRFD
# DETECTOR_MODEL_PATH: None = dùng đường dẫn mặc định trong detection/models/
DETECTOR_MODEL_PATH = _get_env_str("DETECTOR_MODEL_PATH", None)
DETECTOR_CONF_THRESH = _get_env_float("DETECTOR_CONF_THRESH", 0.5)
DETECTOR_NMS_THRESH = _get_env_float("DETECTOR_NMS_THRESH", 0.3) # trùng lặp 30% thì bỏ trùng lặp, gửi lại bbox có acc lớn nhất
DETECTOR_TOP_K = _get_env_int("DETECTOR_TOP_K", 5000)

# Kích thước khuôn mặt tối thiểu (pixel chiều rộng & chiều cao), áp dụng cho
# cả YunNet và SCRFD trước khi tạo detection/track.
# - Tăng giá trị (ví dụ: 60 - 100) để tập trung nhận diện người đứng gần camera/kiosk,
#   tự động bỏ qua người đi lại ở hậu cảnh xa.
# - Giảm giá trị (ví dụ: 20 - 30) nếu muốn nhận diện người từ khoảng cách xa.
DETECTOR_MIN_FACE_SIZE = _get_env_int("DETECTOR_MIN_FACE_SIZE", 60)

# Khoảng cách tối thiểu (pixel) từ khuôn mặt tới 4 mép viền ảnh.
# Bỏ qua các khuôn mặt quá sát mép (< margin) để tránh cắt lẹm mặt gây sai lệch nhận diện & anti-spoofing.
DETECTOR_MARGIN = _get_env_int("DETECTOR_MARGIN", 5)

# Kích thước khung hình khởi tạo cho detector
DETECTOR_INPUT_WIDTH  = _get_env_int("DETECTOR_INPUT_WIDTH", 320)
DETECTOR_INPUT_HEIGHT = _get_env_int("DETECTOR_INPUT_HEIGHT", 320)
DETECTOR_INPUT_SIZE   = (DETECTOR_INPUT_WIDTH, DETECTOR_INPUT_HEIGHT)

# Alias tương thích ngược (từ bản SCRFD cũ, giữ để không vỡ script cũ)
DETECTOR_DET_SIZE = (640, 640)

# Camera Config (dùng cho app.py và các script demo webcam real-time)
CAMERA_INDEX  = _get_env_int("CAMERA_INDEX", 0)
CAMERA_WIDTH  = _get_env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _get_env_int("CAMERA_HEIGHT", 480)
CAMERA_FPS    = _get_env_int("CAMERA_FPS", 30)
# Giảm số frame cũ nằm trong hàng đợi backend camera. Một số backend có thể bỏ qua.
CAMERA_BUFFER_SIZE = _get_env_int("CAMERA_BUFFER_SIZE", 1)

# Detector cho app.py (real-time inference).
# - "yunnet": cv2.FaceDetectorYN, ~122 KB, nhanh trên edge, nhưng cần resize frame nhỏ trước khi detect.
#             App.py sẽ set CAMERA_WIDTH × CAMERA_HEIGHT qua cap.set() để giới hạn frame size.
# - "scrfd" : InsightFace SCRFD (buffalo_s), resize nội bộ qua det_size nhưng vẫn nhận
#             frame camera theo kích thước cấu hình; app in kích thước thực tế vì backend
#             Windows/Linux/macOS có thể không hỗ trợ mọi lệnh cap.set().
APP_DETECTOR = (_get_env_str("APP_DETECTOR", "scrfd") or "scrfd").lower()

# Anti-Spoofing / PAD (Presentation Attack Detection) Config
# PAD_ENABLED: bật/tắt module kiểm tra liveness. Cũng có thể bật/tắt qua CLI --pad / --no-pad.
# Set "true"/"1" để bật mặc định, "false"/"0" để tắt mặc định.
PAD_ENABLED = _get_env_bool("PAD_ENABLED", True)

# PAD_MODEL_FILENAME: checkpoint E1 là runtime contract hiện tại.
PAD_MODEL_FILENAME  = _get_env_str(
    "PAD_MODEL_FILENAME",
    "smartface_pad_artifacts/E1_v5_3_mnv3_small/deployment/mnv3s_e1_preliminary_v5_3_edge_best.onnx",
)
# PAD_THRESHOLD luôn là xác suất P(REAL), không phải logit difference.
# Với E1, operating point đã calibrate: p=0.3356796703127529,
# tương đương d=-0.682607114315033.
PAD_THRESHOLD       = _get_env_float("PAD_THRESHOLD", 0.3356796703127529)
_default_pad_threshold_logit = (
    math.log(PAD_THRESHOLD / (1.0 - PAD_THRESHOLD))
    if 0.0 < PAD_THRESHOLD < 1.0
    else 0.0
)
PAD_THRESHOLD_LOGIT = _get_env_float(
    "PAD_THRESHOLD_LOGIT",
    _default_pad_threshold_logit,
)

if not (0.0 < PAD_THRESHOLD < 1.0):
    raise ValueError(
        f"Cấu hình PAD_THRESHOLD={PAD_THRESHOLD} không hợp lệ. Yêu cầu 0 < threshold < 1."
    )
_expected_pad_threshold_logit = math.log(PAD_THRESHOLD / (1.0 - PAD_THRESHOLD))
if abs(PAD_THRESHOLD_LOGIT - _expected_pad_threshold_logit) > 1e-6:
    raise ValueError(
        "PAD_THRESHOLD và PAD_THRESHOLD_LOGIT không nhất quán: "
        f"expected {_expected_pad_threshold_logit:.12f}, "
        f"received {PAD_THRESHOLD_LOGIT:.12f}."
    )

_raw_color_order = _get_env_str("PAD_COLOR_ORDER", None)
if _raw_color_order is not None and _raw_color_order != "":
    PAD_COLOR_ORDER = _raw_color_order.strip().upper()
    if PAD_COLOR_ORDER not in ("RGB", "BGR"):
        raise ValueError(
            f"Cấu hình PAD_COLOR_ORDER không hợp lệ: '{_raw_color_order}'. "
            f"Chỉ chấp nhận 'RGB', 'BGR' hoặc để trống (auto-detect)."
        )
else:
    PAD_COLOR_ORDER = None

# Adaptive Gamma Correction cho face crop của PAD.
# Bật để robust hơn với điều kiện ánh sáng khác nhau (tối / sáng quá).
# Gamma được tính động theo perceived luma (kênh V của HSV), target về ~110/255.
# Set "true"/"1" để bật, "false"/"0" để tắt.
PAD_GAMMA_ENABLED = _get_env_bool("PAD_GAMMA_ENABLED", False)

# Target luma (kênh V, [0-255]) mà adaptive gamma hướng đến.
# 110 ≈ 43% — đủ sáng để model học texture, không bị over-expose.
PAD_GAMMA_TARGET  = _get_env_float("PAD_GAMMA_TARGET", 110.0)

# Hệ số mở rộng bbox khi crop khuôn mặt đưa vào PAD.
PAD_BBOX_EXPANSION_FACTOR = _get_env_float("PAD_BBOX_EXPANSION_FACTOR", 1.55)

# Đường dẫn thư mục chứa tất cả model anti-spoofing
_ANTISPOOF_MODELS_DIR = os.path.join(os.path.dirname(__file__), "antispoof", "models")
PAD_MODEL_PATH = os.path.join(_ANTISPOOF_MODELS_DIR, PAD_MODEL_FILENAME)

# Alias tương thích ngược (từ biến LIVENESS_* cũ)
LIVENESS_MODEL_PATH = PAD_MODEL_PATH
LIVENESS_THRESHOLD  = PAD_THRESHOLD

# FPS overlay trên màn hình real-time
SHOW_FPS = _get_env_bool("SHOW_FPS", True)
# Số frame dùng để tính FPS trung bình (rolling window)
FPS_AVG_WINDOW = _get_env_int("FPS_AVG_WINDOW", 30)

# Tracking Config
# Khuôn mặt được track bằng IOU giữa các frame; danh tính chỉ được re-verify
# (chạy lại align+embed+DB search) sau mỗi RECOGNIZE_INTERVAL_SECONDS giây
# thay vì đếm số frame. Đặt theo thời gian (0.5s) giúp hệ thống hoạt động
# hoàn toàn nhất quán bất kể chạy trên thiết bị có FPS cao (Mac 30-60 FPS)
# hay thiết bị Edge có FPS thấp (Raspberry Pi 5-15 FPS).
RECOGNIZE_INTERVAL_SECONDS = _get_env_float("RECOGNIZE_INTERVAL_SECONDS", 0.5)
RECOGNIZE_INTERVAL = RECOGNIZE_INTERVAL_SECONDS  # Alias tương thích ngược

# PAD Temporal Smoothing — chống nhấp nháy SPOOF/REAL khi mặt di chuyển.
#
# PAD_SMOOTH_WINDOW: số lần inference PAD dùng để biểu quyết REAL/SPOOF (rolling window).
#   Mỗi lần inference = 1 "phiếu" bầu. Window tích lũy PAD_SMOOTH_WINDOW phiếu gần nhất.
#   Tăng → ổn định hơn, chậm phản ứng hơn.  Giảm → nhanh hơn nhưng dễ nhấp nháy.
PAD_SMOOTH_WINDOW = _get_env_int("PAD_SMOOTH_WINDOW", 5)

# PAD_MIN_VOTES: số phiếu tối thiểu cần tích lũy trước khi track thoát khỏi trạng thái PAD_PENDING
PAD_MIN_VOTES = _get_env_int("PAD_MIN_VOTES", PAD_SMOOTH_WINDOW)

# PAD_SPOOF_MIN_RATIO: tỉ lệ tối thiểu phiếu SPOOF trong window để bị kết luận SPOOF.
#   0.6 = cần ≥ 60% phiếu SPOOF → mới hiện đỏ. Mặt thật di chuyển thường chỉ
#   bị nhiễu ~30-40% phiếu → không đủ ngưỡng → giữ xanh.
PAD_SPOOF_MIN_RATIO = _get_env_float("PAD_SPOOF_MIN_RATIO", 0.6)

# PAD_STALE_TIMEOUT_SECONDS: nếu track không nhận PAD inference nào trong khoảng thời gian này,
# rolling window sẽ bị coi là cũ và bị reset khi inference tiếp theo được gọi.
# Kết quả: track quay về trạng thái PAD_PENDING (không thể điểm danh) cho đến khi đủ vote mới.
# Mặc định 3.0s — đủ dài để che hết khoảng cách giữa các lần detect, đủ ngắn để phát hiện
# khi người dùng rời khỏi frame rồi quay lại (gương/ảnh mới).
PAD_STALE_TIMEOUT_SECONDS = _get_env_float("PAD_STALE_TIMEOUT_SECONDS", 3.0)
PAD_DIAGNOSTIC_LOG = _get_env_bool("PAD_DIAGNOSTIC_LOG", False)

# PAD_INTERVAL_SECONDS: khoảng cách giữa 2 lần chạy PAD inference cho cùng 1 track.
#
# KHÔNG ĐẶT THỦ CÔNG — được tự động tính từ:
#   PAD_INTERVAL = RECOGNIZE_INTERVAL / PAD_SMOOTH_WINDOW
#
# Ý nghĩa: PAD chạy đúng PAD_SMOOTH_WINDOW lần trong mỗi 1 chu kỳ recognition.
# → Khi recognition fire (mỗi 0.5s), rolling window đã đủ SMOOTH_WINDOW phiếu → verdict ổn định.
# → PAD và Recognition luôn đồng bộ, không cần chỉnh thêm.
PAD_INTERVAL_SECONDS = _get_env_float(
    "PAD_INTERVAL_SECONDS",
    round(RECOGNIZE_INTERVAL_SECONDS / max(PAD_SMOOTH_WINDOW, 1), 4),
)

# Face detector cadence. 0 means detect every frame (debug/rollback mode).
# Mặc định đồng bộ với PAD cadence để không chạy detector dày hơn cần thiết.
DETECTION_INTERVAL_SECONDS = _get_env_float(
    "DETECTION_INTERVAL_SECONDS", PAD_INTERVAL_SECONDS
)

# Ngưỡng IOU để coi 2 bbox ở 2 frame liên tiếp là cùng một track.
TRACK_IOU_THRESHOLD = _get_env_float("TRACK_IOU_THRESHOLD", 0.3)

# Số frame liên tiếp không match được detection nào trước khi xóa track
# (coi như khuôn mặt đã rời khỏi khung hình).
TRACK_MAX_MISSING_FRAMES = _get_env_int("TRACK_MAX_MISSING_FRAMES", 10)

# Enrollment Config
# Kết quả thực nghiệm trên LFW Benchmark cho thấy gallery size = 3 là điểm cân bằng
# tối ưu giữa độ chính xác nhận diện và thời gian/công sức đăng ký của người dùng.
RECOMMENDED_GALLERY_SIZE = _get_env_int("RECOMMENDED_GALLERY_SIZE", 3)

# App Config
# MATCH_THRESHOLD đã được calibrate từ LFW Identity Recognition Benchmark:
# - Ngưỡng tối ưu thực nghiệm trên tập calibration LFW là ~0.31.
# - Nâng lên 0.35 cho môi trường production để siết chặt bảo mật (giảm FAR/chặn người lạ tốt hơn).
MATCH_THRESHOLD = _get_env_float("MATCH_THRESHOLD", 0.35)

# Attendance Tracking
ATTENDANCE_MODE = (_get_env_str("ATTENDANCE_MODE", "checkin") or "checkin").lower()
ATTENDANCE_GAP_MINUTES = _get_env_int("ATTENDANCE_GAP_MINUTES", 15)
ATTENDANCE_STABLE_COUNT = _get_env_int("ATTENDANCE_STABLE_COUNT", 3)


# Validation
def validate_config():
    """Kiểm tra tính hợp lệ của các tham số cấu hình hệ thống."""
    if PAD_BBOX_EXPANSION_FACTOR <= 0:
        raise ValueError(
            f"Cấu hình PAD_BBOX_EXPANSION_FACTOR={PAD_BBOX_EXPANSION_FACTOR} không hợp lệ. "
            "Yêu cầu > 0."
        )
    if DETECTOR_MIN_FACE_SIZE < 0:
        raise ValueError(
            f"Cấu hình DETECTOR_MIN_FACE_SIZE={DETECTOR_MIN_FACE_SIZE} không hợp lệ. "
            "Yêu cầu >= 0."
        )
    if PAD_SMOOTH_WINDOW < 1:
        raise ValueError(f"Cấu hình PAD_SMOOTH_WINDOW={PAD_SMOOTH_WINDOW} không hợp lệ. Yêu cầu >= 1.")
    if not (1 <= PAD_MIN_VOTES <= PAD_SMOOTH_WINDOW):
        raise ValueError(
            f"Cấu hình PAD_MIN_VOTES={PAD_MIN_VOTES} không hợp lệ. "
            f"Yêu cầu 1 <= PAD_MIN_VOTES <= PAD_SMOOTH_WINDOW ({PAD_SMOOTH_WINDOW})."
        )
    if not (0.0 <= PAD_SPOOF_MIN_RATIO <= 1.0):
        raise ValueError(
            f"Cấu hình PAD_SPOOF_MIN_RATIO={PAD_SPOOF_MIN_RATIO} không hợp lệ. "
            f"Yêu cầu 0.0 <= PAD_SPOOF_MIN_RATIO <= 1.0."
        )
    if PAD_INTERVAL_SECONDS <= 0:
        raise ValueError(f"Cấu hình PAD_INTERVAL_SECONDS={PAD_INTERVAL_SECONDS} không hợp lệ. Yêu cầu > 0.")
    if PAD_STALE_TIMEOUT_SECONDS <= 0:
        raise ValueError(
            f"Cấu hình PAD_STALE_TIMEOUT_SECONDS={PAD_STALE_TIMEOUT_SECONDS} không hợp lệ. Yêu cầu > 0."
        )
    if DETECTION_INTERVAL_SECONDS < 0:
        raise ValueError(
            f"Cấu hình DETECTION_INTERVAL_SECONDS={DETECTION_INTERVAL_SECONDS} không hợp lệ. "
            "Yêu cầu >= 0 (0 = detect mỗi frame)."
        )
    if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
        raise ValueError(
            f"Cấu hình camera {CAMERA_WIDTH}x{CAMERA_HEIGHT} không hợp lệ. "
            "Yêu cầu width/height > 0."
        )
    if CAMERA_FPS <= 0:
        raise ValueError(f"Cấu hình CAMERA_FPS={CAMERA_FPS} không hợp lệ. Yêu cầu > 0.")
    if CAMERA_BUFFER_SIZE < 0:
        raise ValueError(
            f"Cấu hình CAMERA_BUFFER_SIZE={CAMERA_BUFFER_SIZE} không hợp lệ. "
            "Yêu cầu >= 0 (0 = để backend tự chọn)."
        )


validate_config()
