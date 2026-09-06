"""
Module tiện ích truy xuất thông tin phần cứng hệ thống (CPU, GPU, ONNX Provider).

Chức năng chính:
    1. get_cpu_info(): Lấy thông tin Tên vi xử lý CPU, số nhân/luồng, xung nhịp hiện tại.
    2. get_gpu_info(): Phát hiện card đồ họa rời NVIDIA GPU qua GPUtil hoặc nvidia-smi.
    3. get_execution_provider_name(): Xác định mô hình ONNX đang chạy trên phần cứng nào (CUDA hay CPU).
"""

import platform
import onnxruntime as ort
from typing import Optional

# Thử import các thư viện phần cứng tùy chọn (An toàn nếu chưa cài đặt)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


def get_cpu_info() -> str:
    """
    Lấy chuỗi thông tin chi tiết về vi xử lý CPU hiện tại của hệ thống.

    Trả về:
        str: Chuỗi định dạng tên CPU | Số nhân/luồng | Xung nhịp (ví dụ: "Intel Core i7 | 4C/8T | 2.50 GHz").
    """
    cpu_name = None
    cpu_freq_mhz = None
    cpu_cores = None
    cpu_threads = None

    # Lấy tên nhãn hiệu CPU bằng thư viện py-cpuinfo
    if HAS_CPUINFO:
        try:
            info = cpuinfo.get_cpu_info()
            cpu_name = (
                info.get("brand_raw") or info.get("brand") or info.get("model name")
            )
            if cpu_name:
                cpu_name = cpu_name.strip()
        except Exception:
            pass

    # Lấy số nhân vật lý, số luồng logic và xung nhịp bằng thư viện psutil
    if HAS_PSUTIL:
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            if cpu_freq and cpu_freq.current:
                cpu_freq_mhz = cpu_freq.current
        except Exception:
            pass

    # Fallback về module platform mặc định nếu không đọc được tên CPU
    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"

    parts = []
    if cpu_name:
        parts.append(cpu_name)

    # Định dạng hiển thị số nhân và số luồng (ví dụ 4C/8T)
    if cpu_cores and cpu_threads:
        if cpu_cores == cpu_threads:
            parts.append(f"{cpu_cores} cores")
        else:
            parts.append(f"{cpu_cores}C/{cpu_threads}T")

    # Định dạng hiển thị xung nhịp GHz hoặc MHz
    if cpu_freq_mhz:
        if cpu_freq_mhz >= 1000:
            parts.append(f"{cpu_freq_mhz/1000:.2f} GHz")
        else:
            parts.append(f"{cpu_freq_mhz:.0f} MHz")

    return " | ".join(parts) if parts else "Unknown CPU"


def get_gpu_info() -> Optional[str]:
    """
    Lấy tên card đồ họa rời NVIDIA GPU đang có trên hệ thống.

    Trả về:
        Optional[str]: Tên GPU (ví dụ "NVIDIA GeForce RTX 3060") hoặc None nếu không phát hiện được.
    """
    # 1. Thử đọc thông tin GPU thông qua thư viện GPUtil
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > 0:
                gpu = gpus[0]
                return gpu.name.strip()
        except Exception:
            pass

    # 2. Fallback thử gọi lệnh hệ thống nvidia-smi
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return None


def get_execution_provider_name(session: ort.InferenceSession) -> str:
    """
    Xác định mô hình ONNX Runtime InferenceSession đang thực thi trên môi trường phần cứng nào.

    Tham số:
        session (ort.InferenceSession): Session ONNX Runtime đang hoạt động.

    Trả về:
        str: "CUDA" (nếu đang dùng GPU) hoặc "CPU" (nếu dùng CPU).
    """
    try:
        providers = session.get_providers()
        if "CUDAExecutionProvider" in providers:
            return "CUDA"
        elif "CPUExecutionProvider" in providers:
            return "CPU"
        elif providers:
            return providers[0].replace("ExecutionProvider", "")
        return "Unknown"
    except Exception:
        return "Unknown"
