"""
Module nạp mô hình ONNX với khả năng tự động phát hiện Execution Provider (CUDA GPU / CPU).

Chức năng chính:
    - Kiểm tra đường dẫn file trọng số ONNX (.onnx).
    - Cấu hình SessionOptions (bật tối ưu hóa đồ thị mô hình ONNX).
    - Tự động ưu tiên CUDA (NVIDIA GPU) nếu hệ thống hỗ trợ, nếu không sẽ fallback về CPU.
    - Khởi tạo ONNX Runtime InferenceSession và lấy tên layer đầu vào (input node name).
"""

import onnxruntime as ort
from typing import Tuple, Optional
from pathlib import Path


def load_model(model_path: str) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
    """
    Nạp mô hình ONNX từ đường dẫn truyền vào.

    Tham số:
        model_path (str): Đường dẫn tới file trọng số ONNX (ví dụ: best_model_quantized.onnx).

    Trả về:
        Tuple[Optional[ort.InferenceSession], Optional[str]]:
            - ort_session: Phiên làm việc suy luận ONNX Runtime (hoặc None nếu lỗi).
            - input_name: Tên của node đầu vào trong mô hình ONNX (hoặc None nếu lỗi).
    """
    # 1. Kiểm tra sự tồn tại của file trọng số
    if not Path(model_path).exists():
        return None, None

    try:
        # 2. Cấu hình các tùy chọn tối ưu hóa cho ONNX Runtime Session
        sess_options = ort.SessionOptions()
        # Bật tất cả các mức độ tối ưu hóa đồ thị suy luận (Graph Optimization) (gộp các phép toán, loại bỏ node thừa, constant folding...)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Thiết lập chế độ chạy tuần tự (Sequential) để ổn định, ít overhead
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 3. Tự động kiểm tra và ưu tiên Execution Provider (GPU CUDA -> CPU)
        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available_providers]

        # Nếu không khớp provider ưu tiên nào, dùng danh sách provider có sẵn của hệ thống
        if not providers:
            providers = available_providers

        # 4. Khởi tạo ONNX InferenceSession, và lấy tên input
        ort_session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
        # Lấy tên của tensor đầu vào đầu tiên trong mô hình (thường là 'input' hoặc 'data').
        input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name
    except Exception:
        # Trả về (None, None) nếu xảy ra lỗi trong quá trình load model
        return None, None
