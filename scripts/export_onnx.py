"""
Export MobileNetV3_large Anti-Spoofing từ PyTorch (.pth.tar) sang ONNX (.onnx).

Dùng forward_to_onnx() có sẵn trong model → output là softmax [real_prob, spoof_prob].

Cách chạy:
    cd face_auth
    python scripts/export_onnx.py \
        --weights anti_spoofing/weights/MN3_antispoof.pth.tar \
        --output  anti_spoofing/weights/MN3_antispoof.onnx
"""

import argparse
import os
import sys
from collections import OrderedDict

import torch
import numpy as np

# ── Thêm face_auth vào path ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FACE_AUTH = os.path.dirname(_HERE)
if _FACE_AUTH not in sys.path:
    sys.path.insert(0, _FACE_AUTH)

from anti_spoofing.models import mobilenetv3_large
from anti_spoofing import config as cfg


def load_pytorch_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load checkpoint PyTorch và trả về model ở chế độ eval."""
    model = mobilenetv3_large(**cfg.MODEL_PARAMS)
    model.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Xử lý DataParallel prefix "module."
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    img_size: int = cfg.IMG_SIZE,
):
    """Export model sang ONNX với dynamic batch size."""

    # Dummy input: batch=1, 3 channels, img_size x img_size
    dummy_input = torch.randn(1, 3, img_size, img_size)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},   # hỗ trợ batch inference
            "output": {0: "batch_size"},
        },
    )

    # Verify file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Đã lưu ONNX model: {output_path} ({size_mb:.1f} MB)")


def verify_onnx(onnx_path: str, img_size: int = cfg.IMG_SIZE):
    """Kiểm tra nhanh ONNX model chạy đúng."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Verify] onnxruntime chưa cài, bỏ qua verify. Chạy: pip install onnxruntime")
        return

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    result = session.run(None, {"input": dummy})
    probs = result[0]
    print(f"[Verify] ONNX output shape: {probs.shape}")
    print(f"[Verify] Sample output (softmax): real={probs[0][0]:.4f}, spoof={probs[0][1]:.4f}")
    print(f"[Verify] Sum = {probs[0].sum():.4f} (should be ~1.0)")

    # Batch test
    dummy_batch = np.random.randn(4, 3, img_size, img_size).astype(np.float32)
    result_batch = session.run(None, {"input": dummy_batch})
    print(f"[Verify] Batch=4 output shape: {result_batch[0].shape} ✓")


def main():
    parser = argparse.ArgumentParser(description="Export MN3 Anti-Spoofing to ONNX")
    parser.add_argument(
        "--weights", required=True,
        help="Path to MN3_antispoof.pth.tar",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .onnx path (mặc định: cùng thư mục, đổi đuôi .onnx)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Bỏ qua verify sau export")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.weights.replace(".pth.tar", ".onnx")

    print(f"[Export] Loading PyTorch model từ: {args.weights}")
    model = load_pytorch_model(args.weights)

    # Quan trọng: dùng forward_to_onnx thay vì forward
    # forward_to_onnx đã bao gồm: features → conv_last → avgpool → flatten → spoofer → softmax
    # Output trực tiếp là [real_prob, spoof_prob]
    original_forward = model.forward
    model.forward = model.forward_to_onnx

    print(f"[Export] Exporting sang ONNX (input: 1x3x{cfg.IMG_SIZE}x{cfg.IMG_SIZE})...")
    export_to_onnx(model, args.output)

    # Restore forward gốc
    model.forward = original_forward

    if not args.no_verify:
        print(f"\n[Verify] Kiểm tra ONNX model...")
        verify_onnx(args.output)

    print("\n✅ Export thành công!")


if __name__ == "__main__":
    main()
