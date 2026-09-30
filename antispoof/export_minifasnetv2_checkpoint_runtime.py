"""Export the local PAD-pretrained MiniFASNetV2 checkpoint for runtime smoke tests.

This preserves the checkpoint's existing 3-class classifier. The runtime
threshold is an uncalibrated 0.5 placeholder and must not be used to claim PAD
accuracy or as a production access-control model.
"""

import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "antispoof" / "models"
SOURCE_PATH = MODEL_DIR / "MiniFASNet.py"
CHECKPOINT_PATH = MODEL_DIR / "2.7_80x80_MiniFASNetV2.pth"
MODEL_PATH = MODEL_DIR / "minifasnetv2_available_checkpoint_smoke.onnx"
RUNTIME_CONFIG_PATH = MODEL_DIR / "minifasnetv2_available_checkpoint_smoke_runtime_config.json"
VALIDATION_PATH = MODEL_DIR / "minifasnetv2_available_checkpoint_smoke_onnx_validation.json"

INPUT_SIZE = 80
SEED = 100
OPSET = 17
RTOL = 1e-4
ATOL = 1e-5
EXPECTED_SOURCE_SHA256 = "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7"
EXPECTED_CHECKPOINT_SHA256 = "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if not SOURCE_PATH.is_file() or not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Required MiniFASNet source/checkpoint missing: {SOURCE_PATH}, {CHECKPOINT_PATH}")
    if sha256_file(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("MiniFASNet source SHA256 differs from the verified runtime export source")
    if sha256_file(CHECKPOINT_PATH) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MiniFASNet checkpoint SHA256 differs from the verified official checkpoint")

    sys.path.insert(0, str(MODEL_DIR))
    module = importlib.import_module("MiniFASNet")
    torch.manual_seed(SEED)
    model = module.MiniFASNetV2(
        embedding_size=128,
        conv6_kernel=(5, 5),
        drop_p=0.2,
        num_classes=3,
        img_channel=3,
    )

    raw_state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, OrderedDict) or not raw_state:
        raise RuntimeError("Expected the local MiniFASNet checkpoint to be a non-empty OrderedDict")
    if not all(key.startswith("module.") for key in raw_state):
        raise RuntimeError("Expected every checkpoint key to use the official 'module.' prefix")
    state = OrderedDict((key[7:], value) for key, value in raw_state.items())
    if tuple(state.get("conv_6_dw.conv.weight", torch.empty(0)).shape[-2:]) != (5, 5):
        raise RuntimeError("Checkpoint conv6 kernel is not 5x5")
    if tuple(state.get("prob.weight", torch.empty(0)).shape) != (3, 128):
        raise RuntimeError("Checkpoint classifier does not match the expected 3x128 MiniFASNetV2 head")
    model.load_state_dict(state, strict=True)
    model.eval()

    with tempfile.TemporaryDirectory(prefix="minifasnet_onnx_export_") as temp_dir:
        temporary_model = Path(temp_dir) / MODEL_PATH.name
        export_example = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        torch.onnx.export(
            model,
            export_example,
            str(temporary_model),
            input_names=["input"],
            output_names=["logits"],
            opset_version=OPSET,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )

        graph = onnx.load(str(temporary_model))
        onnx.checker.check_model(graph)
        if max(item.version for item in graph.opset_import if item.domain == "") != OPSET:
            raise RuntimeError("Exported ONNX opset is not 17")
        input_shape = graph.graph.input[0].type.tensor_type.shape.dim
        output_shape = graph.graph.output[0].type.tensor_type.shape.dim
        if graph.graph.input[0].name != "input" or graph.graph.output[0].name != "logits":
            raise RuntimeError("Exported ONNX input/output names do not match the runtime contract")
        if [dim.dim_value for dim in input_shape[1:]] != [3, INPUT_SIZE, INPUT_SIZE]:
            raise RuntimeError("Exported ONNX input shape is not [N,3,80,80]")
        if [dim.dim_value for dim in output_shape[1:]] != [3]:
            raise RuntimeError("Exported ONNX output shape is not [N,3]")
        if input_shape[0].dim_param != "batch" or output_shape[0].dim_param != "batch":
            raise RuntimeError("Exported ONNX batch dimension is not dynamic")

        session = ort.InferenceSession(str(temporary_model), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(SEED)
        errors = []
        for batch_size in (1, 3):
            sample = rng.random((batch_size, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            with torch.inference_mode():
                pytorch_logits = model(torch.from_numpy(sample)).cpu().numpy()
            onnx_logits = session.run(["logits"], {"input": sample})[0]
            if onnx_logits.shape != (batch_size, 3) or not np.isfinite(onnx_logits).all():
                raise RuntimeError(f"Invalid ONNX output shape/values: {onnx_logits.shape}")
            difference = np.abs(pytorch_logits - onnx_logits)
            np.testing.assert_allclose(pytorch_logits, onnx_logits, rtol=RTOL, atol=ATOL)
            errors.extend(difference.reshape(-1).tolist())

        # Publish the ONNX only after graph and PyTorch/ORT parity checks pass.
        pending_model = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".pending")
        pending_model.write_bytes(temporary_model.read_bytes())
        os.replace(pending_model, MODEL_PATH)

    validation = {
        "status": "passed",
        "purpose": "runtime smoke-test parity only; no PAD accuracy claim",
        "provider": "CPUExecutionProvider",
        "input_name": "input",
        "output_name": "logits",
        "input_shapes": [[1, 3, INPUT_SIZE, INPUT_SIZE], [3, 3, INPUT_SIZE, INPUT_SIZE]],
        "output_shapes": [[1, 3], [3, 3]],
        "opset": OPSET,
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs_error": float(max(errors, default=0.0)),
        "mean_abs_error": float(np.mean(errors)) if errors else 0.0,
        "onnx_sha256": sha256_file(MODEL_PATH),
    }
    write_json(VALIDATION_PATH, validation)

    runtime_config = {
        "schema": "face_auth_pad_runtime_config_v1",
        "model_file": MODEL_PATH.name,
        "model_sha256": sha256_file(MODEL_PATH),
        "source_checkpoint_file": CHECKPOINT_PATH.name,
        "source_checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "architecture_source_file": SOURCE_PATH.name,
        "architecture_source_sha256": sha256_file(SOURCE_PATH),
        "model_status": "official_pad_pretrained_base_checkpoint; not locally fine-tuned",
        "model_img_size": INPUT_SIZE,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": ["batch", 3, INPUT_SIZE, INPUT_SIZE],
        "output_shape": ["batch", 3],
        "class_names": ["real", "physical_spoof", "digital_spoof"],
        "color_order": "BGR",
        "input_range": "[0,1]",
        "mean": None,
        "std": None,
        "apply_gamma": False,
        "bbox_expansion_factor": 2.7,
        "crop_mode": "minifasnet_train_v1",
        "crop_smoothing": False,
        "predictor_threshold_probability": 0.5,
        "threshold_source": "uncalibrated 0.5 placeholder for runtime smoke testing only",
        "threshold_status": "not calibrated",
        "onnx_opset": OPSET,
        "onnx_validation_file": VALIDATION_PATH.name,
    }
    write_json(RUNTIME_CONFIG_PATH, runtime_config)

    print(f"ONNX: {MODEL_PATH}")
    print(f"Runtime config: {RUNTIME_CONFIG_PATH}")
    print(f"Parity validation: {VALIDATION_PATH}")
    print(f"Checkpoint SHA256: {runtime_config['source_checkpoint_sha256']}")
    print(f"ONNX SHA256: {runtime_config['model_sha256']}")
    print(f"Max PyTorch/ONNX absolute difference: {validation['max_abs_error']:.8g}")
    print("Threshold 0.5 is uncalibrated; use this setup for runtime smoke testing only.")


if __name__ == "__main__":
    main()
