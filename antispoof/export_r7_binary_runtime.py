"""Export the selected R7-100K checkpoint and its frozen Val15K runtime config.

Run from the project root: python3 antispoof/export_r7_binary_runtime.py
The deployed graph is the clean binary PAD path; spectral selection and auxiliary
heads are training-only. This does not retrain or recalibrate the checkpoint.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "antispoof/models"
SOURCE_PATH = MODEL_DIR / "MiniFASNet.py"
DEFAULT_RUN_DIR = PROJECT_ROOT / "antispoof/notebooks/final/binary_crossdomain_v2/output/17_scale100k_C_P3SF_R7SC_crop15/runs/R7_SC_100K_crop15"
RUN_ID = "R7_SC_100K_crop15"
OPSET = 17


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(pending, path)


# Same feature path, PAD and auxiliary module names as notebook17 PADNet(copied=True).
# Loading the whole training state strictly ensures no trunk/head weights are skipped.
class R7BinaryPAD(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.trunk = module.MiniFASNetV2(
            embedding_size=128, conv6_kernel=(5, 5), drop_p=.2,
            num_classes=3, img_channel=3,
        )
        self.trunk.prob = nn.Identity()
        self.pad_head = nn.Linear(128, 2)
        self.spoof_type_head = nn.Linear(128, 11)
        self.lighting_head = nn.Linear(128, 5)
        self.attributes_head = nn.Linear(128, 40)

    def features(self, x):
        t = self.trunk
        for layer in (t.conv1, t.conv2_dw, t.conv_23, t.conv_3, t.conv_34,
                      t.conv_4, t.conv_45, t.conv_5, t.conv_6_sep,
                      t.conv_6_dw, t.conv_6_flatten):
            x = layer(x)
        if t.embedding_size != 512:
            x = t.linear(x)
        return t.drop(t.bn(x))

    def forward(self, x):
        return self.pad_head(self.features(x))


def load_checkpoint(checkpoint_path, config_path):
    cfg = read_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["run_id"] != RUN_ID or cfg["run_id"] != RUN_ID or checkpoint["config"] != cfg:
        raise ValueError("Checkpoint and run config do not identify the selected R7-100K run")
    spec = cfg["spec"]
    if spec["method"] != "HARMFUL" or any(cfg[key] != 1.5 for key in (
            "train_crop_factor", "val_crop_factor", "test_crop_factor", "lcc_crop_factor")):
        raise ValueError("Expected the frozen harmful-gated R7 crop1.5 configuration")
    folder = config_path.parent
    freeze = read_json(folder / "evaluation_freeze.json")
    completion = read_json(folder / "completion.json")
    threshold = float(freeze["locked_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Frozen source threshold is not finite")
    if any(item["run_id"] != RUN_ID or item["best_epoch"] != checkpoint["epoch"]
           for item in (freeze, completion)):
        raise ValueError("Checkpoint epoch differs from evaluation freeze/completion")
    if completion["locked_threshold"] != threshold or freeze["manifest_sha256"] != cfg["manifest_sha256"]:
        raise ValueError("Frozen threshold/manifests differ from the completed run")
    source_spec = importlib.util.spec_from_file_location("r7_minifasnet_source", SOURCE_PATH)
    module = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(module)
    audit = read_json(folder / "initialization_audit.json")
    if sha256(SOURCE_PATH) != audit["source_sha256"]:
        raise ValueError("Local MiniFASNet source differs from the training architecture")
    model = R7BinaryPAD(module)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, cfg, freeze


def export(checkpoint_path, config_path, output_dir):
    torch.set_num_threads(1)
    model, cfg, freeze = load_checkpoint(checkpoint_path, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / (RUN_ID + ".onnx")
    runtime_path = output_dir / (RUN_ID + "_runtime_config.json")
    validation_path = output_dir / (RUN_ID + "_onnx_validation.json")
    threshold = float(freeze["locked_threshold"])
    rng = np.random.default_rng(100)
    cases = [np.zeros((1, 3, 80, 80), np.float32),
             np.ones((1, 3, 80, 80), np.float32),
             rng.random((3, 3, 80, 80), dtype=np.float32),
             rng.random((8, 3, 80, 80), dtype=np.float32)]
    checks = []
    with tempfile.TemporaryDirectory(prefix="r7_binary_export_") as tmp:
        candidate = Path(tmp) / model_path.name
        torch.onnx.export(model, torch.zeros(1, 3, 80, 80), str(candidate),
                          input_names=["input"], output_names=["logits"],
                          opset_version=OPSET, dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        graph = onnx.load(str(candidate))
        onnx.checker.check_model(graph)
        if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
            raise ValueError("R7 deployment graph must have only one input and binary output")
        # Export should prune auxiliary heads entirely from the deployment graph.
        if any(any(head in tensor.name for head in ("spoof_type_head", "lighting_head", "attributes_head"))
               for tensor in graph.graph.initializer):
            raise ValueError("Training-only auxiliary weights leaked into deployment")
        session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != ["batch", 3, 80, 80] or session.get_outputs()[0].shape != ["batch", 2]:
            raise ValueError("Unexpected ONNX binary PAD input/output shape")
        for index, x in enumerate(cases):
            with torch.inference_mode():
                expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(["logits"], {"input": x})[0]
            if actual.shape != (len(x), 2) or not np.isfinite(actual).all():
                raise ValueError("Invalid ONNX binary logits")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            expected_score = expected[:, 0] - expected[:, 1]
            actual_score = actual[:, 0] - actual[:, 1]
            np.testing.assert_allclose(actual_score, expected_score, rtol=1e-4, atol=2e-5)
            np.testing.assert_array_equal(actual_score >= threshold, expected_score >= threshold)
            checks.append({"case": index, "batch_size": len(x),
                           "max_abs_logit_error": float(np.abs(actual - expected).max()),
                           "max_abs_score_error": float(np.abs(actual_score - expected_score).max()),
                           "verdicts_match": True})
        pending = model_path.with_suffix(".onnx.pending")
        pending.write_bytes(candidate.read_bytes())
        os.replace(pending, model_path)
    probability = 1 / (1 + math.exp(-threshold))
    runtime = {
        "schema": "face_auth_pad_runtime_config_v1", "run_id": RUN_ID,
        "model_file": model_path.name, "model_sha256": sha256(model_path),
        "model_status": "trained R7_SC_100K_crop15; best checkpoint selected on Val15K",
        "model_img_size": 80, "input_name": "input", "output_name": "logits",
        "input_shape": ["batch", 3, 80, 80], "output_shape": ["batch", 2],
        "output_classes": 2, "class_names": ["real", "spoof"],
        "score_type": "binary_logit_difference", "score_formula": "logits[:,0] - logits[:,1]",
        "decision_rule": "real iff score >= calibrated_logit_threshold",
        "color_order": "BGR", "input_range": "[0,1]", "mean": None, "std": None,
        "apply_gamma": False, "bbox_expansion_factor": 1.5,
        "crop_mode": "minifasnet_train_v1", "crop_smoothing": False,
        "calibrated_logit_threshold": threshold, "predictor_threshold_probability": probability,
        "threshold_source": "Frozen scale15k_val.csv calibration for reloaded best checkpoint; no runtime refitting",
        "selected_epoch": freeze["best_epoch"], "manifest_sha256": cfg["manifest_sha256"],
        "source_checkpoint_file": str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.is_relative_to(PROJECT_ROOT) else str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_run_config_sha256": sha256(config_path),
        "architecture_source_file": SOURCE_PATH.name, "architecture_source_sha256": sha256(SOURCE_PATH),
        "onnx_opset": OPSET, "onnx_validation_file": validation_path.name,
    }
    validation = {"status": "passed", "purpose": "offline numerical parity, not camera/dataset accuracy validation",
                  "provider": "CPUExecutionProvider", "torch_version": torch.__version__,
                  "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
                  "checks": checks, "max_abs_logit_error": max(c["max_abs_logit_error"] for c in checks),
                  "model_sha256": runtime["model_sha256"], "source_checkpoint_sha256": runtime["source_checkpoint_sha256"]}
    write_json(validation_path, validation)
    write_json(runtime_path, runtime)
    print("ONNX:", model_path)
    print("Runtime config:", runtime_path)
    print("Validation:", validation_path)
    print("Logit threshold:", threshold, "P(real):", probability)
    print("Max absolute logit error:", validation["max_abs_logit_error"])
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "best.pth")
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_DIR / "config.json")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    export(args.checkpoint.resolve(), args.run_config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
