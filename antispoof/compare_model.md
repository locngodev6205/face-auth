## 🎯 Classifier (Real vs Spoof): MiniFASNetV2SE vs MobileNetV3-Large

| Tiêu chí | MiniFASNetV2SE | MobileNetV3-Large |
|---|---|---|
| Params | ~0.435M | ~3–5.4M |
| FLOPs | ~0.081 GFLOPs | ~0.15–0.22 GFLOPs |
| Kích thước file | ~1.82MB (FP32) / ~600KB (INT8) | Nặng hơn nhiều lần |
| Tốc độ CPU | <10ms | Chậm hơn đáng kể |
| Accuracy (CelebA-Spoof) | ~98.2% (Real 97.6% / Spoof 98.7%, AUC 0.998) | Tùy fine-tune, không có số chuẩn riêng cho spoof |
| Thiết kế | Chuyên biệt: SE attention + Fourier-spectrum auxiliary supervision (bắt artifact tần số từ ảnh in/màn hình) | Backbone tổng quát (ImageNet), phải tự học pattern spoof từ đầu |
| Hệ sinh thái | PyTorch/ONNX Runtime — cài dễ trên ARM/Pi | Cần OpenVINO trong repo hiện tại — khó cài trên Pi (không có pip package ARM chính thức) |
| **Phù hợp Pi** | ✅ Rất phù hợp | ❌ Nặng + khó cài runtime trên Pi |

**→ MiniFASNetV2SE thắng cho use case Pi:** nhẹ hơn ~10 lần, nhanh hơn, thiết kế đúng chuyên môn, và tránh được vấn đề OpenVINO không hỗ trợ ARM tốt.

---

## 🎯 Face Detector: YuNet vs SCRFD-0.5GF

| Tiêu chí | YuNet | SCRFD-0.5GF |
|---|---|---|
| Params | ~75K | ~570K (gấp ~7.6 lần) |
| Backbone | ShuffleNetV2 + FPN | ResNet-style depth-wise conv |
| WIDERFace Easy/Medium/Hard | 88.44 / 86.56 / **75.03** | **90.57 / 88.12** / 68.51 |
| Landmark | Có (5 điểm) | Có (5 điểm) |
| Dependency | Có sẵn trong `opencv-python` — không cần cài thêm | Cần ONNX Runtime/ncnn riêng |
| Tốc độ | Nhanh hơn rõ rệt (ít tham số hơn nhiều) | Chậm hơn |
| **Phù hợp Pi** | ✅ Rất phù hợp | ⚠️ Chạy được nhưng chậm hơn + thêm dependency |

**→ YuNet thắng cho use case Pi:** nhẹ hơn, nhanh hơn, tích hợp sẵn trong OpenCV (bạn đã dùng `cv2` rồi nên không tốn thêm dependency), và thậm chí chính xác hơn ở ảnh khó (Hard: 75.03 vs 68.51).

---

## 📌 Tổng kết cho toàn bộ pipeline chạy trên Raspberry Pi
```
Camera → YuNet (detect + landmark) → crop/align face → MiniFASNetV2SE (real/spoof) → kết quả
```
Cả 2 lựa chọn đều nhẹ, nhanh, và **cùng tránh được vấn đề OpenVINO không hỗ trợ ARM tốt** mà repo gốc `Anti-Spoofing` bạn đang dùng gặp phải — tức là bạn sẽ cần điều chỉnh lại pipeline khỏi OpenVINO, chuyển sang ONNX Runtime (cho MiniFASNetV2SE) + OpenCV DNN có sẵn (cho YuNet).
