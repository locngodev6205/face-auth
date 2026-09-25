# Danh mục Notebooks & Nghiên cứu Thực nghiệm — Anti-Spoofing (PAD)

Thư mục này tập hợp toàn bộ các Jupyter Notebook, mã nguồn chẩn đoán và tài liệu quy chuẩn (*protocol*) nghiên cứu thực nghiệm chống giả mạo khuôn mặt (Face Presentation Attack Detection - PAD) của đề tài.

Mỗi thư mục đại diện cho một phân hệ độc lập, trong đó **mã nguồn huấn luyện và notebook đánh giá của mô hình nào được tổ chức tập trung đúng vào thư mục của mô hình đó**.

---

## 🗺️ Bản đồ Cây Thư mục

```text
face_auth/antispoof/notebooks/
├── eda/                        <-- Khám phá Dữ liệu CelebA-Spoof & Khảo sát Bộ dò SCRFD
├── e1_spatial_v0/              <-- Kiểm thử Nhanh E1 v0 Baseline (MobileNetV3-Large, 10K Mẫu Sơ bộ)
├── e1_spatial_v5_1/            <-- Thực nghiệm & Đánh giá sâu E1 v5.1 Baseline (MobileNetV3-Large, GT BBox)
├── e1_spatial_v5_2/            <-- Huấn luyện & Protocol E1 v5.2 Detector-Aligned (MobileNetV3-Large)
├── e1_spatial_v5_3/            <-- Thực nghiệm & Đánh giá Held-out E1 v5.3 (MobileNetV3-Small Edge — Active Model)
├── e2_frequency_dct/           <-- Thực nghiệm & Đánh giá Held-out E2 Tần số DCT
└── diagnostics/                <-- Bộ công cụ Chẩn đoán Runtime (Webcam & CLI)
```

---

## 📖 Chi tiết Tác dụng của Từng File theo Phân hệ Mô hình

### `eda/` — Khám phá Dữ liệu & Khảo sát Bộ dò (Exploratory Data & Detector Analysis)
Khảo sát phân phối dữ liệu CelebA-Spoof và phân tích tương quan hình học với bộ phát hiện khuôn mặt SCRFD:
* [celeba_spoof_eda.ipynb](eda/celeba_spoof_eda.ipynb): Phân tích phân phối số lượng ảnh Real vs Spoof, cấu trúc nhãn đa tác vụ (43 thuộc tính), cảm biến chụp (sensor), điều kiện chiếu sáng và phân loại dạng tấn công (bản in, màn hình, mặt nạ).
* [celeba_spoof_eda_summary.json](eda/celeba_spoof_eda_summary.json): Lưu trữ các chỉ số thống kê tổng hợp được trích xuất từ notebook (số lượng mẫu train/val/test, tỉ lệ mất cân bằng giữa các lớp).
* [celeba_spoof_sample_gallery.png](eda/celeba_spoof_sample_gallery.png): Bảng ảnh trực quan hóa mẫu đại diện cho mặt thật và các loại tấn công giả mạo tiêu biểu để quan sát trực tiếp đặc trưng hình ảnh.
* [celeba_scrfd_bbox_shift_analysis.ipynb](eda/celeba_scrfd_bbox_shift_analysis.ipynb): Nghiên cứu đo lường độ lệch (shift analysis v1) giữa BBox nhãn chuẩn của CelebA-Spoof và BBox SCRFD nhận diện (phát hiện độ lệch tâm 2.5% và IoU trung vị 0.878).
* [celeba_scrfd_crop_factor_v2.ipynb](eda/celeba_scrfd_crop_factor_v2.ipynb): Nghiên cứu tối ưu hình học nâng cao (v2): quét thực nghiệm để tìm ra hệ số mở rộng khung hình lý tưởng ($1.55\times$) và ngưỡng kích thước mặt tối thiểu (48px) cho bộ tiền xử lý.

---

### `e1_spatial_v0/` — Huấn luyện & Đánh giá E1 v0 Baseline (MobileNetV3-Large, 10K Mẫu Sơ bộ)
Nghiên cứu mô hình baseline đời đầu của nhánh Không gian (Spatial Domain) MobileNetV3-Large (3 class):
* [e1_v5_0_training_executed.ipynb](e1_spatial_v0/e1_v5_0_training_executed.ipynb): Nhật ký chạy huấn luyện hoàn chỉnh của mô hình E1 v0 (MobileNetV3-Large 3-class) trên Kaggle GPU kèm trích xuất ONNX.
* [celeba_spoof_quicktest_10k.ipynb](e1_spatial_v0/celeba_spoof_quicktest_10k.ipynb): Pipeline đánh giá nhanh mô hình trên 10,000 ảnh kiểm thử ngẫu nhiên có chia tầng (stratified) nhằm kiểm chứng pipeline tiền xử lý và đo thời gian inference của mô hình E1 v0.
* Toàn bộ kết quả đầu ra nghiệm thu (biểu đồ ROC/DET, phân phối điểm, quét ngưỡng và phân tích ca lỗi) được lưu trữ tập trung tại [E1_v0_mnv3_large/test/](../models/smartface_pad_artifacts/E1_v0_mnv3_large/test/):
  * `quick_test_summary.json`: Báo cáo chỉ số (Accuracy 94.83%, AUC 97.34%, ACER 4.13%, Latency 17.53 ms/ảnh).
  * `quick_test_evaluation_plots.png`: Biểu đồ ROC, phân phối logit diff và trade-off APCER vs BPCER.
  * `threshold_sweep_results.csv`: Bảng quét 500 ngưỡng logit difference và chi tiết lỗi.
  * `worst_failure_cases.png`: Lưới các ca dự đoán sai lệch nghiêm trọng nhất.

---

### `e1_spatial_v5_1/` — Huấn luyện & Đánh giá Sâu E1 v5.1 (Baseline)
Nghiên cứu nhánh mô hình Không gian (Spatial Domain) phiên bản ổn định v5.1 (MobileNetV3-Large, Ground Truth Crop 1.5x):
* [e1_v5_1_training_template.ipynb](e1_spatial_v5_1/e1_v5_1_training_template.ipynb): Notebook huấn luyện mẫu đã làm sạch cell output (clean template), cấu hình sẵn cơ chế Time-Balanced 100K mẫu, phân chia Subject-Disjoint (seed 42) sẵn sàng đưa lên Kaggle chạy ngay.
* [e1_v5_1_training_executed.ipynb](e1_spatial_v5_1/e1_v5_1_training_executed.ipynb): Nhật ký chạy thực tế hoàn chỉnh (chứa toàn bộ output, log epoch, đồ thị loss và validation metric qua từng bước) của lượt huấn luyện trên Kaggle GPU.
* [e1_v5_1_deep_eval_template.ipynb](e1_spatial_v5_1/e1_v5_1_deep_eval_template.ipynb): Notebook mẫu sạch (chưa chạy) dùng để chạy đánh giá sâu mô hình E1 v5.1 sau huấn luyện.
* [e1_v5_1_deep_eval.ipynb](e1_spatial_v5_1/e1_v5_1_deep_eval.ipynb): Nhật ký đánh giá toàn diện đã thực thi: tính khoảng tin cậy 95% bằng Bootstrap, phân rã ma trận nhầm lẫn theo từng loại tấn công cụ thể và kiểm chứng tính tương đương đầu ra (parity check) giữa PyTorch và ONNX.

---

### `e1_spatial_v5_2/` — Huấn luyện & Protocol E1 v5.2 (Detector-Aligned Large)
Huấn luyện nhánh Không gian căn chỉnh BBox SCRFD sử dụng kiến trúc MobileNetV3-Large:
* [e1_v5_2_detector_aligned.ipynb](e1_spatial_v5_2/e1_v5_2_detector_aligned.ipynb): Notebook huấn luyện mô hình E1 v5.2 sử dụng trực tiếp Bounding Box phát hiện bởi SCRFD thay vì Ground Truth BBox để đồng bộ phân phối với lúc triển khai.
* [E1_v5_2_final_retrain_protocol.md](e1_spatial_v5_2/E1_v5_2_final_retrain_protocol.md): Văn bản quy chuẩn học thuật quy định chặt chẽ các siêu tham số, trọng số loss, augmentation và early stopping cho v5.2.

---

### `e1_spatial_v5_3/` — Huấn luyện & Đánh giá Held-out E1 v5.3 (MobileNetV3-Small Edge)
**Phiên bản mô hình sản xuất chính thức đang chạy trong hệ thống SmartFace** (tối ưu hóa cho Edge/AIoT với 1.1M tham số):
* [e1_v5_3_training.ipynb](e1_spatial_v5_3/e1_v5_3_training.ipynb): Notebook huấn luyện MobileNetV3-Small (576 feature dim) kết hợp với bộ tiền xử lý SCRFD $1.55\times$ BBox expansion.
* [e1_heldout_test.ipynb](e1_spatial_v5_3/e1_heldout_test.ipynb): **Notebook đánh giá kiểm thử độc lập mô hình E1 v5.3** (`mnv3s_e1_preliminary_v5_3_edge_best.onnx`) trên tập dữ liệu Held-out Test hoàn toàn mới.
* [E1_heldout_test_kaggle_guide.md](e1_spatial_v5_3/E1_heldout_test_kaggle_guide.md): Cẩm nang hướng dẫn thiết lập môi trường Kaggle để chạy đánh giá Held-out Test riêng cho mô hình E1 v5.3.
* [E1_artifacts_for_E2_kaggle_guide.md](e1_spatial_v5_3/E1_artifacts_for_E2_kaggle_guide.md): Cẩm nang hướng dẫn bảo toàn và tái sử dụng Data Artifacts của E1 v5.3 cho Kaggle E2.

---

### `e2_frequency_dct/` — Huấn luyện & Đánh giá Held-out E2 (Tần số 2D-DCT)
Nhánh mô hình thuần miền Tần số (Frequency Domain) dựa trên biến đổi 2D-DCT và kiểm thử độc lập:
* [e2_frequency_only_dct_train.ipynb](e2_frequency_dct/e2_frequency_only_dct_train.ipynb): **Notebook huấn luyện mô hình E2** chỉ dùng đặc trưng tần số 2D Discrete Cosine Transform (trích xuất dấu vết lưới màn hình điện tử moiré và viền in ấn).
* [e2_heldout_test.ipynb](e2_frequency_dct/e2_heldout_test.ipynb): **Notebook đánh giá kiểm thử độc lập mô hình E2** trên tập dữ liệu Held-out Test để so sánh ablation đối đầu trực tiếp với nhánh E1.
* [E2_frequency_only_dct_protocol.md](e2_frequency_dct/E2_frequency_only_dct_protocol.md): Tài liệu khoa học mô tả nền tảng toán học của biến đổi 2D-DCT, cấu trúc khối biến đổi và kiến trúc mạng nơ-ron phân tích miền tần số.
* [E1_artifacts_for_E2_kaggle_guide.md](e2_frequency_dct/E1_artifacts_for_E2_kaggle_guide.md): Cẩm nang hướng dẫn nạp các Data Artifacts (manifest split, cache BBox SCRFD) từ E1 v5.3 sang E2 nhằm đảm bảo tính công bằng dữ liệu.
* [E1_E2_heldout_test_kaggle_guide.md](e2_frequency_dct/E1_E2_heldout_test_kaggle_guide.md): Hướng dẫn chạy kiểm thử Held-out và tổng hợp bảng so sánh đối đầu giữa hai nhánh E1 vs E2.

---

### `diagnostics/` — Bộ Công cụ Chẩn đoán Runtime Thực tế (Camera & System Diagnostics)
Chuyên trách kiểm tra tính toàn vẹn hệ thống và độ lệch miền trên thiết bị camera thực tế (không chứa notebook huấn luyện mô hình):
* [camera_pad_diagnostic.py](diagnostics/camera_pad_diagnostic.py): Công cụ chạy trên máy cục bộ có gắn webcam: kiểm tra trực tiếp qua video stream, quét qua các biến thể tiền xử lý (RGB/BGR, gamma correction, crop scale) và trích xuất bảng dữ liệu `camera_pad_diagnostic.csv`.
* [camera_vs_kaggle_pad_diagnostic_guide.md](diagnostics/camera_vs_kaggle_pad_diagnostic_guide.md): Tài liệu hướng dẫn quy trình chuẩn 6 bước chẩn đoán nguyên nhân khi đưa mô hình ra webcam thực tế (phân biệt giữa lỗi crop, lỗi thứ tự màu BGR/RGB, gamma mismatch và domain shift thực tế).
* [diagnose_pad_runtime.py](diagnostics/diagnose_pad_runtime.py): Script dòng lệnh (CLI) kiểm tra tính toàn vẹn của mô hình (checksum mã SHA256, runtime contract, logit thô và xác suất) trên từng bức ảnh tĩnh độc lập mà không cần bật camera.

---

## 📌 Khớp nối Lưu trữ Dài hạn (Artifacts Mapping)

Toàn bộ trọng số mô hình PyTorch (`.pth`), mô hình suy luận ONNX (`.onnx`), file cấu hình ngưỡng (`runtime_config.json`) và cache dữ liệu (`.json`, `.npz`) tương ứng với các notebook trên được bảo toàn tập trung tại:
👉 [face_auth/antispoof/models/smartface_pad_artifacts/](../models/smartface_pad_artifacts/)
