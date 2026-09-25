# Danh mục Notebooks & Nghiên cứu Thực nghiệm — Anti-Spoofing (PAD)

Thư mục này tập hợp toàn bộ các Jupyter Notebook, mã nguồn chẩn đoán và tài liệu quy chuẩn (*protocol*) nghiên cứu thực nghiệm chống giả mạo khuôn mặt (Face Presentation Attack Detection - PAD) của đề tài.

---

## 🗺️ Bản đồ Cây Thư mục

```text
face_auth/antispoof/notebooks/
├── eda/                        <-- Khám phá Dữ liệu CelebA-Spoof
├── quick_test/                 <-- Kiểm thử Nhanh 10K Mẫu Sơ bộ
├── e1_spatial_v5_1/            <-- Thực nghiệm E1 v5.1 Không gian (Baseline)
├── e1_spatial_v5_2/            <-- Thực nghiệm E1 v5.2 Detector-Aligned
├── e2_frequency_dct/           <-- Thực nghiệm E2 Tần số DCT & Held-out Test
└── diagnostics/                <-- Bộ công cụ Chẩn đoán Độ lệch Miền & Hình học BBox
```

---

## 📖 Chi tiết Tác dụng của Từng File trong Hệ thống

### `eda/` — Khám phá & Đánh giá Dữ liệu (Exploratory Data Analysis)
Khảo sát ban đầu về chất lượng và độ đa dạng của tập dữ liệu CelebA-Spoof:
* [celeba_spoof_eda.ipynb](eda/celeba_spoof_eda.ipynb): Phân tích phân phối số lượng ảnh Real vs Spoof, cấu trúc nhãn đa tác vụ (43 thuộc tính), cảm biến chụp (sensor), điều kiện chiếu sáng và phân loại dạng tấn công (bản in, màn hình, mặt nạ).
* [celeba_spoof_eda_summary.json](eda/celeba_spoof_eda_summary.json): Lưu trữ các chỉ số thống kê tổng hợp được trích xuất từ notebook (số lượng mẫu train/val/test, tỉ lệ mất cân bằng giữa các lớp).
* [celeba_spoof_sample_gallery.png](eda/celeba_spoof_sample_gallery.png): Bảng ảnh trực quan hóa mẫu đại diện cho mặt thật và các loại tấn công giả mạo tiêu biểu để quan sát trực tiếp đặc trưng hình ảnh.

---

### `quick_test/` — Kiểm thử Nhanh Mô hình Sơ bộ (10K Samples)
Thực nghiệm nhanh trên tập con 10,000 mẫu để kiểm tra tính khả thi và đo lường độ trễ:
* [celeba_spoof_quicktest_10k.ipynb](quick_test/celeba_spoof_quicktest_10k.ipynb): Pipeline đánh giá nhanh mô hình trên 10,000 ảnh kiểm thử ngẫu nhiên có chia tầng (stratified) nhằm kiểm chứng pipeline tiền xử lý và đo thời gian inference.
* [quick_test_summary.json](quick_test/quick_test_summary.json): Báo cáo số liệu nghiệm thu của đợt test nhanh: Accuracy, APCER (tỉ lệ lọt giả mạo), BPCER (tỉ lệ từ chối mặt thật), HTER và độ trễ trung bình.
* [quick_test_evaluation_plots.png](quick_test/quick_test_evaluation_plots.png): Biểu đồ đường cong ROC, phân phối xác suất dự đoán (Real score distribution) và đường cong đánh đổi APCER vs BPCER.
* [threshold_sweep_results.csv](quick_test/threshold_sweep_results.csv): Bảng dữ liệu quét ngưỡng phân loại $p \in [0.0, 1.0]$ với bước nhảy mịn, ghi nhận chi tiết số ca False Accept, False Reject và HTER tương ứng ở từng mốc.
* [worst_failure_cases.png](quick_test/worst_failure_cases.png): Lưới hình ảnh hiển thị các trường hợp dự đoán sai lệch nghiêm trọng nhất (mặt thật bị từ chối với score cực thấp hoặc tấn công tinh vi bị nhận nhầm là thật) để phân tích nguyên nhân gốc rễ.

---

### `e1_spatial_v5_1/` — Mô hình Không gian E1 v5.1 (Baseline)
Nghiên cứu nhánh mô hình Không gian (Spatial Domain) phiên bản ổn định v5.1 (MobileNetV3):
* [e1_v5_1_training_template.ipynb](e1_spatial_v5_1/e1_v5_1_training_template.ipynb): Notebook huấn luyện mẫu đã được làm sạch đầu ra (clean template), cấu hình sẵn cơ chế Time-Balanced 100K mẫu, phân chia Subject-Disjoint (seed 42) sẵn sàng đưa lên Kaggle chạy ngay.
* [e1_v5_1_training_executed.ipynb](e1_spatial_v5_1/e1_v5_1_training_executed.ipynb): Nhật ký chạy thực tế hoàn chỉnh (chứa toàn bộ output, log epoch, đồ thị loss và validation metric qua từng bước) của lượt huấn luyện trên Kaggle GPU.
* [e1_v5_1_deep_eval.ipynb](e1_spatial_v5_1/e1_v5_1_deep_eval.ipynb): Notebook đánh giá toàn diện sau huấn luyện: tính khoảng tin cậy 95% bằng Bootstrap, phân rã ma trận nhầm lẫn theo từng loại tấn công cụ thể và kiểm chứng tính tương đương đầu ra (parity check) giữa PyTorch và ONNX.

---

### `e1_spatial_v5_2/` — Mô hình E1 v5.2 Detector-Aligned
Cải tiến nhánh Không gian để đồng bộ hóa hoàn toàn với bộ nhận diện khuôn mặt SCRFD:
* [e1_v5_2_detector_aligned.ipynb](e1_spatial_v5_2/e1_v5_2_detector_aligned.ipynb): Notebook huấn luyện mô hình E1 v5.2 sử dụng trực tiếp Bounding Box phát hiện bởi SCRFD thay vì Ground Truth BBox của CelebA-Spoof, loại bỏ triệt để hiện tượng lệch tọa độ lúc triển khai thực tế.
* [E1_v5_2_final_retrain_protocol.md](e1_spatial_v5_2/E1_v5_2_final_retrain_protocol.md): Văn bản quy chuẩn học thuật quy định chặt chẽ các siêu tham số (hyperparameters), trọng số mất mát (loss weights), phương pháp tăng cường ảnh (augmentations) và điều kiện dừng sớm (early stopping).

---

### `e2_frequency_dct/` — Nhánh Tần số E2 & Đánh giá Held-out Test
Phát triển nhánh mô hình Tần số (Frequency Domain) dựa trên biến đổi 2D-DCT và kiểm thử độc lập:
* [e2_frequency_only_dct_train.ipynb](e2_frequency_dct/e2_frequency_only_dct_train.ipynb): Notebook huấn luyện mô hình E2 chỉ dùng đặc trưng tần số 2D Discrete Cosine Transform (trích xuất dấu vết lưới màn hình điện tử moiré và viền in ấn).
* [e1_heldout_test.ipynb](e2_frequency_dct/e1_heldout_test.ipynb): Đánh giá kiểm thử khách quan mô hình Không gian E1 trên tập dữ liệu Held-out Test độc lập (chưa từng xuất hiện trong quá trình train/val).
* [e2_heldout_test.ipynb](e2_frequency_dct/e2_heldout_test.ipynb): Đánh giá kiểm thử khách quan mô hình Tần số E2 trên cùng tập dữ liệu Held-out Test để so sánh trực diện hiệu quả.
* [E2_frequency_only_dct_protocol.md](e2_frequency_dct/E2_frequency_only_dct_protocol.md): Tài liệu khoa học mô tả nền tảng toán học của biến đổi 2D-DCT, cấu trúc khối biến đổi và kiến trúc mạng nơ-ron phân tích miền tần số.
* [E1_artifacts_for_E2_kaggle_guide.md](e2_frequency_dct/E1_artifacts_for_E2_kaggle_guide.md): Cẩm nang hướng dẫn đóng gói và tái sử dụng đúng các Data Artifacts (manifest split, cache BBox SCRFD) từ E1 sang E2 nhằm đảm bảo tính toàn vẹn khoa học.
* [E1_E2_heldout_test_kaggle_guide.md](e2_frequency_dct/E1_E2_heldout_test_kaggle_guide.md): Hướng dẫn từng bước cách thiết lập môi trường Kaggle để chạy kiểm thử Held-out và xuất báo cáo so sánh hai nhánh E1 vs E2.

---

### `diagnostics/` — Bộ Công cụ Chẩn đoán Thực tế (Diagnostic & Domain Shift Toolkit)
Truy tìm và khắc phục sự sai lệch giữa môi trường huấn luyện và camera thực tế:
* [camera_vs_kaggle_pad_diagnostic_guide.md](diagnostics/camera_vs_kaggle_pad_diagnostic_guide.md): Tài liệu hướng dẫn quy trình chuẩn 6 bước chẩn đoán nguyên nhân khi đưa mô hình ra webcam thực tế (phân biệt giữa lỗi crop, lỗi thứ tự màu BGR/RGB, gamma mismatch và domain shift thực tế).
* [camera_pad_diagnostic.py](diagnostics/camera_pad_diagnostic.py): Công cụ chạy trên máy cục bộ có gắn webcam: tương tác thời gian thực, quét qua các biến thể tiền xử lý (RGB/BGR, gamma correction, crop scale) và trích xuất bảng dữ liệu `camera_pad_diagnostic.csv`.
* [pad_post_training_deep_evaluation_kaggle.ipynb](diagnostics/pad_post_training_deep_evaluation_kaggle.ipynb): Template Kaggle chuyên biệt dùng để nhận file CSV xuất từ webcam máy cục bộ, đối chiếu và vẽ biểu đồ so sánh phân phối điểm số của mặt Real ngoài đời với mặt Real trong tập CelebA-Spoof.
* [celeba_scrfd_bbox_shift_analysis.ipynb](diagnostics/celeba_scrfd_bbox_shift_analysis.ipynb): Nghiên cứu đo lường độ lệch (shift analysis v1) giữa Bounding Box nhãn chuẩn của CelebA-Spoof và BBox do bộ dò SCRFD nhận diện (phát hiện độ lệch tâm 2.5% và IoU trung vị 0.878).
* [celeba_scrfd_crop_factor_v2.ipynb](diagnostics/celeba_scrfd_crop_factor_v2.ipynb): Nghiên cứu tối ưu hình học nâng cao (v2): quét thực nghiệm để tìm ra hệ số mở rộng khung hình lý tưởng ($1.55\times$) và ngưỡng kích thước mặt tối thiểu (minimum face size) giúp mô hình đạt độ nhạy tốt nhất.
* [diagnose_pad_runtime.py](diagnostics/diagnose_pad_runtime.py): Script dòng lệnh (CLI) kiểm tra tính toàn vẹn của mô hình (checksum mã SHA256, runtime contract, logit thô và xác suất) trên từng bức ảnh tĩnh độc lập mà không cần bật camera.

---

## 📌 Khớp nối Lưu trữ Dài hạn (Artifacts Mapping)

Toàn bộ trọng số mô hình PyTorch (`.pth`), mô hình suy luận ONNX (`.onnx`), file cấu hình ngưỡng (`runtime_config.json`) và cache dữ liệu (`.json`, `.npz`) tương ứng với các notebook trên được bảo toàn tập trung tại:
👉 [face_auth/antispoof/models/smartface_pad_artifacts/](../models/smartface_pad_artifacts/)
