# 📓 Danh mục Notebooks & Nghiên cứu Thực nghiệm — Anti-Spoofing (PAD)

Thư mục này tập hợp toàn bộ các Jupyter Notebook, mã nguồn huấn luyện, chẩn đoán và tài liệu quy chuẩn (*protocol*) nghiên cứu thực nghiệm chống giả mạo khuôn mặt (Face Presentation Attack Detection - PAD) của đề tài SmartFace AIoT.

Mỗi thư mục con đại diện cho một phân hệ thực nghiệm độc lập, bảo toàn tính tái lập (Reproducibility) và so sánh đối đầu công bằng (Fair Comparison Contract) giữa các nhánh kiến trúc.

---

## 🗺️ Bản đồ Cây Thư Mục

```text
face_auth/antispoof/notebooks/
├── eda/                         <-- Khám phá Dữ liệu CelebA-Spoof & Khảo sát Bộ dò SCRFD
├── e1_spatial_v5_3/             <-- Thực nghiệm & Đánh giá Held-out E1 v5.3 (MobileNetV3-Small Edge — Active Model)
├── e2_frequency_dct/            <-- Thực nghiệm & Đánh giá Held-out E2 Tần số 2D-DCT
├── e3_spatial_frequency_concat/ <-- Thực nghiệm & Đánh giá E3 Fusion Đa Miền (Spatial + Frequency)
├── micro_frequency_search/      <-- Kaggle micro-search PAD Stages 0–2 (M0–M4)
├── csmr/                        <-- Nghiên cứu & Huấn luyện CSMR (Cross-Scale Multi-Representation)
├── final/                       <-- Kaggle Input Guides & Resource Packages
└── minifasnet/                  <-- Thực nghiệm & Benchmark dòng mô hình MiniFASNetV2
```

---

## 📖 Chi Tiết Từng Phân Hệ Thực Nghiệm

### 1. `eda/` — Khám Phá Dữ Liệu & Khảo Sát Bộ Dò (Exploratory Data & Detector Analysis)
Khảo sát phân phối dữ liệu CelebA-Spoof và phân tích tương quan hình học với bộ phát hiện khuôn mặt SCRFD:
* [celeba_spoof_eda.ipynb](eda/celeba_spoof_eda.ipynb): Phân tích phân phối số lượng ảnh Real vs Spoof, cấu trúc nhãn đa tác vụ (43 thuộc tính), cảm biến chụp (sensor), điều kiện chiếu sáng và phân loại dạng tấn công (bản in, màn hình, mặt nạ).
* [celeba_spoof_eda_summary.json](eda/celeba_spoof_eda_summary.json): Thống kê tổng hợp trích xuất từ notebook (mẫu train/val/test, tỉ lệ mất cân bằng).
* [celeba_spoof_sample_gallery.png](eda/celeba_spoof_sample_gallery.png): Bảng ảnh trực quan hóa mẫu mặt thật và các loại tấn công giả mạo tiêu biểu.
* [celeba_scrfd_bbox_shift_analysis.ipynb](eda/celeba_scrfd_bbox_shift_analysis.ipynb): Đo lường độ lệch (shift analysis v1) giữa BBox nhãn chuẩn và BBox SCRFD nhận diện (IoU trung vị 0.878).
* [celeba_scrfd_crop_factor_v2.ipynb](eda/celeba_scrfd_crop_factor_v2.ipynb): Nghiên cứu tối ưu hình học nâng cao (v2): quét tìm hệ số mở rộng khung hình lý tưởng ($1.55\times$) và ngưỡng mặt tối thiểu (48px).

---

### 2. `e1_spatial_v5_3/` — Huấn Luyện & Đánh Giá Held-out E1 v5.3 (MobileNetV3-Small Edge)
**Phiên bản mô hình sản xuất chính thức đang chạy trong hệ thống SmartFace** (tối ưu hóa cho Edge/AIoT với 1.11M tham số):
* [e1_v5_3_training.ipynb](e1_spatial_v5_3/e1_v5_3_training.ipynb): Notebook huấn luyện MobileNetV3-Small (576 feature dim) kết hợp bộ tiền xử lý SCRFD $1.55\times$ BBox expansion.
* [e1_heldout_test.ipynb](e1_spatial_v5_3/e1_heldout_test.ipynb): Đánh giá kiểm thử độc lập mô hình E1 v5.3 (`mnv3s_e1_preliminary_v5_3_edge_best.onnx`) trên tập dữ liệu Held-out Test 10.000 mẫu.
* [E1_heldout_test_kaggle_guide.md](e1_spatial_v5_3/E1_heldout_test_kaggle_guide.md): Cẩm nang hướng dẫn thiết lập môi trường Kaggle để chạy đánh giá Held-out Test cho E1 v5.3.
* [E1_artifacts_for_E2_kaggle_guide.md](e1_spatial_v5_3/E1_artifacts_for_E2_kaggle_guide.md): Hướng dẫn đóng băng và tái sử dụng Data Artifacts của E1 v5.3 cho Kaggle E2.

---

### 3. `e2_frequency_dct/` — Huấn Luyện & Đánh Giá Held-out E2 (Tần Số 2D-DCT)
Nhánh mô hình thuần miền Tần số (Frequency Domain) dựa trên biến đổi 2D-DCT và kiểm thử độc lập:
* [e2_frequency_only_dct_train.ipynb](e2_frequency_dct/e2_frequency_only_dct_train.ipynb): Huấn luyện mô hình E2 chỉ dùng đặc trưng tần số 2D Discrete Cosine Transform (17.1K params).
* [e2_heldout_test.ipynb](e2_frequency_dct/e2_heldout_test.ipynb): Đánh giá kiểm thử độc lập mô hình E2 trên tập Held-out Test để so sánh ablation đối đầu với E1.
* [E2_frequency_only_dct_protocol.md](e2_frequency_dct/E2_frequency_only_dct_protocol.md): Nền tảng toán học của biến đổi 2D-DCT, cấu trúc khối biến đổi và kiến trúc mạng nơ-ron phân tích tần số.
* [E1_artifacts_for_E2_kaggle_guide.md](e2_frequency_dct/E1_artifacts_for_E2_kaggle_guide.md): Hướng dẫn nạp Data Artifacts từ E1 v5.3 sang E2.
* [E1_E2_heldout_test_kaggle_guide.md](e2_frequency_dct/E1_E2_heldout_test_kaggle_guide.md): Hướng dẫn chạy kiểm thử Held-out và tổng hợp bảng so sánh đối đầu E1 vs E2.

---

### 4. `e3_spatial_frequency_concat/` — Fusion Đa Miền Không Gian + Tần Số (E3 Concat)
Thực nghiệm mô hình kết hợp đa miền Không gian (MobileNetV3-Small) và Tần số (2D-DCT Branch):
* [antispoof_e3_spatial_frequency_concat_v1.ipynb](e3_spatial_frequency_concat/antispoof_e3_spatial_frequency_concat_v1.ipynb): Notebook huấn luyện mô hình E3 Concat Fusion (1.13M params).
* [antispoof_e3_heldout_test_v1.ipynb](e3_spatial_frequency_concat/antispoof_e3_heldout_test_v1.ipynb): Đánh giá kiểm thử held-out test cho E3 (AUC: 98.57%, ACER: 9.13%).
* [E3_spatial_frequency_concat_protocol.md](e3_spatial_frequency_concat/E3_spatial_frequency_concat_protocol.md): Quy chuẩn thiết kế kiến trúc ghép nối đa miền (feature concatenation) và quy trình training.
* [E3_kaggle_run_and_artifact_guide.md](e3_spatial_frequency_concat/E3_kaggle_run_and_artifact_guide.md): Hướng dẫn chạy Kaggle GPU và xuất artifacts ONNX cho E3.

---

### 5. `micro_frequency_search/` — Kaggle PAD Micro-Search (Stages 0–2)
Bộ notebook khám phá nhanh năm biểu diễn tần số M0–M4 trên cùng manifest 5K/2K và checkpoint spatial dùng chung:
* [00_stage0_build_micro_resources.ipynb](micro_frequency_search/00_stage0_build_micro_resources.ipynb): Đóng băng micro Train/Val và tái sử dụng LCC SCRFD cache tương thích.
* [01_stage1_train_shared_micro_e1.ipynb](micro_frequency_search/01_stage1_train_shared_micro_e1.ipynb): Huấn luyện baseline MobileNetV3-Small micro dùng chung.
* [02_m0_global_dct_gap.ipynb](micro_frequency_search/02_m0_global_dct_gap.ipynb) đến [06_m4_block_dct_8x8.ipynb](micro_frequency_search/06_m4_block_dct_8x8.ipynb): 5 phương án biểu diễn tần số (Global DCT, Pool 4x4, Coord DCT, Band-Aware, Block DCT 8x8).
* [07_compare_m0_m4_results.ipynb](micro_frequency_search/07_compare_m0_m4_results.ipynb): Tổng hợp và so sánh screening heuristics.
* [MICRO_SEARCH_STAGE0_2_PROTOCOL.md](micro_frequency_search/MICRO_SEARCH_STAGE0_2_PROTOCOL.md) & [MICRO_SEARCH_STAGE0_2_FINAL_AUDIT.md](micro_frequency_search/MICRO_SEARCH_STAGE0_2_FINAL_AUDIT.md): Tài liệu quy chuẩn và audit kết quả.

---

### 6. `csmr/` & `minifasnet/` — Nghiên Cứu CSMR & Benchmark MiniFASNet
* **`csmr/`**: Thực nghiệm Cross-Scale Multi-Representation (CSMR) với các biến thể crop scale ($2.7\times$), seed frozen checks và đánh giá robustness.
* **`minifasnet/`**: Huấn luyện, tinh chỉnh siêu tham số và đánh giá các biến thể mô hình siêu nhẹ MiniFASNetV2 trên tập dữ liệu chuẩn và CASIA-FASD.

---

## 📌 Khớp Nối Lưu Trữ Dài Hạn (Artifacts Mapping)

Toàn bộ trọng số PyTorch (`.pth`), mô hình ONNX (`.onnx`), runtime configuration (`runtime_config.json`) và biểu đồ nghiệm thu được bảo toàn tại:
👉 [face_auth/antispoof/models/smartface_pad_artifacts/](../models/smartface_pad_artifacts/README.md)
