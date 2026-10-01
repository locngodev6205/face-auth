# SmartFace PAD Artifacts Archive (Cơ sở Lưu trữ Dài hạn)

Thư mục này được tổ chức theo đúng quy chuẩn **Section 3 trong [E1_artifacts_for_E2_kaggle_guide.md](../../notebooks/e2_frequency_dct/E1_artifacts_for_E2_kaggle_guide.md)** và **Section 16 trong [E1_E2_heldout_test_kaggle_guide.md](../../notebooks/e2_frequency_dct/E1_E2_heldout_test_kaggle_guide.md)** nhằm phục vụ:
1. Nghiên cứu khoa học có tính tái lập (Reproducibility).
2. Bảo toàn tính công bằng của Ablation Study (Fair Comparison Contract giữa E1, E2, E3).
3. Đóng gói triển khai sản phẩm thực tế (Edge Deployment).

---

## Cấu trúc tổng thể các thực nghiệm (PAD Experiments)

```text
smartface_pad_artifacts/
├── README.md                                          <-- Tài liệu hướng dẫn này
│
├── E1_v0_mnv3_large/                                  <-- [E1 v0 Baseline ban đầu] MobileNetV3-Large (224px, 3-class)
│   ├── deployment/
│   │   └── mnv3_large_3class_best.onnx                <-- Model ONNX (~16.8 MB)
│   ├── pytorch/
│   │   ├── best_mnv3_large_3class.pth                 <-- Checkpoint PyTorch (~17 MB)
│   │   └── checkpoint_last.pth                        <-- Last checkpoint (~50 MB)
│   ├── metadata/
│   │   ├── training_history.png                       <-- Biểu đồ Loss & Accuracy
│   │   └── plots/
│   │       └── __results___13_2.png
│   └── test/                                          <-- [ĐÃ HOÀN TẤT] Kết quả Quick Test 10.000 mẫu sơ bộ
│       ├── quick_test_summary.json                    <-- Test AUC: 97.34%, ACER: 4.13%, Latency: 17.53 ms/ảnh
│       ├── quick_test_evaluation_plots.png            <-- Biểu đồ ROC, phân phối logit diff và lỗi
│       ├── threshold_sweep_results.csv                <-- Quét ngưỡng logit diff và chi tiết lỗi
│       └── worst_failure_cases.png                    <-- Các ca dự đoán sai lệch nghiêm trọng nhất
│
├── E1_v5_1_mnv3/                                      <-- [E1 v5.1 Preliminary] MobileNetV3 (224px, 100k samples)
│   ├── data_protocol/                                 <-- Split manifest seed 42
│   │   └── celeba_spoof_subject_disjoint_preliminary_split_seed42.npz
│   ├── pytorch/                                       <-- Trọng số PyTorch (.pth)
│   │   ├── mnv3_e1_preliminary_v5_1_best.pth
│   │   └── mnv3_e1_preliminary_v5_1_checkpoint_last.pth
│   ├── deployment/                                    <-- ONNX Export
│   │   ├── mnv3_e1_preliminary_v5_1_best.onnx
│   │   ├── mnv3_e1_preliminary_v5_1_best.onnx.data
│   │   └── mnv3_e1_preliminary_v5_1_runtime_config.json
│   ├── metadata/                                      <-- Tham số, lịch sử huấn luyện & đồ thị
│   │   ├── mnv3_e1_preliminary_v5_1_best_meta.json
│   │   ├── mnv3_e1_preliminary_v5_1_run_config.json
│   │   ├── mnv3_e1_preliminary_v5_1_training_history.json
│   │   ├── mnv3_e1_preliminary_v5_1_training_history.png
│   │   └── plots/
│   └── test/                                          <-- [ĐÃ HOÀN TẤT] Kết quả Deep Eval & Test 10.000 mẫu
│       ├── pad_deep_evaluation_summary.json           <-- Val AUC: 99.99%, Test AUC: 99.31%, BPCER: 0.27%, ACER: 8.49%
│       ├── test_predictions_detailed.csv              <-- Dự đoán chi tiết từng ảnh test
│       ├── validation_predictions_detailed.csv        <-- Dự đoán chi tiết tập val
│       ├── test_bootstrap_ci.csv                      <-- Khoảng tin cậy Bootstrap 95%
│       ├── test_per_attack_breakdown.csv              <-- Phân rã lỗi theo từng loại spoof
│       ├── test_subject_breakdown.csv                 <-- Phân rã lỗi theo ID đối tượng
│       ├── plots/                                     <-- 16 đồ thị ROC, DET, Confusion Matrix, Error Galleries
│       └── cache/                                     <-- Cache prediction CSVs
│
├── E1_v5_3_mnv3_small/                                <-- [E1 v5.3 Edge] MobileNetV3-Small (Runtime Active)
│   ├── data_protocol/                                 <-- BẮT BUỘC: Đóng băng dữ liệu cho E2 reuse
│   │   ├── celeba_scrfd_bbox_cache_v5_3_preliminary_seed42.json
│   │   ├── celeba_spoof_preliminary_v5_3_edge_mnv3_small_seed42.npz
│   │   └── celeba_spoof_preliminary_v5_3_preprocess_audit.json
│   ├── pytorch/                                       <-- Checkpoint PyTorch (.pth) phục vụ E3
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_best.pth
│   │   └── mnv3s_e1_preliminary_v5_3_edge_checkpoint_last.pth
│   ├── deployment/                                    <-- Model ONNX & Runtime config
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_best.onnx
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_best.onnx.data
│   │   └── mnv3s_e1_preliminary_v5_3_edge_runtime_config.json
│   ├── metadata/                                      <-- Log huấn luyện & Biểu đồ
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_best_meta.json
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_run_config.json
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_training_history.json
│   │   ├── mnv3s_e1_preliminary_v5_3_edge_training_history.png
│   │   └── plots/
│   └── test/                                          <-- [ĐÃ HOÀN TẤT] Kết quả Held-out Test 10.000 mẫu
│       ├── e1_heldout_test_summary.json               <-- AUC: 98.14%, BPCER: 0.40%, ACER: 9.66%, Latency: 2.47ms
│       ├── e1_test_predictions.csv                    <-- Chi tiết dự đoán từng ảnh
│       ├── e1_attack_breakdown.csv                    <-- Phân rã lỗi theo từng loại spoof
│       ├── e1_binary_confusion_matrix.png             <-- Ma trận nhầm lẫn 180 DPI
│       ├── e1_score_distribution.png                  <-- Biểu đồ phân phối điểm số
│       └── e1_preliminary_heldout_test_results.zip    <-- Gói ZIP lưu trữ toàn bộ
│
├── E2_dct_v1/                                         <-- [E2 Frequency-only 2D-DCT] Huấn luyện hoàn tất (17K tham số)
│   ├── deployment/                                    <-- Model ONNX & Runtime config
│   │   ├── e2_freq_only_preliminary_dct_v1_best.onnx
│   │   ├── e2_freq_only_preliminary_dct_v1_best.onnx.data
│   │   └── e2_freq_only_preliminary_dct_v1_runtime_config.json
│   ├── pytorch/                                       <-- Trọng số PyTorch & nhánh tần số cho E3
│   │   ├── e2_freq_only_preliminary_dct_v1_best.pth
│   │   ├── e2_freq_only_preliminary_dct_v1_checkpoint_last.pth
│   │   ├── e2_freq_only_preliminary_dct_v1_frequency_branch_best.pth   <-- Trích xuất nhánh tần số (8.48K params)
│   │   └── e2_freq_only_preliminary_dct_v1_frequency_branch_spec.json
│   ├── metadata/                                      <-- Log huấn luyện & Biểu đồ
│   │   ├── e2_freq_only_preliminary_dct_v1_best_meta.json  <-- Val AUC: 95.88%, ACER: 9.70%, Epoch 20
│   │   ├── e2_freq_only_preliminary_dct_v1_run_config.json
│   │   ├── e2_freq_only_preliminary_dct_v1_training_history.json
│   │   ├── e2_freq_only_preliminary_dct_v1_training_history.png
│   │   └── plots/
│   │       └── __results___13_46.png
│   └── test/                                          <-- [ĐÃ HOÀN TẤT] Kết quả Held-out Test 10.000 mẫu
│       ├── e2_heldout_test_summary.json               <-- AUC: 83.66%, ACER: 25.25%, Latency: 0.57ms/ảnh
│       ├── e2_test_predictions.csv                    <-- Chi tiết dự đoán từng ảnh test
│       ├── e2_attack_breakdown.csv                    <-- Phân rã lỗi theo từng loại spoof
│       ├── e2_binary_confusion_matrix.png             <-- Ma trận nhầm lẫn 180 DPI
│       ├── e2_score_distribution.png                  <-- Biểu đồ phân phối điểm số
│       ├── e2_test_protocol_snapshot.json             <-- Lưu snapshot cấu hình protocol test
│       └── e2_preliminary_heldout_test_results.zip    <-- Gói ZIP lưu trữ toàn bộ
│
├── E3_concat_v1/                                      <-- [E3 Concat Fusion] MobileNetV3-Small + 2D-DCT Branch (1.13M tham số)
│   ├── deployment/                                    <-- Model ONNX & Runtime config
│   │   ├── e3_concat_preliminary_v1_best.onnx         <-- Model ONNX (~4.5 MB)
│   │   └── e3_concat_preliminary_v1_runtime_config.json
│   ├── pytorch/                                       <-- Trọng số PyTorch (.pth)
│   │   ├── e3_concat_preliminary_v1_best.pth          <-- Best PyTorch checkpoint (~4.6 MB)
│   │   └── e3_concat_preliminary_v1_checkpoint_last.pth
│   ├── metadata/                                      <-- Log huấn luyện, cấu hình & benchmark
│   │   ├── e3_concat_preliminary_v1_best_meta.json    <-- Val AUC: 99.97%, ACER: 0.35%, Epoch 12
│   │   ├── e3_concat_preliminary_v1_run_config.json
│   │   ├── e3_concat_preliminary_v1_training_history.json
│   │   ├── e3_concat_preliminary_v1_training_history.png
│   │   └── e3_concat_preliminary_v1_efficiency.json   <-- Benchmarked CPU latency (Model: ~2.21ms, Total PAD: ~8.85ms)
│   └── test/                                          <-- [ĐÃ HOÀN TẤT] Kết quả Held-out Test 10.000 mẫu CelebA
│       ├── e3_heldout_test_summary.json               <-- AUC: 98.57%, ACER: 9.13%, BPCER: 0.51%
│       ├── e3_test_predictions.csv                    <-- Chi tiết dự đoán 10.000 mẫu
│       ├── e3_attack_breakdown.csv                    <-- Phân rã lỗi theo loại spoof
│       ├── e3_face_size_breakdown.csv                 <-- Phân rã lỗi theo kích thước khuôn mặt
│       ├── e3_binary_confusion_matrix.png             <-- Ma trận nhầm lẫn 180 DPI
│       ├── e3_score_distribution.png                  <-- Phân phối điểm d (Real vs Spoof)
│       ├── e3_roc.png                                 <-- Đường cong ROC kiểm thử
│       └── e3_test_protocol_snapshot.json             <-- Snapshot cấu hình protocol test
│
├── cross_dataset_v1/                                  <-- [Thực nghiệm Ngoài Miền - Zero-Shot Benchmark] E1 vs E3
│   └── preliminary/
│       ├── manifests/                                 <-- Danh mục mẫu ngoại miền (LCC-FASD 3.762 ảnh)
│       │   ├── lcc_manifest.csv
│       │   ├── evaluation_unit_manifest.csv
│       │   ├── external_dataset_manifest.csv
│       │   ├── external_dataset_manifest.json
│       │   └── sampled_frame_manifest.csv
│       ├── detector_cache/                            <-- Cache phát hiện khuôn mặt SCRFD-500M
│       │   └── external_scrfd_cache.json
│       ├── predictions/                               <-- Kết quả so sánh đối đầu E1 vs E3
│       │   ├── cross_dataset_model_comparison.csv     <-- Bảng so sánh trực tiếp chỉ số E1 vs E3
│       │   ├── cross_dataset_summary.json             <-- E3 Accuracy 65.58% (E1: 61.40%), APCER 35.34% (E1: 40.44%)
│       │   ├── cross_dataset_face_size_breakdown.csv
│       │   ├── lcc_fasd_image_predictions.csv
│       │   └── unit_coverage_and_predictions.csv
│       ├── plots/                                     <-- Biểu đồ ROC & phân phối điểm ngoài miền
│       │   ├── roc_lcc_fasd.png
│       │   ├── score_distribution_lcc_fasd.png
│       │   ├── lcc_fasd_E1_roc.png & lcc_fasd_E1_score_distribution.png
│       │   └── lcc_fasd_E3_roc.png & lcc_fasd_E3_score_distribution.png
│       └── cross_dataset_preliminary_v1_results.zip   <-- Gói ZIP nén toàn bộ
│
├── minifasnet_v2se/                                   <-- [Mô hình kế thừa MiniFASNet] INT8 (128x128, 2-class)
│   └── deployment/
│       └── best_model_quantized.onnx                  <-- Siêu nhẹ ~600 KB
│
└── mobilenet_v4/                                      <-- [Mô hình thử nghiệm nâng cao] MobileNetV4 (224x224)
    └── deployment/
        └── mnv4_best_224.onnx                         <-- Model ONNX (~5.0 MB)
```

---

## 📊 Bảng Thống kê Số lượng Trọng số & Kích thước Mô hình (Model Parameter Benchmark)

Bảng tổng hợp thông số kỹ thuật được trích xuất và đo lường trực tiếp từ các file nhị phân (`.onnx`, `.pth`) và metadata huấn luyện:

| Phân hệ / Thư mục | Tên file mô hình chính | Kiến trúc Backbone | Kích thước đầu vào | **Số lượng tham số (Parameters)** | Dung lượng file | Vai trò trong hệ thống |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **[`E2_dct_v1`](E2_dct_v1/)** | `e2_freq_only_...best.onnx` | Tiny Frequency CNN (2D-DCT) | $1 \times 224 \times 224$ | **17,187** <br>*(Nhánh tần số: 8,480)* | **~105 KB** | Nghiên cứu miền Tần số (Ablation E2 & Module cho E3) |
| **[`minifasnet_v2se`](minifasnet_v2se/)** | `best_model_quantized.onnx` | MiniFASNetV2SE (INT8) | $3 \times 128 \times 128$ | **468,769** *(~0.47M)* | **~600 KB** | Mô hình siêu nhẹ kế thừa (Legacy baseline 2-class) |
| **[`E1_v5_3_mnv3_small`](E1_v5_3_mnv3_small/)** | `mnv3s_e1_...edge_best.onnx` | **MobileNetV3-Small** | $3 \times 224 \times 224$ | **1,108,515** *(~1.11M)* | **~4.4 MB** | ⭐ **Mô hình không gian chính thức (Active Spatial Edge)** |
| **[`E3_concat_v1`](E3_concat_v1/)** | `e3_concat_preliminary_v1_best.onnx` | **MobileNetV3-Small + 2D-DCT** | Spatial ($3 \times 224 \times 224$) + Freq ($1 \times 224 \times 224$) | **1,125,187** *(~1.13M)* | **~4.5 MB** | 🚀 **Mô hình Fusion Đa Miền (Không gian + Tần số)** |
| **[`mobilenet_v4`](mobilenet_v4/)** | `mnv4_best_224.onnx` | MobileNetV4-Conv-Small | $3 \times 224 \times 224$ | **1,253,283** *(~1.25M)* | **~4.8 MB** | Thực nghiệm kiến trúc thế hệ mới |
| **[`E1_v5_1_mnv3`](E1_v5_1_mnv3/)** | `mnv3_e1_...v5_1_best.onnx` | MobileNetV3-Large (Modified Head) | $3 \times 224 \times 224$ | **3,276,722** *(~3.28M)* | **~12.6 MB** | Đánh giá chuyên sâu 100K mẫu (E1 Preliminary) |
| **[`E1_v0_mnv3_large`](E1_v0_mnv3_large/)** | `mnv3_large_3class_best.onnx` | MobileNetV3-Large (Full Classifier) | $3 \times 224 \times 224$ | **4,230,321** *(~4.23M)* | **~16.2 MB** | Mô hình Spatial Baseline ban đầu (10K Quick Test) |

### 📌 Điểm nhấn phân tích kiến trúc:
1. **Tối ưu hóa Edge:** Mô hình sản xuất [E1_v5_3_mnv3_small](E1_v5_3_mnv3_small/) giảm **~74%** số tham số so với MobileNetV3-Large (từ `4.23M` xuống `1.11M`), đưa độ trễ suy luận xuống mức cực thấp **~2.47 ms/ảnh** mà vẫn duy trì AUC $98.14\%$.
2. **Miền tần số siêu gọn:** Mô hình [E2_dct_v1](E2_dct_v1/) chỉ tốn **17,187 tham số** (nhánh trích xuất `frequency_branch` chỉ **8,480 tham số**), chứng minh đặc trưng 2D-DCT nén thông tin giả mạo rất cô đọng mà không cần backbone tích chập sâu.
3. **Hiệu quả của E3 Concat:** Mô hình [E3_concat_v1](E3_concat_v1/) kết hợp nhánh Spatial (MobileNetV3-Small) và nhánh Frequency (2D-DCT) chỉ làm tăng tổng tham số thêm **~16.6K** (từ `1.11M` lên `1.13M`, tăng chỉ ~1.5%), trong khi thời gian suy luận riêng của mô hình (model-only latency) gần như tương đương (~2.21 ms vs ~2.03 ms trên CPU).

