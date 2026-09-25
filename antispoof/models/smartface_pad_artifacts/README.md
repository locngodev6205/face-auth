# SmartFace PAD Artifacts Archive (Cơ sở Lưu trữ Dài hạn)

Thư mục này được tổ chức theo đúng quy chuẩn **Section 3 trong [E1_artifacts_for_E2_kaggle_guide.md](../../notebooks/05_e2_frequency_dct/E1_artifacts_for_E2_kaggle_guide.md)** và **Section 16 trong [E1_E2_heldout_test_kaggle_guide.md](../../notebooks/05_e2_frequency_dct/E1_E2_heldout_test_kaggle_guide.md)** nhằm phục vụ:
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
│   │   └── mnv3_e1_preliminary_v5_1_best.onnx.data
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
├── E2_dct_v1/                                         <-- [E2 Frequency-only: Sẵn sàng nhận output]
│   ├── pytorch/                                       <-- Lưu frequency_branch_best.pth
│   ├── deployment/                                    <-- Lưu e2_*.onnx & runtime_config.json
│   ├── metadata/                                      <-- Lưu e2 run config & history
│   └── test/                                          <-- Lưu e2 heldout test ZIP & CSVs
│
├── minifasnet_v2se/                                   <-- [Mô hình kế thừa MiniFASNet] INT8 (128x128, 2-class)
│   └── deployment/
│       └── best_model_quantized.onnx                  <-- Siêu nhẹ ~600 KB
│
└── mobilenet_v4/                                      <-- [Mô hình thử nghiệm nâng cao] MobileNetV4 (224x224)
    └── deployment/
        └── mnv4_best_224.onnx                         <-- Model ONNX (~5.0 MB)
```
