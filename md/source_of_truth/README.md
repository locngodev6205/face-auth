# Source of Truth — Kho Lưu Trữ Dữ Liệu Thực Nghiệm & Kết Quả Chuẩn Hóa

**Thư mục:** `SmartFace-AIoT-Based-Staff-Management-Attendance-System/face_auth/md/source_of_truth/`  
**Ngày cập nhật:** 24/10/2026  
**Mục đích:** Tập trung duy nhất toàn bộ các kết quả thực nghiệm **đã hoàn thành hoàn toàn**, các checkpoint mô hình được kiểm chứng (verified), các notebook tái lập và tài liệu tổng hợp số liệu chính thức phục vụ cho Báo cáo Đồ án, Slide bảo vệ và Bài báo khoa học.

---

## 1. Cấu Trúc Thư Mục Tối Giản & Trọng Tâm

```text
source_of_truth/
├── README.md                                  # File này: Tổng quan ma trận trạng thái & hướng dẫn quy trình
├── FULLSCALE_3SEEDS_SUMMARY_REPORT.md         # Báo cáo tổng hợp số liệu đa seed (n=3 và n=4) của Notebook 18, 21 & 22
├── FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md       # Nguồn chân lý chi tiết từng epoch, threshold trên CelebA, LCC, CASIA
├── R7_HARMFUL_GATE_ABLATION_SOURCE_OF_TRUTH.md # Báo cáo chi tiết Ablation Harmful Gate (NB 22: S100, S101, S105)
├── notebooks/                                 # Các Jupyter Notebook đã dùng để chạy
│   ├── 18_fullscale_C_P3SF_R7SC_crop15.ipynb  # Huấn luyện C, P3_SF, R7_SC
│   ├── 20_casia_fasd_train_test_C_P3SF_R7SC_100k_fullscale.ipynb # Đánh giá CASIA-FASD (Auto-discovery)
│   ├── 21_fullscale_CE_clean_baseline_crop15.ipynb # Huấn luyện CE-only clean baseline
│   ├── 22_fullscale_R7_harmful_gate_ablation_crop15.ipynb # Huấn luyện Ablation Harmful Gate
│   └── seed/                                  # Biến thể chạy từng seed riêng biệt (Đã kiểm chứng hash)
│       ├── P3_SF_FULL_crop15_s101.ipynb       # WBST full-scale Seed 101
│       ├── P3_SF_FULL_crop15_s102.ipynb       # WBST full-scale Seed 102
│       ├── R7_SC_FULL_crop15_s101.ipynb       # MAST-PAD full-scale Seed 101
│       ├── R7_SC_FULL_crop15_s102.ipynb       # MAST-PAD full-scale Seed 102
│       ├── R7_SC_FULL_crop15_s105.ipynb       # MAST-PAD full-scale Seed 105
│       ├── 20_casia_fasd_train_test_C_P3SF_R7SC_s105.ipynb # Đánh giá CASIA R7_SC Seed 105
│       ├── 21_fullscale_CE_clean_baseline_crop15_s105.ipynb # CE-only clean baseline Seed 105
│       ├── 22_fullscale_R7_harmful_gate_ablation_crop15_s100.ipynb # Ablation Harmful Gate Seed 100
│       ├── 22_fullscale_R7_harmful_gate_ablation_crop15_s101.ipynb # Ablation Harmful Gate Seed 101
│       └── 22_fullscale_R7_harmful_gate_ablation_crop15_s105.ipynb # Ablation Harmful Gate Seed 105
└── outputs/                                   # Toàn bộ Artifacts của các Run ĐÃ HOÀN THÀNH HOÀN TOÀN
    ├── notebook_18/                           # Huấn luyện đa phương pháp (C_FULL, P3_SF, R7_SC)
    │   ├── s100/                              # Seed 100: Đầy đủ 3 mô hình (C_FULL, P3_SF, R7_SC)
    │   ├── s101/                              # Seed 101: Đầy đủ 3 mô hình (C_FULL, P3_SF, R7_SC)
    │   ├── s102/                              # Seed 102: Đầy đủ 3 mô hình (C_FULL, P3_SF, R7_SC)
    │   └── s105/                              # Seed 105: R7_SC_FULL_crop15 hoàn thành (Epoch 11 + CASIA)
    ├── notebook_20/                           # Đánh giá độc lập CASIA-FASD
    │   ├── casia_eval_s100/                   # Đánh giá hoàn chỉnh Seed 100
    │   ├── casia_eval_clean_aux_s101_s102/    # Đánh giá Clean+Aux Seed 101 & 102
    │   └── casia_eval_s105/                   # Đánh giá hoàn chỉnh R7_SC Seed 105
    ├── notebook_21/                           # Huấn luyện CE-only Clean Baseline
    │   ├── s100/                              # Seed 100: CE_CLEAN_FULL_crop15 hoàn thành
    │   ├── s101/                              # Seed 101: CE_CLEAN_FULL_crop15 hoàn thành
    │   ├── s102/                              # Seed 102: CE_CLEAN_FULL_crop15 hoàn thành
    │   └── s105/                              # Seed 105: CE_CLEAN_FULL_crop15 hoàn thành (Epoch 11)
    └── notebook_22/                           # Khảo sát Ablation Harmful Gate
        ├── s100/                              # Seed 100: R7_SC_NOGATE_FULL_crop15 hoàn thành (Epoch 11)
        ├── s101/                              # Seed 101: R7_SC_NOGATE_FULL_crop15 hoàn thành (Epoch 9)
        └── s105/                              # Seed 105: R7_SC_NOGATE_FULL_crop15 hoàn thành (Epoch 10)
```

---

## 2. Ma Trận Trạng Thái Thực Nghiệm Hiện Tại (Seeds × Methods)

| Phương Pháp (Method) | Tên Chạy (Run ID) | Seed 100 | Seed 101 | Seed 102 | Seed 105 | Trạng Thái Đánh Giá Target (LCC / CASIA) |
|---|---|:---:|:---:|:---:|:---:|---|
| **1. CE-only (Baseline)** | `CE_CLEAN_FULL_crop15` (NB 21) | ✅ Hoàn thành | ✅ Hoàn thành | ✅ Hoàn thành | ✅ **Hoàn thành (Epoch 11)** | **ĐỦ 4 SEEDS:**<br>• CelebA Test: 0.9920 ± 0.0002 (ACER 4.43%)<br>• LCC Combined: 0.8387 ± 0.0055 (ACER 29.84%)<br>• CASIA Calibrated: HTER 21.20 ± 1.17% |
| **2. Clean + Aux** | `C_FULL_crop15` (NB 18) | ✅ Hoàn thành | ✅ Hoàn thành | ✅ Hoàn thành | ⏳ Tùy chọn | • CelebA Test: Đủ 3 seeds (99.08 ± 0.07%)<br>• LCC Combined: Đủ 3 seeds (85.10 ± 0.43%)<br>• CASIA Combined: Đủ 3 seeds (90.93 ± 0.99%) |
| **3. WBST (Worst-band)** | `P3_SF_FULL_crop15` (NB 18) | ✅ Hoàn thành | ✅ Hoàn thành | ✅ Hoàn thành | ⏳ Tùy chọn | • CelebA Test: Đủ 3 seeds (99.21 ± 0.05%)<br>• LCC Combined: Đủ 3 seeds (86.30 ± 1.24%)<br>• CASIA: Đủ 3 seeds |
| **4. MAST-PAD (Đề xuất)** | `R7_SC_FULL_crop15` (NB 18) | ✅ Hoàn thành | ✅ Hoàn thành | ✅ Hoàn thành | ✅ **Hoàn thành (Epoch 11)** | **ĐỦ 4 SEEDS:**<br>• S100-S102: Đủ 3 seeds CelebA + LCC + CASIA<br>• S105: CelebA (0.9905), LCC (84.98%, ACER 25.63%), CASIA Video (88.77%, Calib HTER 22.30%) |
| **5. Ablation w/o Gate** | `R7_SC_NOGATE_FULL_crop15` (NB 22) | ✅ **Hoàn thành (Epoch 11)** | ✅ **Hoàn thành (Epoch 9)** | — | ✅ **Hoàn thành (Epoch 10)** | **ĐỦ 3 SEEDS 100%:**<br>• CelebA Test: 0.9909 ± 0.0006<br>• LCC Combined: 0.8569 ± 0.0094 (ACER 26.66%)<br>• CASIA Video Combined: 0.9033 ± 0.0115 (Calib HTER 17.96%) |

---

## 3. Quy Trình Cập Nhật Khi Có Kết Quả Mới

1. **Khi chạy xong Việc 2 (Đánh giá CASIA cho Clean+Aux Seed 101 & 102):**
   * Giải nén kết quả vào: `outputs/notebook_20/casia_eval_clean_aux_s101_s102/`
   * Cập nhật bảng CASIA trong: `FULLSCALE_3SEEDS_SUMMARY_REPORT.md`
2. **Khi chạy xong MAST-PAD hoặc WBST các seed tiếp theo:**
   * Copy run vào `outputs/notebook_18/s101/<run_id>/` hoặc `s102/<run_id>/`
   * Cập nhật chỉ số vào `FULLSCALE_3SEEDS_SUMMARY_REPORT.md`
3. **Đưa toàn bộ sang nhánh `freeze_logic`:**
   * Sau khi thu thập đủ toàn bộ các seed, thực hiện:
     ```bash
     git checkout freeze_logic
     git checkout main -- face_auth/md/source_of_truth
     git add face_auth/md/source_of_truth
     git commit -m "feat(source_of_truth): sync completed fullscale outputs and reports"
     git push origin freeze_logic
     ```

- [R7_HARMFUL_GATE_ABLATION_SOURCE_OF_TRUTH.md](R7_HARMFUL_GATE_ABLATION_SOURCE_OF_TRUTH.md): Báo cáo thực nghiệm khảo sát cơ chế Harmful Gate (Notebook 22: `R7_SC_NOGATE_FULL_crop15`), đối chiếu in-domain (CelebA) và cross-domain (LCC FASD, CASIA-FASD).
