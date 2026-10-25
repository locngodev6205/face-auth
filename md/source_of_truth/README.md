# Source of Truth — Kho Lưu Trữ Dữ Liệu Thực Nghiệm & Kết Quả Chuẩn Hóa

**Thư mục:** `SmartFace-AIoT-Based-Staff-Management-Attendance-System/face_auth/md/source_of_truth/`  
**Ngày cập nhật:** 24/10/2026  
**Mục đích:** Tập trung duy nhất toàn bộ các kết quả thực nghiệm **đã hoàn thành hoàn toàn**, các checkpoint mô hình được kiểm chứng (verified), các notebook tái lập và tài liệu tổng hợp số liệu chính thức phục vụ cho Báo cáo Đồ án, Slide bảo vệ và Bài báo khoa học.

---

## 1. Cấu Trúc Thư Mục Tối Giản & Trọng Tâm

```text
source_of_truth/
├── README.md                                  # File này: Tổng quan ma trận trạng thái & hướng dẫn quy trình
├── FULLSCALE_3SEEDS_SUMMARY_REPORT.md         # Báo cáo tổng hợp số liệu 3 seed (n=3) của Notebook 18 & 21 (NƠI CẬP NHẬT KHI CÓ SEED MỚI)
├── FULLSCALE_RESULTS_SOURCE_OF_TRUTH.md       # Nguồn chân lý chi tiết từng epoch, threshold trên CelebA, LCC, CASIA
├── notebooks/                                 # Các Jupyter Notebook đã dùng để chạy
│   ├── 18_fullscale_C_P3SF_R7SC_crop15.ipynb  # Huấn luyện C, P3_SF, R7_SC
│   ├── 20_casia_fasd_train_test_C_P3SF_R7SC_100k_fullscale.ipynb # Đánh giá CASIA-FASD (Auto-discovery)
│   ├── 21_fullscale_CE_clean_baseline_crop15.ipynb # Huấn luyện CE-only clean baseline
│   └── seed/                                  # Biến thể chạy từng seed riêng biệt (Đầy đủ cả 4 notebooks)
│       ├── P3_SF_FULL_crop15_s101.ipynb       # WBST full-scale Seed 101
│       ├── P3_SF_FULL_crop15_s102.ipynb       # WBST full-scale Seed 102
│       ├── R7_SC_FULL_crop15_s101.ipynb       # MAST-PAD full-scale Seed 101
│       └── R7_SC_FULL_crop15_s102.ipynb       # MAST-PAD full-scale Seed 102
└── outputs/                                   # Toàn bộ Artifacts của các Run ĐÃ HOÀN THÀNH HOÀN TOÀN
    ├── notebook_18/                           # Huấn luyện đa phương pháp (C_FULL, P3_SF, R7_SC)
    │   ├── s100/                              # Seed 100: Đầy đủ 3 mô hình (C_FULL, P3_SF, R7_SC)
    │   ├── s101/                              # Seed 101: C_FULL_crop15 & P3_SF_FULL_crop15 hoàn thành
    │   └── s102/                              # Seed 102: C_FULL_crop15 & R7_SC_FULL_crop15 hoàn thành
    ├── notebook_20/                           # Đánh giá độc lập CASIA-FASD
    │   ├── casia_eval_s100/                   # Đánh giá hoàn chỉnh Seed 100
    │   └── casia_eval_clean_aux_s101_s102/    # Đánh giá Clean+Aux Seed 101 & 102 [Đã hoàn thành]
    └── notebook_21/                           # Huấn luyện CE-only Clean Baseline
        ├── s100/                              # Seed 100: CE_CLEAN_FULL_crop15 hoàn thành
        ├── s101/                              # Seed 101: CE_CLEAN_FULL_crop15 hoàn thành
        └── s102/                              # Seed 102: CE_CLEAN_FULL_crop15 hoàn thành
```

---

## 2. Ma Trận Trạng Thái Thực Nghiệm Hiện Tại (3 Seeds × 4 Methods)

| Phương Pháp (Method) | Tên Chạy (Run ID) | Seed 100 | Seed 101 | Seed 102 | Trạng Thái Đánh Giá Target (LCC / CASIA) |
|---|---|:---:|:---:|:---:|---|
| **1. CE-only (Baseline)** | `CE_CLEAN_FULL_crop15` (NB 21) | ✅ Hoàn thành | ✅ Hoàn thành | ✅ Hoàn thành | • CelebA Test: Đủ 3 seeds<br>• LCC Combined: Đủ 3 seeds (83.68 ± 0.51%)<br>• CASIA Combined: Đủ 3 seeds (89.57 ± 1.94%) |
| **2. Clean + Aux** | `C_FULL_crop15` (NB 18) | ✅ Hoàn thành | ✅ Hoàn thành | ✅ Hoàn thành | • CelebA Test: Đủ 3 seeds (99.08 ± 0.07%)<br>• LCC Combined: Đủ 3 seeds (85.10 ± 0.43%)<br>• CASIA Combined: **Đủ 3 seeds (90.93 ± 0.99%) [Mới bổ sung]** |
| **3. WBST (Worst-band)** | `P3_SF_FULL_crop15` (NB 18) | ✅ Hoàn thành | ✅ Hoàn thành (Epoch 9) | ⚠️ Gián đoạn Epoch 7 | • S100: CelebA (0.9922), LCC (86.08%), CASIA (90.49%)<br>• S101: CelebA (0.9925), LCC (87.64%) [Mới bổ sung]<br>• S102: Cần chạy lại khi có quota |
| **4. MAST-PAD (Đề xuất)** | `R7_SC_FULL_crop15` (NB 18) | ✅ Hoàn thành | ⏳ Đang chạy / Chờ GPU | ✅ Hoàn thành (Epoch 9) | • S100: CelebA (0.9923), LCC (86.03%), CASIA (92.41%)<br>• S102: CelebA (0.9909), LCC (85.64%) [Mới bổ sung]<br>• S101: Cần chạy tiếp khi có quota |

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
