# Báo Cáo Số Liệu Kết Quả Thực Nghiệm Đa Seed (3 Seeds)
## MiniFASNet V2 Full-scale Cross-Domain: Notebook 18 & Notebook 21

- **Tập dữ liệu nguồn (Source):** CelebA-Spoof (~500K ảnh)
- **Tập dữ liệu đích (Target):** LCC FASD (18,794 ảnh), CASIA-FASD
- **Tham số cấu hình:** Batch Size 256, Crop scale 1.5, Input 80x80 BGR, PyTorch AMP
- **Danh sách Random Seeds:** Seed 100, Seed 101, Seed 102

---

## 1. Trạng Thái Dữ Liệu Thực Nghiệm

| Notebook | Mô hình | Seed 100 | Seed 101 | Seed 102 | Tập Đánh Giá Đã Hoàn Thành |
|---|---|:---:|:---:|:---:|---|
| **Notebook 18** | **C_FULL_crop15** (Clean CE) | Hoàn thành | Hoàn thành | Hoàn thành | CelebA Test, LCC (Train, Dev, Eval, Combined) |
| **Notebook 18** | **P3_SF_FULL_crop15** (Worst-band CE) | Hoàn thành | Hoàn thành (Epoch 9) | Dừng epoch 7 | CelebA Test, LCC Combined (Seed 100, 101) |
| **Notebook 18** | **R7_SC_FULL_crop15** (Harmful-band CE) | Hoàn thành | Chưa chạy | Chưa chạy | CelebA Test, LCC Combined (Seed 100) |
| **Notebook 21** | **CE_CLEAN_FULL_crop15** (Clean Baseline) | Hoàn thành | Hoàn thành | Hoàn thành | CelebA Test, LCC (4 splits), CASIA-FASD (3 splits) |

---

## 2. Kết Quả Notebook 21 (`21_fullscale_CE_clean_baseline_crop15`)

### 2.1. CelebA-Spoof Test (Test-as-Val)
*Quy mô tập test: 66,379 ảnh (19,864 Real, 46,515 Spoof).*

| Seed | Epoch Tối Ưu | Ngưỡng Cắt (Logit) | AUC | EER (%) | ACER (%) | APCER (%) | BPCER (%) | TPR @ FPR=1% (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S100** | 9 | 4.5967 | 0.992144 | 4.476% | 4.450% | 5.074% | 3.826% | 85.45% |
| **S101** | 9 | 4.6982 | 0.992027 | 4.463% | 4.430% | 5.099% | 3.761% | 85.39% |
| **S102** | 12 | 4.9775 | 0.991719 | 4.451% | 4.436% | 4.850% | 4.022% | 84.61% |
| **Mean ± Std** | — | — | **0.9920 ± 0.0002** | **4.463% ± 0.010%** | **4.439% ± 0.008%** | **5.008% ± 0.112%** | **3.870% ± 0.111%** | **85.15% ± 0.38%** |

### 2.2. LCC FASD (Source-Calibrated Frozen Threshold)
*Ngưỡng đóng băng trực tiếp từ CelebA-Spoof.*

| Phân Tập LCC | Số Mẫu | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LCC Training** | 8,283 | 0.8703 | 0.8557 | 0.8545 | **0.8601 ± 0.0072** | 22.97% ± 0.31% | 29.23% ± 0.87% | 23.01% ± 1.71% |
| **LCC Development** | 2,947 | 0.8176 | 0.8173 | 0.8277 | **0.8209 ± 0.0048** | 25.82% ± 1.73% | 31.95% ± 0.77% | 18.60% ± 1.63% |
| **LCC Evaluation** | 7,564 | 0.8127 | 0.8139 | 0.8057 | **0.8107 ± 0.0036** | 26.27% ± 0.80% | 30.29% ± 1.14% | 17.36% ± 3.54% |
| **LCC Combined** | 18,794 | 0.8440 | 0.8341 | 0.8324 | **0.8368 ± 0.0051** | 24.45% ± 0.23% | 30.16% ± 0.82% | 19.38% ± 1.72% |

### 2.3. CASIA-FASD
*So sánh giữa Zero-Shot (ngưỡng đóng băng CelebA) và Domain Calibrated (ngưỡng fit trên CASIA Train).*

| Phân Tập CASIA | Chính Sách Ngưỡng | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CASIA Train** | Zero-Shot (CelebA Frozen) | 0.9047 | 0.8822 | 0.8986 | **0.8952 ± 0.0095** | 19.92% ± 0.70% | 31.52% ± 0.17% | 30.69% ± 5.76% |
| **CASIA Train** | Domain Calibrated (CASIA-Fit) | 0.9124 | 0.8875 | 0.9103 | **0.9034 ± 0.0113** | 19.72% ± 0.61% | 17.54% ± 0.74% | 37.29% ± 5.54% |
| **CASIA Test** | Zero-Shot (CelebA Frozen) | 0.8621 | 0.8389 | 0.8761 | **0.8590 ± 0.0153** | 22.76% ± 1.60% | 34.04% ± 1.78% | 27.69% ± 3.96% |
| **CASIA Test** | Domain Calibrated (CASIA-Fit) | 0.8837 | 0.8623 | 0.9014 | **0.8825 ± 0.0160** | 20.96% ± 2.75% | 22.16% ± 1.34% | 37.50% ± 0.00% |
| **CASIA Combined** | Zero-Shot (CelebA Frozen) | 0.8743 | 0.8503 | 0.8844 | **0.8697 ± 0.0143** | 21.61% ± 1.45% | 32.66% ± 1.55% | 30.27% ± 4.48% |
| **CASIA Combined** | Domain Calibrated (CASIA-Fit) | 0.8983 | 0.8751 | 0.9137 | **0.8957 ± 0.0159** | 19.89% ± 2.66% | 21.45% ± 1.26% | 40.00% ± 1.81% |

---

## 3. Kết Quả Notebook 18 (`18_fullscale_C_P3SF_R7SC_crop15`)

### 3.1. `C_FULL_crop15` qua 3 Seeds (100, 101, 102)

| Tập Dữ Liệu / Phân Tập | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CelebA Test** | 0.9907 | 0.9900 | 0.9917 | **0.9908 ± 0.0007** | 4.70% ± 0.16% | 4.66% ± 0.18% | 82.94% ± 1.99% |
| **LCC Training** | 0.8692 | 0.8745 | 0.8664 | **0.8701 ± 0.0033** | 21.78% ± 0.48% | 27.93% ± 0.45% | 23.63% ± 6.03% |
| **LCC Development** | 0.8516 | 0.8429 | 0.8472 | **0.8472 ± 0.0036** | 23.53% ± 0.68% | 26.15% ± 0.41% | 20.12% ± 1.57% |
| **LCC Evaluation** | 0.8116 | 0.8225 | 0.8288 | **0.8210 ± 0.0071** | 25.15% ± 1.15% | 29.38% ± 1.32% | 15.61% ± 4.98% |
| **LCC Combined** | 0.8460 | 0.8540 | 0.8529 | **0.8510 ± 0.0035** | 23.30% ± 0.20% | 27.92% ± 0.55% | 20.08% ± 4.46% |

### 3.2. So Sánh Mô Hình C, P3_SF, R7_SC (Seed 100)

| Mô Hình | Phương Pháp Huấn Luyện | CelebA Test AUC | CelebA Test ACER | LCC Eval AUC | LCC Combined AUC | LCC Combined ACER | LCC TPR@1% |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C_FULL_crop15** | Clean Baseline (Cross-Entropy) | 0.9907 | 4.62% | 0.8116 | 0.8460 | 28.05% | 14.01% |
| **P3_SF_FULL_crop15** | Worst-band CE | 0.9922 | 4.35% | 0.8197 | 0.8608 | 26.79% | 22.57% |
| **R7_SC_FULL_crop15** | Harmful-band CE | 0.9923 | 4.35% | 0.8401 | 0.8603 | 27.12% | 19.99% |

### 3.3. `P3_SF_FULL_crop15` (WBST) qua 2 Seeds (100, 101)

| Phân Tập Đánh Giá | Seed 100 AUC | Seed 101 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CelebA Test** | 0.9922 | 0.9925 | **0.9924 ± 0.0002** | **4.43% ± 0.05%** | **4.40% ± 0.06%** | **86.91% ± 0.60%** |
| **LCC Training** | 0.8813 | 0.8928 | **0.8870 ± 0.0081** | **19.90% ± 0.08%** | **25.90% ± 1.04%** | **27.44% ± 1.94%** |
| **LCC Development** | 0.8528 | 0.8641 | **0.8585 ± 0.0080** | **22.33% ± 0.20%** | **24.63% ± 1.78%** | **21.11% ± 7.16%** |
| **LCC Evaluation** | 0.8197 | 0.8500 | **0.8348 ± 0.0214** | **23.61% ± 1.72%** | **26.43% ± 1.28%** | **15.76% ± 0.67%** |
| **LCC Combined** | 0.8608 | 0.8764 | **0.8686 ± 0.0110** | **21.26% ± 0.76%** | **25.82% ± 1.37%** | **22.21% ± 0.51%** |

---

## 4. Đối Chiếu Kết Quả Giữa Notebook 18 và Notebook 21 (Baseline CE)

| Tiêu Chí Đánh Giá | Notebook 18 (`C_FULL`) Mean ± Std | Notebook 21 (`CE_CLEAN_FULL`) Mean ± Std | Chênh Lệch (|NB18 - NB21|) |
|---|:---:|:---:|:---:|
| **CelebA Test AUC** | 0.9908 ± 0.0007 | 0.9920 ± 0.0002 | 0.0012 |
| **CelebA Test ACER** | 4.66% ± 0.18% | 4.44% ± 0.01% | 0.22% |
| **LCC Combined AUC** | 0.8510 ± 0.0035 | 0.8368 ± 0.0051 | 0.0142 |
| **LCC Combined ACER** | 27.92% ± 0.55% | 30.16% ± 0.82% | 2.24% |

---

## 5. Bảng Đối Chiếu Số Liệu Với Các Nghiên Cứu Đã Công Bố

| Mô Hình | Tập Huấn Luyện | CelebA Test AUC | CelebA ACER (%) | LCC Combined AUC | LCC ACER (%) | Số Tham Số (Params) | FLOPs (GFLOPs) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AENET** (CVPR 2020) | CelebA-Spoof | 0.9990 | 3.25% | 0.8680 | 22.61% | 11.22 M | 3.640 G |
| **MobileNetV3 Large** | CelebA-Spoof | 0.9980 | 3.80% | 0.9210 | 16.33% | 3.02 M | 0.150 G |
| **MobileNetV3 Small** | CelebA-Spoof | 0.9940 | 5.05% | 0.8890 | 19.69% | 1.00 M | 0.040 G |
| **MobileNetV3 Small 0.75** | CelebA-Spoof | 0.9910 | 6.09% | 0.8790 | 21.04% | 0.60 M | 0.030 G |
| **Dự án: CE Baseline (3 Seeds)** | CelebA-Spoof | 0.9920 ± 0.0002 | 4.44% ± 0.01% | 0.8368 ± 0.0051 | 30.16% ± 0.82% | 0.43 M (~1.7 MB) | 0.081 G |
| **Dự án: P3_SF (2 Seeds: 100, 101)** | CelebA-Spoof | 0.9924 ± 0.0002 | 4.40% ± 0.06% | 0.8686 ± 0.0110 | 25.82% ± 1.37% | 0.43 M (~1.7 MB) | 0.081 G |
| **Dự án: R7_SC (Seed 100)** | CelebA-Spoof | 0.9923 | 4.35% | 0.8603 | 27.12% | 0.43 M (~1.7 MB) | 0.081 G |
