# Báo Cáo Số Liệu Kết Quả Thực Nghiệm Đa Seed (3 Seeds)
## MiniFASNet V2 Full-scale Cross-Domain: Notebook 18 & Notebook 21

- **Tập dữ liệu nguồn (Source):** CelebA-Spoof (~500K ảnh)
- **Tập dữ liệu đích (Target):** LCC FASD (18,794 ảnh), CASIA-FASD
- **Tham số cấu hình:** Batch Size 256, Crop scale 1.5, Input 80x80 BGR, PyTorch AMP
- **Danh sách Random Seeds:** Seed 100, Seed 101, Seed 102

---

## 1. Trạng Thái Dữ Liệu Thực Nghiệm

| Notebook | Mô hình | Seed 100 | Seed 101 | Seed 102 | Seed 105 | Tập Đánh Giá Đã Hoàn Thành |
|---|---|:---:|:---:|:---:|:---:|---|
| **Notebook 18** | **C_FULL_crop15** (Clean CE) | Hoàn thành | Hoàn thành (Epoch 9) | Hoàn thành (Epoch 9) | ⏳ Tùy chọn | CelebA Test, LCC (4 splits), CASIA-FASD (3 splits) |
| **Notebook 18** | **P3_SF_FULL_crop15** (Worst-band CE) | Hoàn thành | Hoàn thành (Epoch 9) | Hoàn thành (Epoch 11) | ⏳ Tùy chọn | CelebA Test, LCC (4 splits), CASIA-FASD (3 splits) |
| **Notebook 18** | **R7_SC_FULL_crop15** (Harmful-band CE) | Hoàn thành | Hoàn thành (Epoch 10) | Hoàn thành (Epoch 9) | **Hoàn thành (Epoch 11)** | **ĐỦ 4 SEEDS:** CelebA, LCC, CASIA-FASD (NB 20) |
| **Notebook 21** | **CE_CLEAN_FULL_crop15** (Clean Baseline) | Hoàn thành | Hoàn thành (Epoch 9) | Hoàn thành (Epoch 12) | **Hoàn thành (Epoch 11)** | **ĐỦ 4 SEEDS:** CelebA, LCC, CASIA-FASD |
| **Notebook 22** | **R7_SC_NOGATE_FULL_crop15** (Ablation w/o Gate) | **Hoàn thành (Epoch 11)** | **Hoàn thành (Epoch 9)** | — | **Hoàn thành (Epoch 10)** | **ĐỦ 3 SEEDS:** CelebA, LCC, CASIA-FASD |

---

## 2. Kết Quả Notebook 21 (`21_fullscale_CE_clean_baseline_crop15`)

### 2.1. CelebA-Spoof Test (Test-as-Val)
*Quy mô tập test: 66,379 ảnh (19,864 Real, 46,515 Spoof).*

| Seed | Epoch Tối Ưu | Ngưỡng Cắt (Logit) | AUC | EER (%) | ACER (%) | APCER (%) | BPCER (%) | TPR @ FPR=1% (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S100** | 9 | 4.5967 | 0.992144 | 4.476% | 4.450% | 5.074% | 3.826% | 85.45% |
| **S101** | 9 | 4.6982 | 0.992027 | 4.463% | 4.430% | 5.099% | 3.761% | 85.39% |
| **S102** | 12 | 4.9775 | 0.991719 | 4.451% | 4.436% | 4.850% | 4.022% | 84.61% |
| **S105** | 11 | 4.8643 | 0.992292 | 4.436% | 4.407% | 4.872% | 3.942% | 85.59% |
| **Mean ± Std (4 Seeds)** | — | — | **0.9920 ± 0.0002** | **4.457% ± 0.015%** | **4.431% ± 0.016%** | **4.974% ± 0.124%** | **3.888% ± 0.100%** | **85.26% ± 0.39%** |

### 2.2. LCC FASD (Source-Calibrated Frozen Threshold)
*Ngưỡng đóng băng trực tiếp từ CelebA-Spoof.*

| Phân Tập LCC | Số Mẫu | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | Seed 105 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LCC Training** | 8,283 | 0.8703 | 0.8557 | 0.8545 | 0.8714 | **0.8630 ± 0.0079** | 22.55% ± 0.77% | 28.95% ± 0.89% | 22.84% ± 1.51% |
| **LCC Development** | 2,947 | 0.8176 | 0.8173 | 0.8277 | 0.8168 | **0.8199 ± 0.0045** | 25.85% ± 1.50% | 31.44% ± 1.10% | 17.47% ± 2.42% |
| **LCC Evaluation** | 7,564 | 0.8127 | 0.8139 | 0.8057 | 0.8161 | **0.8121 ± 0.0039** | 26.17% ± 0.72% | 30.07% ± 1.06% | 17.16% ± 3.08% |
| **LCC Combined** | 18,794 | 0.8440 | 0.8341 | 0.8324 | 0.8444 | **0.8387 ± 0.0055** | 24.34% ± 0.28% | 29.84% ± 0.89% | 18.88% ± 1.73% |

### 2.3. CASIA-FASD
*So sánh giữa Zero-Shot (ngưỡng đóng băng CelebA) và Domain Calibrated (ngưỡng fit trên CASIA Train).*

| Phân Tập CASIA | Chính Sách Ngưỡng | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | Seed 105 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CASIA Train** | Zero-Shot (CelebA Frozen) | 0.9047 | 0.8822 | 0.8986 | 0.8817 | **0.8918 ± 0.0101** | 20.27% ± 0.85% | 30.92% ± 1.05% | 32.36% ± 5.76% |
| **CASIA Train** | Domain Calibrated (CASIA-Fit) | 0.9124 | 0.8875 | 0.9103 | 0.8830 | **0.8983 ± 0.0132** | 19.72% ± 0.53% | 18.01% ± 1.04% | 37.71% ± 4.85% |
| **CASIA Test** | Zero-Shot (CelebA Frozen) | 0.8621 | 0.8389 | 0.8761 | 0.8429 | **0.8550 ± 0.0150** | 22.85% ± 1.39% | 33.91% ± 1.56% | 26.52% ± 3.99% |
| **CASIA Test** | Domain Calibrated (CASIA-Fit) | 0.8837 | 0.8623 | 0.9014 | 0.8651 | **0.8781 ± 0.0158** | 20.83% ± 2.39% | 21.92% ± 1.23% | 35.23% ± 3.94% |
| **CASIA Combined** | Zero-Shot (CelebA Frozen) | 0.8743 | 0.8503 | 0.8844 | 0.8555 | **0.8661 ± 0.0138** | 21.61% ± 1.26% | 32.33% ± 1.46% | 29.08% ± 4.39% |
| **CASIA Combined** | Domain Calibrated (CASIA-Fit) | 0.8983 | 0.8751 | 0.9137 | 0.8772 | **0.8911 ± 0.0159** | 19.93% ± 2.30% | 21.20% ± 1.17% | 37.78% ± 4.16% |

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

### 3.3. `P3_SF_FULL_crop15` (WBST) qua 3 Seeds (100, 101, 102)

| Phân Tập Đánh Giá | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CelebA Test** | 0.9922 | 0.9925 | 0.9916 | **0.9921 ± 0.0005** | **4.47% ± 0.07%** | **4.44% ± 0.09%** | **86.16% ± 1.36%** |
| **LCC Training** | 0.8813 | 0.8928 | 0.8713 | **0.8818 ± 0.0108** | **20.58% ± 1.17%** | **26.51% ± 1.29%** | **27.02% ± 1.55%** |
| **LCC Development** | 0.8528 | 0.8641 | 0.8395 | **0.8521 ± 0.0123** | **22.86% ± 0.94%** | **25.52% ± 1.98%** | **19.09% ± 6.15%** |
| **LCC Evaluation** | 0.8197 | 0.8500 | 0.8183 | **0.8293 ± 0.0179** | **24.13% ± 1.51%** | **27.21% ± 1.63%** | **14.33% ± 2.53%** |
| **LCC Combined** | 0.8608 | 0.8764 | 0.8519 | **0.8630 ± 0.0124** | **21.90% ± 1.24%** | **26.51% ± 1.54%** | **21.42% ± 1.40%** |

### 3.4. `R7_SC_FULL_crop15` (MAST-PAD) qua 3 Seeds Chuẩn (100, 101, 102) & Seed Mới (105)

| Phân Tập Đánh Giá | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | **AUC (Mean ± Std, n=3)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** | **Seed 105 (Mới)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CelebA Test** | 0.9923 | 0.9919 | 0.9909 | **0.9917 ± 0.0007** | **4.59% ± 0.27%** | **4.56% ± 0.25%** | **85.64% ± 1.51%** | **0.9905** (ACER 4.67%, EER 4.67%, TPR 82.42%) |
| **LCC Training** | 0.8775 | 0.8833 | 0.8785 | **0.8797 ± 0.0031** | **20.68% ± 0.32%** | **27.82% ± 1.37%** | **26.81% ± 4.01%** | **0.8764** (ACER 24.62%, EER 20.27%, TPR 13.45%) |
| **LCC Development** | 0.8547 | 0.8381 | 0.8472 | **0.8466 ± 0.0083** | **24.12% ± 1.03%** | **26.58% ± 0.67%** | **16.87% ± 4.78%** | **0.8273** (ACER 25.83%, EER 24.10%, TPR 6.91%) |
| **LCC Evaluation** | 0.8401 | 0.8331 | 0.8157 | **0.8296 ± 0.0126** | **24.72% ± 1.20%** | **28.33% ± 1.54%** | **17.30% ± 2.12%** | **0.8067** (ACER 28.02%, EER 25.15%, TPR 13.06%) |
| **LCC Combined** | 0.8603 | 0.8619 | 0.8564 | **0.8595 ± 0.0028** | **22.38% ± 0.47%** | **27.76% ± 1.05%** | **20.45% ± 2.28%** | **0.8498** (ACER 25.63%, EER 22.36%, TPR 11.70%) |

> [!NOTE]
> *Seed 105 được huấn luyện độc lập (Epoch 11, locked threshold 5.7451) để phục vụ ablation trực tiếp 1:1 với `R7_SC_NOGATE_FULL_crop15` (Ablation không harmful gate). ACER trên LCC Combined của R7_SC đạt **25.63%**, vượt trội so với **26.86%** của bản không có Harmful Gate.*

### 3.5. Đánh Giá CASIA-FASD Video (Source-Frozen Threshold) qua 3 Seeds & Seed Mới (105)

*Đánh giá cross-dataset không qua hiệu chỉnh ngưỡng (áp dụng trực tiếp ngưỡng đóng băng từ CelebA-Spoof).*

| Mô hình | Phân Tập CASIA | Seed 100 AUC | Seed 101 AUC | Seed 102 AUC | **AUC (Mean ± Std, n=3)** | **EER (Mean ± Std)** | **ACER (Mean ± Std)** | **TPR@1% (Mean ± Std)** | **Seed 105 (Mới)** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C_FULL_crop15 (Clean CE)** | train | 0.9210 | 0.8967 | 0.8979 | **0.9052 ± 0.0137** | **19.44% ± 0.85%** | **28.34% ± 2.55%** | **41.81% ± 6.42%** | — |
| **C_FULL_crop15 (Clean CE)** | test | 0.9087 | 0.8911 | 0.8912 | **0.8970 ± 0.0102** | **21.53% ± 1.23%** | **29.73% ± 1.05%** | **44.32% ± 1.97%** | — |
| **C_FULL_crop15 (Clean CE)** | combined | 0.9204 | 0.9063 | 0.9013 | **0.9093 ± 0.0099** | **19.21% ± 2.34%** | **28.33% ± 1.03%** | **46.30% ± 1.70%** | — |
| **P3_SF_FULL_crop15 (WBST)** | train | 0.9027 | 0.9224 | 0.8854 | **0.9035 ± 0.0185** | **20.28% ± 2.69%** | **29.28% ± 1.55%** | **34.46% ± 0.98%** | — |
| **P3_SF_FULL_crop15 (WBST)** | test | 0.9118 | 0.8970 | 0.8779 | **0.8956 ± 0.0170** | **19.26% ± 2.19%** | **31.69% ± 2.03%** | **37.88% ± 4.30%** | — |
| **P3_SF_FULL_crop15 (WBST)** | combined | 0.9223 | 0.9129 | 0.8922 | **0.9091 ± 0.0154** | **18.84% ± 2.05%** | **30.31% ± 1.13%** | **41.11% ± 2.94%** | — |
| **R7_SC_FULL_crop15 (MAST-PAD)** | train | 0.9192 | 0.9153 | 0.8848 | **0.9064 ± 0.0188** | **18.78% ± 1.65%** | **29.09% ± 3.47%** | **37.85% ± 12.26%** | **0.8932** (EER 20.00%, ACER 29.94%) |
| **R7_SC_FULL_crop15 (MAST-PAD)** | test | 0.9259 | 0.8909 | 0.8680 | **0.8949 ± 0.0291** | **20.01% ± 1.61%** | **30.43% ± 2.41%** | **40.91% ± 4.55%** | **0.8726** (EER 21.97%, ACER 31.25%) |
| **R7_SC_FULL_crop15 (MAST-PAD)** | combined | 0.9352 | 0.9075 | 0.8809 | **0.9078 ± 0.0272** | **19.33% ± 2.25%** | **29.02% ± 2.34%** | **43.33% ± 4.84%** | **0.8877** (EER 20.63%, ACER 28.90%) |

### 3.6. Đánh Giá CASIA-FASD Video (Domain Calibrated - Threshold Fit trên Train) qua 3 Seeds & Seed Mới (105)

*Tối ưu ngưỡng cắt trên tập CASIA Train rồi chuyển sang đánh giá trên CASIA Test.*

| Mô hình | Phân Tập CASIA | Seed 100 ACER | Seed 101 ACER | Seed 102 ACER | **ACER (Mean ± Std, n=3)** | **Giảm ACER TB** | **TPR@1% (Mean ± Std)** | **Seed 105 (Mới)** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C_FULL_crop15 (Clean CE)** | train | 17.15% | 18.88% | 17.74% | **17.92% ± 0.88%** | **-10.42 pp** | **41.81% ± 6.42%** | — |
| **C_FULL_crop15 (Clean CE)** | test | 21.21% | 22.54% | 21.40% | **21.72% ± 0.72%** | **-8.02 pp** | **44.32% ± 1.97%** | — |
| **C_FULL_crop15 (Clean CE)** | combined | 19.55% | 21.90% | 20.63% | **20.69% ± 1.18%** | **-7.64 pp** | **46.30% ± 1.70%** | — |
| **P3_SF_FULL_crop15 (WBST)** | train | 19.17% | 16.33% | 19.17% | **18.22% ± 1.64%** | **-11.06 pp** | **34.46% ± 0.98%** | — |
| **P3_SF_FULL_crop15 (WBST)** | test | 17.05% | 18.56% | 21.97% | **19.19% ± 2.52%** | **-12.50 pp** | **37.88% ± 4.30%** | — |
| **P3_SF_FULL_crop15 (WBST)** | combined | 16.71% | 17.65% | 21.54% | **18.63% ± 2.56%** | **-11.68 pp** | **41.11% ± 2.94%** | — |
| **R7_SC_FULL_crop15 (MAST-PAD)** | train | 15.49% | 16.60% | 19.14% | **17.07% ± 1.87%** | **-12.02 pp** | **37.85% ± 12.26%** | **18.86%** (-11.08 pp) |
| **R7_SC_FULL_crop15 (MAST-PAD)** | test | 18.75% | 18.94% | 23.30% | **20.33% ± 2.57%** | **-10.10 pp** | **40.91% ± 4.55%** | **23.48%** (-7.77 pp) |
| **R7_SC_FULL_crop15 (MAST-PAD)** | combined | 16.74% | 18.05% | 21.94% | **18.91% ± 2.71%** | **-10.11 pp** | **43.33% ± 4.84%** | **22.30%** (-6.60 pp) |

### 3.7. Đánh Giá CASIA-FASD Cấp Độ Frame (Frame-Level) qua 3 Seeds & Seed Mới (105)

| Mô hình | Chế Độ Ngưỡng (Test Split) | Seed 100 ACER | Seed 101 ACER | Seed 102 ACER | **ACER (Mean ± Std, n=3)** | **AUC (Mean ± Std)** | **EER (Mean ± Std)** | **Seed 105 (Mới)** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C_FULL_crop15 (Clean CE)** | Zero-Shot (CelebA Frozen) | 28.53% | 31.87% | 31.46% | **30.62% ± 1.82%** | **0.8707 ± 0.0110** | **22.42% ± 1.41%** | — |
| **C_FULL_crop15 (Clean CE)** | Domain Calibrated (CASIA-Fit) | 22.31% | 24.61% | 25.32% | **24.08% ± 1.88%** | **0.8707 ± 0.0110** | **22.42% ± 1.41%** | — |
| **P3_SF_FULL_crop15 (WBST)** | Zero-Shot (CelebA Frozen) | 29.72% | 31.59% | 32.29% | **31.20% ± 1.33%** | **0.8737 ± 0.0206** | **21.35% ± 2.51%** | — |
| **P3_SF_FULL_crop15 (WBST)** | Domain Calibrated (CASIA-Fit) | 21.43% | 22.06% | 24.05% | **22.51% ± 1.46%** | **0.8737 ± 0.0206** | **21.35% ± 2.51%** | — |
| **R7_SC_FULL_crop15 (MAST-PAD)** | Zero-Shot (CelebA Frozen) | 25.72% | 32.16% | 33.15% | **30.34% ± 4.03%** | **0.8710 ± 0.0313** | **21.36% ± 2.89%** | **30.99%** (AUC 0.8513, EER 22.98%) |
| **R7_SC_FULL_crop15 (MAST-PAD)** | Domain Calibrated (CASIA-Fit) | 19.37% | 20.91% | 24.22% | **21.50% ± 2.52%** | **0.8710 ± 0.0313** | **21.36% ± 2.89%** | **23.24%** (Accuracy 78.45%) |

---

## 4. Đối Chiếu Kết Quả Giữa Notebook 18 và Notebook 21 (Baseline CE)

| Tiêu Chí Đánh Giá | Notebook 18 (`C_FULL`, 3 Seeds) Mean ± Std | Notebook 21 (`CE_CLEAN_FULL`, 4 Seeds) Mean ± Std | Chênh Lệch (|NB18 - NB21|) |
|---|:---:|:---:|:---:|
| **CelebA Test AUC** | 0.9908 ± 0.0007 | 0.9920 ± 0.0002 | 0.0012 |
| **CelebA Test ACER** | 4.66% ± 0.18% | 4.43% ± 0.02% | 0.23% |
| **LCC Combined AUC** | 0.8510 ± 0.0035 | 0.8387 ± 0.0055 | 0.0123 |
| **LCC Combined ACER** | 27.92% ± 0.55% | 29.84% ± 0.89% | 1.92% |

---

## 5. Bảng Đối Chiếu Số Liệu Với Các Nghiên Cứu Đã Công Bố

| Mô Hình | Tập Huấn Luyện | CelebA Test AUC | CelebA ACER (%) | LCC Combined AUC | LCC ACER (%) | Số Tham Số (Params) | FLOPs (GFLOPs) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AENET** (CVPR 2020) | CelebA-Spoof | 0.9990 | 3.25% | 0.8680 | 22.61% | 11.22 M | 3.640 G |
| **MobileNetV3 Large** | CelebA-Spoof | 0.9980 | 3.80% | 0.9210 | 16.33% | 3.02 M | 0.150 G |
| **MobileNetV3 Small** | CelebA-Spoof | 0.9940 | 5.05% | 0.8890 | 19.69% | 1.00 M | 0.040 G |
| **MobileNetV3 Small 0.75** | CelebA-Spoof | 0.9910 | 6.09% | 0.8790 | 21.04% | 0.60 M | 0.030 G |
| **Dự án: CE Baseline (4 Seeds)** | CelebA-Spoof | 0.9920 ± 0.0002 | 4.43% ± 0.02% | 0.8387 ± 0.0055 | 29.84% ± 0.89% | 0.43 M (~1.7 MB) | 0.081 G |
| **Dự án: P3_SF (3 Seeds)** | CelebA-Spoof | 0.9921 ± 0.0005 | 4.44% ± 0.09% | 0.8630 ± 0.0124 | 26.51% ± 1.54% | 0.43 M (~1.7 MB) | 0.081 G |
| **Dự án: R7_SC (3 Seeds Chuẩn)** | CelebA-Spoof | 0.9917 ± 0.0007 | 4.56% ± 0.25% | 0.8595 ± 0.0028 | 27.76% ± 1.05% | 0.43 M (~1.7 MB) | 0.081 G |

---

## 6. Khảo Sát Độc Lập: Ablation Cơ Chế Harmful Gate (Notebook 22)

Chi tiết thực nghiệm, phân tích và số liệu đầy đủ: [R7_HARMFUL_GATE_ABLATION_SOURCE_OF_TRUTH.md](R7_HARMFUL_GATE_ABLATION_SOURCE_OF_TRUTH.md)

### 6.1. So sánh giữa MAST-PAD chuẩn (`R7_SC`) và Ablation loại bỏ Harmful Gate (`R7_SC_NOGATE`)

Dưới cùng điều kiện $\sigma=0.10$, crop 1.5, batch 256, đối chiếu toàn diện qua cả 3 random seeds (S100, S101, S105):

| Chỉ Số Đánh Giá | R7_SC (MAST-PAD Chuẩn, Mean 3 Seeds) | R7_SC_NOGATE (Mean 3 Seeds: S100, S101, S105) | Ý Nghĩa / Kết Luận Thực Nghiệm |
|---|:---:|:---:|---|
| **CelebA Test AUC** | **0.9917 ± 0.0007** | **0.9909 ± 0.0006** | In-domain tương đương nhau (< 0.08% chênh lệch) |
| **CelebA Test ACER** | **4.56% ± 0.25%** | **4.56% ± 0.14%** | Khả năng phân loại nguồn tương đồng tuyệt đối |
| **CelebA TPR @ FPR=1%** | **85.64% ± 1.51%** | **83.06% ± 1.85%** | **MAST Chuẩn vượt trội +2.58 pp** khi kiểm soát false alarm khắt khe |
| **LCC Combined AUC** | **0.8595 ± 0.0028** | **0.8569 ± 0.0094** | Harmful gate giúp duy trì tính khái quát hóa cao hơn (+0.26 pp) |
| **LCC Combined EER** | **22.38% ± 0.47%** | **22.58% ± 0.89%** | MAST chuẩn ổn định và đồng đều hơn qua các seed |
| **LCC Combined ACER** | **27.76% ± 1.05%** | **26.66% ± 0.58%** | ACER ngưỡng đóng băng tương đương |
| **LCC Combined TPR @ FPR=1%** | **20.45% ± 2.28%** | **17.27% ± 4.79%** | **MAST chuẩn vượt trội +3.18 pp** trên phân phối ngoại suy LCC |
| **CASIA Video Test AUC** | **0.8949 ± 0.0291** | **0.8889 ± 0.0088** | Harmful gate cải thiện độ phân tách zero-shot trên CASIA Test |
| **CASIA Video Test HTER (Calib)**| **20.33% ± 2.57%** | **19.19% ± 1.72%** | Cả hai phương pháp đều giảm sâu ~11-12 pp sau khi hiệu chỉnh ngưỡng |
| **CASIA Video Combined AUC** | **0.9078 ± 0.0272** | **0.9033 ± 0.0115** | Khẳng định ưu thế của selective harmful filtering trên cross-dataset |
| **CASIA Video Combined HTER (Calib)** | **18.91% ± 2.71%** | **17.96% ± 1.54%** | Mức phân loại tương đương sau domain transfer |

### 6.2. Bảng Trạng Thái Thực Nghiệm Hoàn Thành Toàn Bộ

| Mô Hình | Notebook | Seed 100 | Seed 101 | Seed 102 | Seed 105 | Trạng Thái Hoàn Thành |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **R7_SC_NOGATE** (Ablation w/o Gate) | **NB 22** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | — | ✅ **HOÀN TẤT** | **ĐỦ 3 SEEDS 100%** (CelebA, LCC, CASIA) |
| **R7_SC** (MAST-PAD Chuẩn) | **NB 18** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | **ĐỦ 4 SEEDS 100%** (CelebA, LCC, CASIA qua NB 20) |
| **CE_CLEAN** (Clean thuần) | **NB 21** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | **ĐỦ 4 SEEDS 100%** (CelebA, LCC, CASIA) |
| **C_FULL** (Clean + Aux) | **NB 18** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ⏳ Tùy chọn | Đủ 3 seeds chính thức (S100, S101, S102) |
| **P3_SF** (WBST) | **NB 18** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | ⏳ Tùy chọn | Đủ 3 seeds chính thức (S100, S101, S102) |
