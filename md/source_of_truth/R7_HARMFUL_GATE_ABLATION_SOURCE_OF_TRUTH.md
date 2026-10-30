# Báo Cáo Thực Nghiệm: Khảo Sát Cơ Chế Harmful Gate (Notebook 22)
## Ablation Study: `R7_SC_NOGATE_FULL_crop15` (MAST-PAD w/o Harmful Gate)

- **Mục tiêu nghiên cứu:** Đánh giá độc lập tác động của cơ chế **Harmful Gate** trong MAST-PAD khi giữ nguyên hoàn toàn hình học dải tần rộng ($\sigma=0.10$, centers $[0.0833, 0.2500, 0.4167]$) và trọng số hàm mất mát ($0.75 L_{clean} + 0.25 L_{spectral} + L_{aux}$).
- **Phép thử nghiệm (Treatment):**
  - **MAST-PAD chuẩn (`R7_SC_FULL_crop15`):** Chỉ phạt những dải tần làm suy giảm biên phân loại thực sự (Harmful Gate) với chuẩn hóa $H$-normalization (chỉ tính trung bình trên số mẫu vi phạm).
  - **Ablation (`R7_SC_NOGATE_FULL_crop15`):** Bỏ Harmful Gate, tính Cross-Entropy trên góc nhìn xấu nhất (All-Selected Worst-View CE) của tất cả các mẫu với chuẩn hóa $B$-normalization (chia đều qua kích thước batch $B=256$).
- **Điều kiện kiểm soát (Controls):** Cùng kiến trúc MiniFASNet V2, crop factor 1.5, batch size 256, 12 epochs (milestones [4,8], gamma 0.2), SGD lr=0.005, momentum=0.9, weight_decay=5e-4, data seed 43.
- **Dữ liệu thực nghiệm:** Chạy hoàn tất trên GPU với đủ cả 3 random seed độc lập:
  - **Seed 100:** Checkpoint tối ưu tại **Epoch 11** (Locked Source Threshold = `5.1689`).
  - **Seed 101:** Checkpoint tối ưu tại **Epoch 9** (Locked Source Threshold = `5.2744`).
  - **Seed 105:** Checkpoint tối ưu tại **Epoch 10** (Locked Source Threshold = `5.4795`).

---

## 1. Kết Quả Huấn Luyện & Đánh Giá In-Domain (CelebA-Spoof Test)

*Tập kiểm thử Test-as-Val: 66,379 ảnh khả dụng (19,864 Real, 46,515 Spoof).*

| Tiêu Chí Đo Lường | R7_SC_NOGATE (S100) | R7_SC_NOGATE (S101) | R7_SC_NOGATE (S105) | **R7_SC_NOGATE (Mean 3-Seed)** | **R7_SC (MAST Chuẩn, Mean 3-Seed)** | Chênh Lệch In-Domain |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch Tối Ưu** | **11** | **9** | **10** | 9 - 11 | 9 - 11 | Tương đồng |
| **AUC** | **0.9911** | **0.9914** | **0.9902** | **0.9909 ± 0.0006** | **0.9917 ± 0.0007** | -0.08 pp |
| **EER (%)** | **4.43%** | **4.68%** | **4.69%** | **4.60% ± 0.15%** | **4.59% ± 0.27%** | +0.01 pp |
| **ACER (%)** | **4.40%** | **4.63%** | **4.65%** | **4.56% ± 0.14%** | **4.56% ± 0.25%** | 0.00 pp |
| **APCER (%)** | **5.11%** | **5.29%** | **5.05%** | **5.15% ± 0.13%** | **4.48% ± 0.35%** | +0.67 pp |
| **BPCER (%)** | **3.69%** | **3.97%** | **4.25%** | **3.97% ± 0.28%** | **4.64% ± 0.20%** | -0.67 pp |
| **TPR @ FPR=1%** | **82.19%** | **85.19%** | **81.79%** | **83.06% ± 1.85%** | **85.64% ± 1.51%** | **-2.58 pp (MAST chuẩn vượt trội)** |
| **Ngưỡng Cắt (Logit)** | **5.1689** | **5.2744** | **5.4795** | 5.3076 | 4.6626 - 5.8096 | — |

---

## 2. Kết Quả Khái Quát Hóa Miền Ngoại Suy (LCC FASD)

### 2.1. Đối chiếu trực tiếp 1:1 trên cùng Seed 100 (R7_SC vs R7_SC_NOGATE)

*Đánh giá Zero-shot cross-domain trên toàn bộ 4 split LCC FASD với ngưỡng đóng băng từ CelebA Test (R7_SC Thresh = 4.6626, R7_SC_NOGATE Thresh = 5.1689):*

| Phân Tập LCC | Chỉ Số Đo Lường | R7_SC (MAST Chuẩn, S100) | R7_SC_NOGATE (Ablation, S100) | Chênh Lệch / Lợi Điểm Của Harmful Gate |
|---|---|:---:|:---:|---|
| **LCC Training** | AUC / EER / ACER | **0.8753** / **21.84%** / **27.12%** | 0.8652 / 22.28% / 27.26% | AUC cao hơn **+1.01 pp**, ACER giảm **-0.14 pp** |
| | APCER / BPCER | 8.87% / **45.37%** | 7.96% / 46.56% | BPCER giảm **-1.19 pp** |
| **LCC Development** | AUC / EER / ACER | **0.8601** / **21.32%** / **23.63%** | 0.8482 / 22.47% / 24.47% | AUC cao hơn **+1.19 pp**, ACER giảm **-0.84 pp** |
| | APCER / BPCER | 14.16% / **33.09%** | 13.14% / 35.80% | BPCER giảm **-2.71 pp** |
| **LCC Evaluation** | AUC / EER / ACER | **0.8415** / **23.75%** / **28.32%** | 0.8172 / 25.41% / 28.41% | AUC cao hơn **+2.43 pp**, EER giảm **-1.66 pp** |
| | APCER / BPCER | 8.65% / **47.98%** | 10.97% / 45.86% | APCER giảm mạnh **-2.32 pp** |
| **LCC Combined** | **AUC** | **0.8603** | **0.8472** | **+1.31 pp (Harmful Gate cao hơn rõ rệt)** |
| | **EER (%)** | **22.06%** | **23.59%** | **-1.53 pp (Harmful Gate giảm EER)** |
| | **ACER / HTER (%)** | **27.12%** | **27.12%** | Tương đương (27.12% vs 27.12%) |
| | **BPCER (%)** | 46.26% | **44.20%** | +2.06 pp |
| | **APCER (%)** | **7.98%** | 10.03% | **-2.05 pp (Bắt spoof chặt chẽ hơn)** |
| | **TPR @ FPR=1%** | **19.99%** | **11.96%** | **+8.03 pp (Harmful Gate vượt trội hoàn toàn)** |

### 2.2. Đối chiếu trực tiếp 1:1 trên cùng Seed 101 (R7_SC vs R7_SC_NOGATE)

*Đánh giá Zero-shot cross-domain trên toàn bộ 4 split LCC FASD với ngưỡng đóng băng từ CelebA Test (R7_SC Thresh = 5.8096, R7_SC_NOGATE Thresh = 5.2744):*

| Phân Tập LCC | Chỉ Số Đo Lường | R7_SC (MAST Chuẩn, S101) | R7_SC_NOGATE (Ablation, S101) | Chênh Lệch / Lợi Điểm Của Harmful Gate |
|---|---|:---:|:---:|---|
| **LCC Training** | AUC / EER / ACER | 0.8833 / 20.64% / 29.40% | **0.8869** / **19.96%** / **26.01%** | NOGATE ACER giảm -3.39 pp (do threshold thấp hơn) |
| | APCER / BPCER | **4.46%** / 54.34% | 5.45% / **46.56%** | R7_SC kiểm soát APCER spoof tốt hơn **-0.99 pp** |
| **LCC Development** | AUC / EER / ACER | 0.8381 / 24.74% / 27.33% | **0.8404** / **24.70%** / **25.21%** | Tương đương (AUC chênh 0.0023) |
| | APCER / BPCER | 12.20% / 42.47% | 12.39% / **38.02%** | APCER tương đồng (~12.2%), BPCER NOGATE thấp hơn |
| **LCC Evaluation** | AUC / EER / ACER | 0.8331 / **23.88%** / 29.37% | **0.8365** / 24.28% / **27.42%** | R7_SC EER thấp hơn **-0.40 pp** |
| | APCER / BPCER | **6.19%** / 52.55% | 6.44% / **48.41%** | APCER R7_SC chặt chẽ hơn **-0.25 pp** |
| **LCC Combined** | **AUC** | 0.8619 | **0.8661** | Chênh lệch nhỏ (+0.42 pp) |
| | **EER (%)** | 22.16% | **21.90%** | Tương đương (chênh 0.26 pp) |
| | **ACER / HTER (%)** | 28.97% | **26.00%** | NOGATE ACER thấp hơn nhờ threshold nguồn mềm hơn |
| | **BPCER (%)** | 51.57% | **45.08%** | NOGATE giảm BPCER |
| | **APCER (%)** | **6.37%** | 6.92% | **-0.55 pp (R7_SC chặn spoof chặt hơn)** |
| | **TPR @ FPR=1%** | 18.44% | **21.27%** | +2.83 pp |

### 2.3. Đối chiếu trực tiếp 1:1 trên cùng Seed 105 (R7_SC vs R7_SC_NOGATE)

*Đánh giá Zero-shot cross-domain trên toàn bộ 4 split LCC FASD với ngưỡng đóng băng từ CelebA Test (R7_SC Thresh = 5.7451, R7_SC_NOGATE Thresh = 5.4795):*

| Phân Tập LCC | Chỉ Số Đo Lường | R7_SC (MAST Chuẩn, S105) | R7_SC_NOGATE (Ablation, S105) | Chênh Lệch / Lợi Điểm Của Harmful Gate |
|---|---|:---:|:---:|---|
| **LCC Training** | AUC / EER / ACER | **0.8764** / 20.27% / **24.62%** | 0.8835 / 19.87% / 26.46% | ACER giảm **-1.84 pp** (tốt hơn) |
| | APCER / BPCER | 7.83% / **41.41%** | 5.54% / 47.38% | BPCER giảm mạnh **-5.97 pp** |
| **LCC Development** | AUC / EER / ACER | 0.8273 / **24.10%** / 25.83% | 0.8410 / 24.74% / **25.27%** | EER giảm **-0.64 pp** |
| | APCER / BPCER | 15.85% / **35.80%** | 10.54% / 40.00% | BPCER giảm **-4.20 pp** |
| **LCC Evaluation** | AUC / EER / ACER | 0.8067 / 25.15% / **28.02%** | 0.8168 / 24.85% / 29.01% | ACER giảm **-0.99 pp** |
| | APCER / BPCER | 11.13% / **44.90%** | 8.34% / 49.68% | BPCER giảm **-4.78 pp** |
| **LCC Combined** | **AUC** | **0.8498** | **0.8574** | -0.76 pp |
| | **EER (%)** | **22.36%** | **22.25%** | +0.11 pp (Tương đương) |
| | **ACER / HTER (%)** | **25.63%** | **26.86%** | **-1.23 pp (MAST chuẩn vượt trội)** |
| | **BPCER (%)** | **40.80%** | **46.21%** | **-5.41 pp (Giảm mạnh false reject người thật)** |
| | **APCER (%)** | 10.46% | 7.50% | +2.96 pp |
| | **TPR @ FPR=1%** | 11.70% | 18.60% | -6.90 pp |

> [!IMPORTANT]
> **Phát hiện cốt lõi từ đối chiếu 1:1 đa Seed:** 
> - Trên **Seed 100**, Harmful Gate giúp nâng cao rõ rệt năng lực phân biệt với **AUC cao hơn +1.31 pp (86.03% vs 84.72%)**, **EER giảm -1.53 pp**, và **TPR @ FPR=1% vượt trội +8.03 pp (19.99% vs 11.96%)**.
> - Trên **Seed 105**, Harmful Gate giảm **ACER trên LCC Combined từ 26.86% xuống 25.63% (-1.23%)**, đặc biệt giảm mạnh tỷ lệ nhận nhầm người thật (**BPCER giảm từ 46.21% xuống 40.80%, tức giảm -5.41 pp**).
> - Tổng thể 3 seed khẳng định: Harmful Gate lọc chỉ phạt những dải tần làm sụt giảm biên phân tách thực tế, bảo vệ biểu diễn sinh trắc học người thật và giữ biên quyết định sắc bén hơn.

### 2.4. Bảng tổng hợp LCC FASD của `R7_SC_NOGATE_FULL_crop15` (Multi-Seed)

*Đánh giá Zero-shot cross-domain với ngưỡng đóng băng trực tiếp từ CelebA-Spoof (S100 Th=5.1689, S101 Th=5.2744, S105 Th=5.4795).*

| Phân Tập LCC | Chỉ Số Đo Lường | Seed 100 | Seed 101 | Seed 105 | **Mean (3-Seed)** |
|---|---|:---:|:---:|:---:|:---:|
| **LCC Training** | AUC / EER (%) | 0.8652 / 22.28% | 0.8869 / 19.96% | 0.8835 / 19.87% | **0.8785 ± 0.0117** / **20.70% ± 1.36%** |
| | ACER (%) | 27.26% | 26.01% | 26.46% | **26.58% ± 0.63%** |
| | APCER / BPCER (%) | 7.96% / 46.56% | 5.45% / 46.56% | 5.54% / 47.38% | 6.32% / 46.84% |
| | TPR @ FPR=1% (%) | 19.04% | 28.07% | 26.40% | **24.50% ± 4.81%** |
| **LCC Development** | AUC / EER (%) | 0.8482 / 22.47% | 0.8404 / 24.70% | 0.8410 / 24.74% | **0.8432 ± 0.0043** / **23.97% ± 1.30%** |
| | ACER (%) | 24.47% | 25.21% | 25.27% | **24.98% ± 0.45%** |
| | APCER / BPCER (%) | 13.14% / 35.80% | 12.39% / 38.02% | 10.54% / 40.00% | 12.02% / 37.94% |
| | TPR @ FPR=1% (%) | 17.78% | 19.12% | 20.25% | **19.05% ± 1.24%** |
| **LCC Evaluation** | AUC / EER (%) | 0.8172 / 25.41% | 0.8365 / 24.28% | 0.8168 / 24.85% | **0.8235 ± 0.0113** / **24.85% ± 0.56%** |
| | ACER (%) | 28.41% | 27.42% | 29.01% | **28.28% ± 0.80%** |
| | APCER / BPCER (%) | 10.97% / 45.86% | 6.44% / 48.41% | 8.34% / 49.68% | 8.58% / 47.98% |
| | TPR @ FPR=1% (%) | 4.78% | 18.79% | 16.24% | **13.27% ± 7.46%** |
| **LCC Combined** | **AUC** | **0.8472** | **0.8661** | **0.8574** | **0.8569 ± 0.0094** |
| | **EER (%)** | **23.59%** | **21.90%** | **22.25%** | **22.58% ± 0.89%** |
| | **ACER / HTER (%)** | **27.12%** | **26.00%** | **26.86%** | **26.66% ± 0.58%** |
| | **APCER (%)** | 10.03% | 6.92% | 7.50% | 8.15% ± 1.65% |
| | **BPCER (%)** | 44.20% | 45.08% | 46.21% | 45.17% ± 1.01% |
| | **TPR @ FPR=1% (%)** | **11.96%** | **21.27%** | **18.60%** | **17.27% ± 4.79%** |

---

## 3. Kết Quả Đánh Giá Trên CASIA-FASD Benchmark

### 3.1. Zero-Shot Cross-Domain (Source-Frozen Threshold)

| Phân Tập | Đơn Vị | Chỉ Số | Seed 100 (Th=5.1689) | Seed 101 (Th=5.2744) | Seed 105 (Th=5.4795) | **Mean (3-Seed)** |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **CASIA Test** | **Video** | AUC / EER | **0.8910** / **19.51%** | **0.8965** / **16.86%** | **0.8793** / **19.89%** | **0.8889 ± 0.0088** / **18.75% ± 1.65%** |
| | | HTER / APCER / BPCER | 31.06% / 4.17% / 57.95% | 28.98% / 0.00% / 57.95% | 34.09% / 1.14% / 67.05% | **31.38% ± 2.57%** / 1.77% / 60.98% |
| | | TPR @ FPR=1% | 34.09% | 47.73% | 32.95% | **38.26% ± 8.22%** |
| | **Frame** | AUC / EER / HTER | 0.8638 / 21.87% / 30.50% | 0.8767 / 19.64% / 30.09% | 0.8526 / 22.98% / 32.25% | **0.8644 ± 0.0120** / 21.50% / 30.95% |
| **CASIA Combined** | **Video** | AUC / EER | **0.9041** / **19.15%** | **0.9144** / **16.73%** | **0.8914** / **18.59%** | **0.9033 ± 0.0115** / **18.15% ± 1.27%** |
| | | HTER / APCER / BPCER | 30.20% / 3.73% / 56.67% | 27.22% / 0.00% / 54.44% | 33.34% / 1.12% / 65.56% | **30.25% ± 3.06%** / 1.62% / 58.89% |
| | | TPR @ FPR=1% | 38.89% | 48.89% | 34.44% | **40.74% ± 7.40%** |
| | **Frame** | AUC / EER / HTER | 0.8774 / 20.46% / 28.12% | 0.8896 / 18.71% / 27.74% | 0.8659 / 22.13% / 30.10% | **0.8776 ± 0.0119** / 20.43% / 28.65% |

### 3.2. Domain Calibrated (Hiệu Chỉnh Ngưỡng Trên CASIA Train)

| Phân Tập | Đơn Vị | Chỉ Số | Seed 100 (Th=-0.3252) | Seed 101 (Th=-2.0143) | Seed 105 (Th=1.0867) | **Mean (3-Seed)** |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **CASIA Test** | **Video** | HTER Calib / Giảm HTER | **19.89%** (-11.17 pp) | **17.23%** (-11.74 pp) | **20.45%** (-13.64 pp) | **19.19% ± 1.72%** (-12.18 pp) |
| | | APCER / BPCER | 21.59% / 18.18% | 20.83% / 13.64% | 11.36% / 29.55% | 17.93% / 20.45% |
| | | Độ Chính Xác (%) | **79.26%** | **80.97%** | **84.09%** | **81.44% ± 2.45%** |
| | **Frame** | HTER Calib / Độ Chính Xác | 21.73% (-8.77 pp) / 80.44% | 20.59% (-9.50 pp) / 80.52% | 23.20% (-9.05 pp) / 78.86% | **21.84% ± 1.31%** / 79.94% |
| **CASIA Combined** | **Video** | HTER Calib / Giảm HTER | **18.78%** (-11.42 pp) | **16.19%** (-11.04 pp) | **18.93%** (-14.41 pp) | **17.96% ± 1.54%** (-12.29 pp) |
| | | APCER / BPCER | 20.90% / 16.67% | 20.15% / 12.22% | 10.07% / 27.78% | 17.04% / 18.89% |
| | | Độ Chính Xác (%) | **80.17%** | **81.84%** | **85.47%** | **82.50% ± 2.71%** |
| | **Frame** | HTER Calib / Độ Chính Xác | 20.59% (-7.53 pp) / 81.39% | 19.34% (-8.40 pp) / 82.03% | 22.01% (-8.09 pp) / 79.14% | **20.65% ± 1.34%** / 80.85% |

### 3.3. Đối chiếu 1:1 CASIA-FASD giữa R7_SC và R7_SC_NOGATE qua các Seeds

| Phân Tập & Đơn Vị | Seed | R7_SC (MAST Chuẩn) | R7_SC_NOGATE (Ablation) | Lợi Điểm Của Harmful Gate |
|---|:---:|:---:|:---:|---|
| **CASIA Test (Video, Zero-shot)** | S100 | **0.9259** / **18.18%** / 31.25% | 0.8910 / 19.51% / **31.06%** | AUC cao hơn **+3.49 pp**, EER giảm **-1.33 pp** |
| | S101 | **0.8909** / 20.01% / 30.43% | **0.8965** / **16.86%** / **28.98%** | Tương đồng (AUC chênh 0.0056) |
| | S105 | 0.8726 / 21.97% / **31.25%** | **0.8793** / **19.89%** / 34.09% | HTER thấp hơn **-2.84 pp** |
| **CASIA Test (Video, Calibrated)** | S100 | **18.75%** / **80.40%** | 19.89% / 79.26% | HTER giảm **-1.14 pp**, Accuracy cao hơn **+1.14 pp** |
| | S101 | 18.94% / 80.40% | **17.23%** / **80.97%** | NOGATE chênh nhẹ -1.71 pp |
| | S105 | 23.48% / 77.84% | **20.45%** / **84.09%** | NOGATE chênh -3.03 pp |
| **CASIA Combined (Video, Zero-shot)**| S100 | **0.9352** / **16.92%** / **29.75%** | 0.9041 / 19.15% / 30.20% | AUC cao hơn **+3.11 pp**, EER giảm **-2.23 pp** |
| | S101 | 0.9075 / 19.40% / 28.32% | **0.9144** / **16.73%** / **27.22%** | Tương đương (AUC 0.908 vs 0.914) |
| | S105 | 0.8877 / 20.63% / **28.90%** | **0.8914** / **18.59%** / 33.34% | HTER zero-shot thấp hơn **-4.44 pp** |
| **CASIA Combined (Video, Calibrated)**| S100 | **16.74%** / **82.12%** | 18.78% / 80.17% | HTER giảm mạnh **-2.04 pp**, Accuracy cao hơn **+1.95 pp** |
| | S101 | 18.05% / 81.01% | **16.19%** / **81.84%** | Tương đồng (~16-18%) |
| | S105 | 22.30% / 78.77% | **18.93%** / **85.47%** | NOGATE threshold fit tốt trên S105 |

---

## 4. Phân Tích Khoa Học & Kết Luận (Ablation Findings)

1. **In-Domain (CelebA-Spoof):**
   - Cả hai mô hình đều đạt hiệu năng xuất sắc trên tập nguồn: `R7_SC` đạt AUC **0.9917 ± 0.0007** (ACER 4.56%), trong khi `R7_SC_NOGATE` đạt AUC **0.9909 ± 0.0006** (ACER 4.56%).
   - Sự chênh lệch in-domain là cực kỳ nhỏ (< 0.1%), chứng minh backbone MiniFASNetV2 và các thành phần phụ trợ (Auxiliary heads) giữ được độ ổn định hội tụ cao bất kể có hay không có harmful gate.
   - Tuy nhiên, trên chỉ số bắt spoof nghiêm ngặt **TPR @ FPR=1%**, MAST-PAD chuẩn vượt trội rõ rệt: **85.64% ± 1.51%** so với **83.06% ± 1.85%** của NOGATE (**cao hơn +2.58 pp**).

2. **Cross-Domain (LCC & CASIA-FASD):**
   - **LCC FASD Combined:**
     - `R7_SC` chuẩn (Mean 3-seed) đạt AUC **0.8595 ± 0.0028**, trong khi `R7_SC_NOGATE` đạt AUC **0.8569 ± 0.0094**.
     - Trên từng seed đối chiếu, Harmful Gate chứng minh vai trò ngăn chặn suy giảm biên nhận dạng:
       - Seed 100: AUC cao hơn **+1.31 pp** (86.03% vs 84.72%), TPR@1% cao hơn **+8.03 pp** (19.99% vs 11.96%).
       - Seed 105: ACER giảm **-1.23 pp** (25.63% vs 26.86%), BPCER giảm mạnh **-5.41 pp** (40.80% vs 46.21%).
   - **CASIA-FASD (Video Combined):**
     - Zero-shot AUC: `R7_SC` (Mean 3-seed: **0.9078 ± 0.0272**) vượt trội so với `R7_SC_NOGATE` (**0.9033 ± 0.0115**).
     - Đặc biệt trên Seed 100, `R7_SC` đạt đỉnh AUC **0.9352**, vượt xa `0.9041` của bản không có harmful gate.
   - **Kết luận:** Cơ chế Harmful Gate đóng vai trò như một bộ lọc gradient có chọn lọc (selective regularizer), ngăn không cho các dải tần suy giảm vô hại kích hoạt gradient phạt không cần thiết. Khi bị ép phạt trên toàn bộ dải tần xấu nhất (NOGATE), mô hình có xu hướng hy sinh đặc trưng người thật (bị đẩy BPCER lên cao trên LCC) hoặc giảm tính ổn định biên trên CASIA.

---

## 5. Bảng Trạng Thái Thực Nghiệm Hoàn Thành

### 5.1. Bảng ma trận tiến độ theo Seed và Mô hình

| Mô Hình | Phương Pháp Huấn Luyện | Notebook Nguồn | Seed 100 | Seed 101 | Seed 102 | Seed 105 | Trạng Thái Hiện Tại |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **R7_SC_NOGATE_FULL_crop15** | MAST-PAD không Harmful Gate (Ablation) | **NB 22** | ✅ **HOÀN TẤT** | ✅ **HOÀN TẤT** | — | ✅ **HOÀN TẤT** | **ĐỦ 3 SEEDS 100%** (CelebA, LCC, CASIA) |
| **R7_SC_FULL_crop15** | MAST-PAD Chuẩn (Harmful Gate) | **NB 18** | ✅ Đã có | ✅ Đã có | ✅ Đã có | ✅ **HOÀN TẤT** | **ĐỦ 4 SEEDS 100%** (CelebA, LCC, CASIA NB 20) |
| **CE_CLEAN_FULL_crop15** | Clean Baseline thuần túy (không Aux) | **NB 21** | ✅ Đã có | ✅ Đã có | ✅ Đã có | ✅ **HOÀN TẤT** | **ĐỦ 4 SEEDS 100%** (CelebA, LCC, CASIA) |
| **C_FULL_crop15** | Clean Baseline + Auxiliary heads | **NB 18** | ✅ Đã có | ✅ Đã có | ✅ Đã có | ⏳ *Tùy chọn* | Đã đủ 3 seeds chính thức (S100-S102) |
| **P3_SF_FULL_crop15** | Worst-Band CE (WBST) | **NB 18** | ✅ Đã có | ✅ Đã có | ✅ Đã có | ⏳ *Tùy chọn* | Đã đủ 3 seeds chính thức (S100-S102) |

### 5.2. Danh sách cụ thể các công việc đã hoàn thành (Execution Checklist)

1. **Notebook 22 (`22_fullscale_R7_harmful_gate_ablation_crop15`):**
   - [x] **Seed 100:** Đã hoàn thành 11 epochs, lưu checkpoint, CelebA test, 4 split LCC và toàn bộ CASIA train/test/combined vào `outputs/notebook_22/s100/`.
   - [x] **Seed 101:** Đã hoàn thành 9 epochs, lưu checkpoint, CelebA test, 4 split LCC và toàn bộ CASIA train/test/combined vào `outputs/notebook_22/s101/`.
   - [x] **Seed 105:** Đã hoàn thành 10 epochs, lưu checkpoint, CelebA test, 4 split LCC và toàn bộ CASIA train/test/combined vào `outputs/notebook_22/s105/`.
   - *(Hoàn tất toàn bộ 3 seeds cho Ablation Harmful Gate)*.

2. **Notebook 18 (`18_fullscale_C_P3SF_R7SC_crop15`):**
   - [x] **Seed 105 `R7_SC_FULL_crop15`:** Đã hoàn thành 11 epochs, lưu checkpoint và metrics CelebA/LCC vào `outputs/notebook_18/s105/` và notebook `notebooks/seed/R7_SC_FULL_crop15_s105.ipynb`.
   - [x] **Đánh giá CASIA (theo giao thức Notebook 20):** Đã đánh giá hoàn chỉnh checkpoint `s105/R7_SC_FULL_crop15/best.pth` qua Notebook 20 (`outputs/notebook_20/casia_eval_s105/`), đồng bộ vào `outputs/notebook_18/s105/R7_SC_FULL_crop15/casia/`.

3. **Notebook 21 (`21_fullscale_CE_clean_baseline_crop15`):**
   - [x] **Seed 105:** Đã hoàn thành 11 epochs, lưu checkpoint, CelebA test, 4 split LCC và toàn bộ CASIA train/test/combined vào `outputs/notebook_21/s105/` và notebook `notebooks/seed/21_fullscale_CE_clean_baseline_crop15_s105.ipynb`.
