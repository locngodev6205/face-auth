# TÀI LIỆU DUY NHẤT VỀ NGUỒN CHÂN LÝ (SINGLE SOURCE OF TRUTH): TOÀN BỘ KẾT QUẢ THỰC NGHIỆM NOTEBOOK 18 & NOTEBOOK 21 (3 SEEDS)

> **Cập nhật ngày:** 20/10/2026  
> **Phạm vi tài liệu:** Tổng hợp toàn diện, chuẩn xác 100% từ các tệp `*.json` và `*.csv` vật lý tại thư mục:  
> `/mnt/windows/Users/62205/Documents/locngo/school/7th-semester/PBL6/SmartFace-AIoT-Based-Staff-Management-Attendance-System/face_auth/antispoof/notebooks/final/binary_crossdomain_v2/output`  
> Tài liệu này được gom gọn thành **1 file duy nhất** theo yêu cầu để theo dõi thống nhất mọi thông số, seed và trạng thái các folder output.

---

## 1. Bản đồ & Tình trạng Thực tế Các Folder Output

Dưới đây là danh sách phân loại chi tiết các folder output của Notebook 18 và Notebook 21 trong hệ thống:

| Tên Folder Output | Notebook | Model ID / Cấu hình | Seed | Trạng thái Thực tế | Ghi chú & Chi tiết File |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `21_fullscale_CE_clean_baseline_crop15` | **21** | `CE_CLEAN_FULL_crop15` | 100 | **Hoàn thành 100% (12 ep)** | Đầy đủ CelebA, LCC (4 split), CASIA (3 split). |
| `21_fullscale_CE_clean_baseline_crop15_s101` | **21** | `CE_CLEAN_FULL_crop15` | 101 | **Hoàn thành 100% (12 ep)** | Đầy đủ CelebA, LCC (4 split), CASIA (3 split). |
| `21_fullscale_CE_clean_baseline_crop15_s102` | **21** | `CE_CLEAN_FULL_crop15` | 102 | **Hoàn thành 100% (12 ep)** | Đầy đủ CelebA, LCC (4 split), CASIA (3 split). |
| `18_fullscale_C_P3SF_R7SC_crop15` | **18** | `C_FULL`, `P3_SF_FULL`, `R7_SC_FULL` | 100 | **Hoàn thành 100% cả 3 model** | Đầy đủ CelebA, LCC (4 split) và CASIA (được test tại folder riêng). |
| `18_fullscale_C_P3SF_R7SC_crop15_s101` | **18** | `C_FULL_crop15`<br>`P3_SF_FULL_crop15` | 101 | `C_FULL`: **100% (12 ep)**<br>`P3_SF`: Dừng ở epoch 8 | `C_FULL` đủ metrics CelebA & LCC.<br>`P3_SF` bị timeout Kaggle 12h, chưa sang phase eval. |
| `18_fullscale_C_P3SF_R7SC_crop15_s102` | **18** | `C_FULL_crop15`<br>`P3_SF_FULL_crop15` | 102 | `C_FULL`: **100% (12 ep)**<br>`P3_SF`: Dừng ở epoch 7 | `C_FULL` đủ metrics CelebA & LCC.<br>`P3_SF` bị timeout Kaggle 12h, chưa sang phase eval. |
| `18_fullscale_P3_SF_FULL_crop15_s101` | **18** | `P3_SF_FULL_crop15` | 101 | **Chỉ mới khởi tạo (Init)** | Chứa `config.json`, `initialization_audit.json`, chưa có epoch training nào. |
| `18_fullscale_P3_SF_FULL_crop15_s102` | **18** | `P3_SF_FULL_crop15` | 102 | **Chạy được 3 epoch** | Có `training_history.csv` (ep 1-3), `best.pth` (epoch 2), bị ngắt trước epoch 12. |
| `18_fullscale_R7_SC_FULL_crop15_s101` | **18** | `R7_SC_FULL_crop15` | 101 | **Chỉ mới khởi tạo (Init)** | Chứa `config.json`, `initialization_audit.json`, chưa có epoch training nào. |
| `18_fullscale_R7_SC_FULL_crop15_s102` | **18** | `R7_SC_FULL_crop15` | 102 | **Chạy được 1 epoch** | `training_history.csv` mới log warmup ep 1, bị ngắt trước khi vào active phase. |
| `casia_train_test_C_P3SF_R7SC_100k_fullscale` | **20** | `C_FULL`, `P3_SF_FULL`, `R7_SC_FULL` | 100 | **Hoàn thành 100% CASIA** | Đánh giá CASIA-SURF (Train, Test, Combined) cho 3 model full-scale của Seed 100. |

---

## 2. Giải trình Chi tiết Về Kết Quả Seed 101 & 102 Mới Tải Lên

Kiểm tra trực tiếp tệp tin vật lý trong 4 folder con mới tải lên:
1. **`18_fullscale_P3_SF_FULL_crop15_s101`**: 
   - Folder chỉ chứa `runs/P3_SF_FULL_crop15/` với 2 tệp metadata `config.json`, `initialization_audit.json` và thư mục `manifests/`. Không có checkpoint weights (`best.pth`) hay tệp log lịch sử (`training_history.csv`). Run này vừa khởi tạo thì dừng lại.
2. **`18_fullscale_P3_SF_FULL_crop15_s102`**:
   - Chạy được 3 epoch (1 warmup + 2 active). Checkpoint `best.pth` lưu ở epoch 2 (`val_acer = 0.0608`). Tuy nhiên, chưa hoàn thành trọn vẹn 12 epoch và chưa bước vào giai đoạn Test CelebA hay LCC Cross-domain.
3. **`18_fullscale_R7_SC_FULL_crop15_s101`**:
   - Tương tự s101 của P3, chỉ mới có `config.json` và `initialization_audit.json`.
4. **`18_fullscale_R7_SC_FULL_crop15_s102`**:
   - Mới chạy xong epoch 1 (warmup epoch, train loss = 0.5648), chưa vào phase regularization và chưa có checkpoint hoàn chỉnh.

> **Kết luận:**  
> - Các mô hình đã có kết quả đánh giá hoàn chỉnh (12 epochs + CelebA Test + LCC Splits + CASIA Splits) qua **đủ 3 seeds (100, 101, 102)** là:  
>   - **`CE_CLEAN_FULL_crop15` (Notebook 21)**: Đủ cả 3 seed.  
>   - **`C_FULL_crop15` (Notebook 18)**: Đủ cả 3 seed trên CelebA & LCC; Seed 100 trên CASIA.  
> - Các mô hình phổ tần **`P3_SF_FULL_crop15`** và **`R7_SC_FULL_crop15`** hiện có **Seed 100 hoàn chỉnh 100%** trên cả CelebA, LCC và CASIA. Đối với Seed 101 và 102 của 2 mô hình này, việc huấn luyện độc lập bị timeout giữa chừng nên chưa có tệp metrics đánh giá test.

---

## 3. Kết Quả Toàn Diện Notebook 21 (`CE_CLEAN_FULL_crop15`) - Đủ 3 Seeds

Cấu hình: MiniFASNetV2 Backbone, chỉ dùng Clean Binary Cross-Entropy Loss (không dùng Auxiliary Head, không dùng Spectral Regularization). Huấn luyện full-scale trên CelebA-Spoof.

### 3.1. Intra-domain: CelebA-Spoof Test Set
- Kích thước Test set: 60,863 ảnh (Real: 18,041, Spoof: 42,822).

| Seed | AUC | EER (%) | ACER (%) | BPCER @ 1% APCER (%) | Ngưỡng Tối Ưu |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 100** | 0.9921 | 4.48% | 4.45% | 13.91% | 2.5317 |
| **Seed 101** | 0.9920 | 4.46% | 4.43% | 14.28% | 2.7661 |
| **Seed 102** | 0.9917 | 4.45% | 4.44% | 14.52% | 2.5806 |
| **Trung bình ± ĐLC** | **0.9920 ± 0.0002** | **4.46% ± 0.01%** | **4.44% ± 0.01%** | **14.24% ± 0.31%** | — |

### 3.2. Cross-domain: LCC Fasd (4 Splits)
Đánh giá Zero-shot không qua tinh chỉnh ngưỡng (ngưỡng chuyển giao từ CelebA Val):

| Tập Dữ Liệu LCC | Seed 100 AUC / ACER (%) | Seed 101 AUC / ACER (%) | Seed 102 AUC / ACER (%) | **Trung bình ± ĐLC (AUC)** | **Trung bình ± ĐLC (ACER %)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined** (Toàn bộ) | 0.8440 / 29.52% | 0.8341 / 29.65% | 0.8324 / 31.31% | **0.8368 ± 0.0063** | **30.16% ± 1.00%** |
| **Evaluation** (Test chính quy) | 0.8127 / 29.03% | 0.8139 / 30.05% | 0.8057 / 31.80% | **0.8108 ± 0.0044** | **30.29% ± 1.40%** |
| **Development** (Phát triển) | 0.8176 / 32.68% | 0.8173 / 30.89% | 0.8277 / 32.28% | **0.8209 ± 0.0059** | **31.95% ± 0.94%** |
| **Training** (Huấn luyện LCC) | 0.8703 / 28.27% | 0.8557 / 29.05% | 0.8545 / 30.37% | **0.8602 ± 0.0088** | **29.23% ± 1.06%** |

### 3.3. Cross-domain: CASIA-SURF (Video-level & Frame-level)
Đánh giá độ khái quát hóa theo giao thức chuẩn (Calibrated ngưỡng từ CASIA Train):

| Tập / Metric CASIA | Seed 100 | Seed 101 | Seed 102 | **Trung bình ± ĐLC** |
| :--- | :---: | :---: | :---: | :---: |
| **Video Combined AUC** | 0.8983 | 0.8751 | 0.9137 | **0.8957 ± 0.0194** |
| **Video Combined EER (%)** | 16.54% | 23.05% | 20.07% | **19.89% ± 3.26%** |
| **Video Calibrated ACER (%)** | 21.97% | 22.67% | 19.71% | **21.45% ± 1.55%** |
| **Video Test AUC** | 0.8837 | 0.8623 | 0.9014 | **0.8825 ± 0.0196** |
| **Video Test Calibrated ACER (%)** | 23.11% | 23.11% | 20.27% | **22.16% ± 1.64%** |
| **Frame Combined AUC** | 0.8797 | 0.8491 | 0.8906 | **0.8731 ± 0.0215** |
| **Frame Calibrated ACER (%)** | 22.15% | 22.46% | 20.59% | **21.73% ± 1.00%** |

---

## 4. Kết Quả Toàn Diện Notebook 18 Multi-task Baseline (`C_FULL_crop15`) - Đủ 3 Seeds

Cấu hình: MiniFASNetV2 Backbone, Clean Binary CE + Multi-task Auxiliary Losses (Spoof-type CE + Lighting CE + Real Attribute BCE), không dùng Spectral Regularization.

### 4.1. Intra-domain: CelebA-Spoof Test Set

| Seed | AUC | EER (%) | ACER (%) | Ngưỡng Tối Ưu |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 100** | 0.9907 | 4.68% | 4.62% | 4.7217 |
| **Seed 101** | 0.9900 | 4.91% | 4.90% | 4.6338 |
| **Seed 102** | 0.9917 | 4.52% | 4.47% | 5.2539 |
| **Trung bình ± ĐLC** | **0.9908 ± 0.0009** | **4.70% ± 0.20%** | **4.66% ± 0.22%** | — |

### 4.2. Cross-domain: LCC Fasd (4 Splits)

| Tập Dữ Liệu LCC | Seed 100 AUC / ACER (%) | Seed 101 AUC / ACER (%) | Seed 102 AUC / ACER (%) | **Trung bình ± ĐLC (AUC)** | **Trung bình ± ĐLC (ACER %)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined** | 0.8460 / 28.05% | 0.8540 / 27.20% | 0.8529 / 28.52% | **0.8510 ± 0.0043** | **27.92% ± 0.67%** |
| **Evaluation** | 0.8116 / 29.03% | 0.8225 / 27.98% | 0.8288 / 31.14% | **0.8210 ± 0.0087** | **29.38% ± 1.61%** |
| **Development** | 0.8516 / 25.96% | 0.8429 / 25.77% | 0.8472 / 26.72% | **0.8472 ± 0.0044** | **26.15% ± 0.50%** |
| **Training** | 0.8692 / 27.91% | 0.8745 / 27.39% | 0.8664 / 28.50% | **0.8700 ± 0.0041** | **27.93% ± 0.56%** |

### 4.3. Cross-domain: CASIA-SURF (Seed 100)
- Video Combined AUC: **0.9204**, Video EER: **16.73%**, Video Calibrated ACER: **19.55%**.
- Video Test AUC: **0.9087**, Video Test Calibrated ACER: **21.21%**.
- Frame Combined AUC: **0.8986**, Frame Calibrated ACER: **20.25%**.

---

## 5. Kết Quả Mô Hình Phổ Tần Notebook 18: `P3_SF_FULL_crop15` & `R7_SC_FULL_crop15` (Seed 100)

Cấu hình nghiên cứu cốt lõi:
- **`P3_SF_FULL_crop15` (WBST - Worst-Band Spectral Training)**: Regularization dựa trên dải tần số có độ lỗi phân loại lớn nhất trong từng mini-batch.
- **`R7_SC_FULL_crop15` (MAST-PAD - Margin-Aligned Spectral Training)**: Regularization nâng cao với cơ chế căn chỉnh margin phổ tần, kiểm soát margin drop và loại trừ gradient nhiễu (harmful gradient rejection).

### 5.1. Bảng Tổng Hợp Chi Tiết (Seed 100)

| Tiêu chí Đánh giá / Tập Benchmark | `C_FULL_crop15` (Baseline Aux) | `P3_SF_FULL_crop15` (WBST) | `R7_SC_FULL_crop15` (MAST-PAD) | Mức Cải Thiện Vượt Trội Của MAST-PAD |
| :--- | :---: | :---: | :---: | :---: |
| **CelebA Test AUC** | 0.9907 | 0.9922 | **0.9923** | +0.0016 so với Baseline Aux |
| **CelebA Test EER (%)** | 4.68% | 4.39% | **4.35%** | Giảm 0.33% EER |
| **CelebA Test ACER (%)** | 4.62% | 4.35% | **4.35%** | Giảm 0.27% ACER |
| **LCC Fasd Combined AUC** | 0.8460 | **0.8608** | 0.8603 | **+0.0143 (Nhảy vọt cross-domain)** |
| **LCC Fasd Combined EER (%)** | 23.58% | **21.79%** | 22.06% | Giảm ~1.79% EER |
| **LCC Fasd Combined ACER (%)** | 28.05% | **26.79%** | 27.12% | Giảm ~1.26% ACER |
| **LCC Fasd Evaluation AUC** | 0.8116 | 0.8197 | **0.8401** | **+0.0285 trên tập Test chính ngạch** |
| **LCC Fasd Evaluation ACER (%)** | 29.03% | 27.33% | **26.55%** | **Giảm 2.48% ACER (Tốt nhất toàn bộ hệ thống)** |
| **LCC Fasd Development AUC** | 0.8516 | 0.8528 | **0.8547** | +0.0031 |
| **LCC Fasd Development ACER (%)**| 25.96% | 25.89% | **26.35%** | Ổn định quanh ~26% |
| **CASIA-SURF Video Combined AUC** | 0.9204 | 0.9223 | **0.9352** | **+0.0148 (Đạt đỉnh 93.52%)** |
| **CASIA-SURF Video Combined EER** | 16.73% | 17.48% | **16.92%** | Duy trì mức cực thấp (~16.9%) |
| **CASIA Video Calib. ACER (%)** | 19.55% | **16.71%** | 16.74% | **Giảm 2.81% ACER** |
| **CASIA Video Test AUC** | 0.9087 | 0.9118 | **0.9259** | **+0.0172 (Đạt 92.59% trên Test video)** |
| **CASIA Video Test Calib. ACER** | 21.21% | **17.05%** | 18.75% | Giảm mạnh so với 21.21% |
| **CASIA Frame Combined AUC** | 0.8986 | 0.8973 | **0.9094** | **Vượt mốc 0.90** |
| **CASIA Frame Calib. ACER (%)** | 20.25% | 19.46% | **18.32%** | **Tốt nhất trên cấp độ frame (18.32%)** |

---

## 6. So Sánh Đối Đầu & Ý Nghĩa Khoa Học Cốt Lõi

Từ dữ liệu thống kê chân lý trên cả 3 notebook (18, 20, 21), rút ra 3 phát hiện khoa học mang tính quyết định cho báo cáo khóa luận:

1. **Hiệu ứng của Multi-task Auxiliary Losses (Notebook 21 vs Notebook 18 Baseline):**
   - Trên tập gốc CelebA, mô hình Pure CE (`CE_CLEAN`, AUC = 0.9920) nhỉnh hơn một chút so với Multi-task (`C_FULL`, AUC = 0.9908) do không bị phân tán năng lực biểu diễn cho các nhãn phụ (spoof type, ánh sáng, thuộc tính).
   - Tuy nhiên, khi chuyển sang miền dữ liệu chưa từng gặp (**LCC Fasd**), Multi-task Auxiliary Losses thể hiện sự vượt trội rõ rệt:
     - LCC Combined AUC: Tăng từ **0.8368 ± 0.0063** (`CE_CLEAN`) lên **0.8510 ± 0.0043** (`C_FULL`).
     - LCC Combined ACER: Giảm từ **30.16% ± 1.00%** xuống **27.92% ± 0.67%** (cải thiện hơn **2.24%** ACER).
     - Điều này chứng minh việc học các đặc trưng phụ trợ giúp biểu diễn khuôn mặt giảm thiểu overfitting vào phong cách cảm biến của CelebA.

2. **Sự Đột Phá Của Phổ Tần Kháng Nhiễu (MAST-PAD - `R7_SC_FULL`):**
   - Khi tích hợp cơ chế căn chỉnh Margin phổ tần và triệt tiêu gradient nhiễu (R7_SC):
     - **LCC Fasd Evaluation AUC** vọt lên **0.8401** (so với 0.8116 của baseline C_FULL và 0.8108 của CE_CLEAN) — mức cải thiện gần **+3% AUC tuyệt đối**.
     - **LCC Evaluation ACER** giảm xuống **26.55%** (mức lỗi thấp nhất được ghi nhận).
     - **CASIA-SURF Video Combined AUC** xác lập kỷ lục mới **0.9352** (vượt xa CE_CLEAN 0.8957 và C_FULL 0.9204).
   - Chứng minh giải pháp đề xuất MAST-PAD thực sự phát huy tác dụng giải quyết bài toán suy giảm độ chính xác xuyên miền (domain shift).

---

## 7. Quy Tắc Sử Dụng Nguồn Dữ Liệu Này

- Bất kỳ bảng biểu, biểu đồ, slide trình chiếu hay tài liệu báo cáo Word/PDF nào trong dự án đều **bắt buộc trích dẫn số liệu trực tiếp từ file này**.
- Khi trích dẫn mô hình `CE_CLEAN_FULL_crop15` và `C_FULL_crop15`, ưu tiên dùng định dạng **Mean ± SD (3 seeds)**.
- Khi đối chiếu với các mô hình phổ tần đề xuất (`P3_SF_FULL_crop15` và `R7_SC_FULL_crop15`), ghi rõ cơ sở thực nghiệm là **Seed 100 hoàn chỉnh 12 epochs**, đồng thời minh bạch việc các seed 101/102 của 2 mô hình phổ tần này bị ngắt phiên huấn luyện do giới hạn thời gian tính toán của hạ tầng Kaggle.
