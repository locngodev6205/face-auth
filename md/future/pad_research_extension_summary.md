# Mở rộng mô hình Spatial–Frequency PAD thành Research Paper (Q3 → Q2/Q1)

> Tài liệu tổng hợp: bối cảnh, động lực, nền tảng lý thuyết, hướng triển khai và roadmap.
> Kế thừa từ 2 tài liệu gốc: `pad_spatial_frequency_architecture.md`, `pad_frequency_domain_references.md`.

---

## 1. Bối cảnh

### 1.1. Điểm xuất phát

Đồ án hiện tại đã xây dựng một mô hình Face Presentation Attack Detection (PAD) nhẹ, xử lý ảnh RGB single-frame, gồm:

- **Spatial branch**: MobileNetV3-Large (ImageNet pretrained) → spatial feature 256-D.
- **Frequency branch**: 2D DCT (deterministic, không tham số) → Tiny CNN nhẹ → frequency feature 64-D.
- **Fusion**: Concatenation → MLP → classifier REAL/SPOOF.
- **Dataset**: CelebA-Spoof, subject-disjoint split.
- **Mục tiêu ban đầu**: kiểm chứng giả thuyết "bổ sung frequency feature có cải thiện cross-domain generalization so với spatial-only baseline hay không", đồng thời đo chi phí triển khai (params, FLOPs, latency) cho edge device.

Đây là một thiết kế **dual-branch concat** hợp lý cho scope đồ án, nhưng về bản chất kỹ thuật khá tương đồng với hướng tiếp cận mà nhiều paper 2020–2021 đã công bố (xem mục 3).

### 1.2. Vấn đề khi mở rộng lên research paper

Khi đặt mục tiêu submit Q3/Q2/Q1, thiết kế "spatial + frequency concat/gated fusion, so sánh vài phép biến đổi tần số" **không còn đủ novelty** vì:

- Chen et al. (2020) đã dùng kiến trúc two-stream với high-pass/low-pass filter cố định + Cross-Frequency Spatial Attention.
- Huang et al. (2020) đã đưa frequency spectrum vào CNN riêng biệt (FreqSpatialTemporalNet).
- Roy et al. (2021 — Bi-FPNFAS) đã dùng auxiliary supervision tái tạo Fourier spectrum.
- Zhang & Xiang (2020) đã kết hợp DWT-LBP-DCT.

Nếu paper chỉ lặp lại "concat vs gated fusion, DCT vs FFT vs wavelet", reviewer Q2/Q1 sẽ hỏi trực diện: *"Điểm khác với các công trình trên là gì?"*

### 1.3. Bức tranh SOTA hiện tại (2024–2026)

Field đã dịch chuyển trọng tâm sang các hướng nặng hơn nhiều so với dual-branch concat cổ điển:

| Hướng SOTA hiện tại | Đặc điểm | Hạn chế đối với bài toán edge |
|---|---|---|
| CLIP/LLM-prompt cho domain generalization (TeG-DG, CCPE, FLIP) | Dùng backbone CLIP-ViT lớn + text prompt | Không phù hợp thiết bị edge/mobile |
| Unified physical + digital attack detection (FA³-CLIP, Mixture-of-Attack-Experts) | Một model xử lý cả PA vật lý và deepfake | Backbone rất nặng, scope khác (có digital manipulation) |
| Self-supervised frequency pretraining (MFAE, FAMIM) | Masked autoencoder trên frequency map | Cần data lớn, pretraining cost cao |
| Test-time domain generalization | Thích ứng tại inference, không cần retrain | Kỹ thuật phức tạp, chưa phù hợp effort hiện tại |
| Interpretable FAS (MLLM-based, 2024–2025) | Diễn giải bằng ngôn ngữ tự nhiên qua MLLM | Backbone rất nặng, không phù hợp edge |

**Khoảng trống (gap) nhận diện được**: gần như toàn bộ SOTA về generalization/interpretability hiện tại đều dùng backbone rất nặng. Chưa có nhiều nghiên cứu kết hợp **lightweight + adaptive frequency decomposition + interpretability dựa trên vật lý + robustness dưới nén ảnh thực tế** trong cùng một hệ thống nhỏ, hướng edge.

---

## 2. Động lực nghiên cứu

### 2.1. Tại sao miền tần số vẫn còn giá trị

- Spoof artifact (moiré của màn hình, halftone dot của máy in, viền mask, aliasing/blurring) tồn tại ở dạng vi cấu trúc, phân bố trên các dải tần số không gian khác nhau — khó quan sát trực tiếp trong miền không gian nhưng rõ ràng hơn trong miền tần số.
- Nghiên cứu 2024 (Kim et al., CVPRW) chỉ ra: tín hiệu phân biệt real/spoof chủ yếu cư trú ở dải **tần số cao vi mô**, nhưng dataset bias (ánh sáng, camera, background) lại chi phối mạnh miền không gian — khiến CNN thông thường dễ học shortcut sai.

### 2.2. Ba khoảng trống cụ thể được chọn để khai thác

1. **DCT/FFT là phép biến đổi cố định**, ranh giới "low/high frequency" do con người tự đặt (Chen et al. 2020 dùng đúng 2–3 bộ lọc cố định) — không tối ưu theo dữ liệu thực tế.
2. **Frequency feature bị coi là hộp đen** — các paper hiện tại hiếm khi giải thích *dải tần số nào* tương ứng với *nguyên nhân vật lý nào* (moiré, halftone...), và các paper interpretable mới (2024–2025) giải quyết bằng MLLM rất nặng, không hợp edge.
3. **Ít nghiên cứu kiểm chứng frequency feature có sống sót qua nén ảnh/video thực tế** (JPEG, codec video call) — đây là rủi ro triển khai thật nhưng ít được benchmark nghiêm túc.

### 2.3. Câu hỏi nghiên cứu trung tâm

> Liệu một mô hình PAD nhẹ, nếu được trang bị (i) cơ chế phân tách tần số linh hoạt hơn DCT cố định, (ii) khả năng diễn giải gắn với nguyên nhân vật lý của spoof, và (iii) được huấn luyện/đánh giá có tính đến suy hao do nén ảnh thực tế — có đạt được generalization tốt hơn và đáng tin cậy hơn để triển khai edge, so với thiết kế dual-branch concat cổ điển?

---

## 3. Nền tảng lý thuyết

### 3.1. Cơ sở từ literature đã kiểm chứng (5 paper nền)

| # | Công trình | Đóng góp lý thuyết dùng lại |
|---|---|---|
| 1 | Chen, Yang, Wang (2020), IEEE MIPR | Xác nhận thiết kế two-stream (spatial/LF – frequency/HF) hợp lý; đồng thời là baseline cần vượt qua (fixed high/low-pass filter) |
| 2 | Huang, Zhang, Wang (2020), arXiv | Chứng minh frequency spectrum có thể học như feature map 2D bằng CNN riêng |
| 3 | Roy et al. (2021), MDPI Sensors — Bi-FPNFAS | Bằng chứng thực nghiệm mạnh cho cross-dataset generalization khi có frequency-aware supervision (ACER 2.92% trên OULU-NPU Protocol IV) |
| 4 | Kim et al. (2024), CVPRW | Lý giải nguyên nhân: cues phân biệt cư trú ở high-frequency vi mô; dataset bias chi phối miền không gian |
| 5 | Zhang & Xiang (2020), Signal Processing: Image Communication | Cơ sở cho hướng kết hợp nhiều loại biến đổi/đặc trưng (DWT-LBP-DCT), hỗ trợ luận điểm edge-efficient |

### 3.2. Ba giả thuyết nghiên cứu (H1–H3) — kế thừa và mở rộng

- **H1 — Complementary evidence**: spatial feature và frequency feature mang thông tin trực giao, bù trừ cho nhau (Chen 2020, Huang 2020).
- **H2 — Cải thiện cross-dataset generalization**: frequency-aware model giảm mức độ sụp đổ hiệu năng khi đổi domain (Roy 2021, Kim 2024).
- **H3 — Edge-friendly efficiency**: phép biến đổi tần số không tốn tham số + CNN nhẹ giữ chi phí triển khai thấp (Zhang & Xiang 2020).

### 3.3. Mở rộng giả thuyết cho paper mới (H4–H6)

- **H4 (Adaptive decomposition)**: phân tách tần số thành nhiều dải với trọng số học được (thay cho ranh giới cố định của Chen 2020) giúp mô hình tập trung đúng dải mang spoof cue, cải thiện hiệu năng so với fixed-band.
- **H5 (Physical interpretability)**: trọng số attention học được trên các dải tần số có thể tương quan với đặc tính vật lý đã biết của attack instrument (tần số moiré phụ thuộc refresh rate/pixel pitch màn hình; halftone phụ thuộc DPI máy in), cho phép diễn giải quyết định của mô hình bằng bằng chứng định lượng, không cần MLLM.
- **H6 (Compression robustness)**: mô hình được huấn luyện có nhận thức về suy hao do nén ảnh (JPEG) sẽ suy giảm ACER ít hơn dưới điều kiện nén thực tế, so với mô hình chỉ học trên ảnh gốc.

---

## 4. Hướng triển khai

### 4.1. Ba hướng nghiên cứu được chốt kết hợp

| Hướng | Vai trò | Thiết kế đề xuất (bản low-effort, khả thi Q3+) |
|---|---|---|
| **1. Physically-grounded interpretability** | Lớp diễn giải đặt lên cơ chế frequency | Attention nhỏ học trọng số cho từng dải tần số + inverse-transform để visualize vùng ảnh tương ứng |
| **2. Adaptive/learnable frequency decomposition** | Cơ chế nền, thay thế fixed DCT-band của Chen 2020 | K dải hình khuyên tần số cố định (K=6–8, tính 1 lần, không cần tham số học cho ranh giới) — vẫn "thích nghi" hơn fixed 2–3 band, dễ code hơn learnable-boundary đầy đủ |
| **3. Compression robustness** | Quy trình training + đánh giá, không phải module riêng | JPEG augmentation ngẫu nhiên (quality 30–95) lúc train; đánh giá offline ở 4 mức quality cố định (90/70/50/30) |

### 4.2. Kiến trúc tổng thể

```text
                    FACE IMAGE 224×224×3
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
     SPATIAL BRANCH                  FREQUENCY BRANCH
     MobileNetV3-Large                     │
     (giữ nguyên từ đồ án)                 ▼
            │                        2D DCT (fixed)
            ▼                               │
    Spatial Feature 256-D                   ▼
                                  K dải tần số hình khuyên (fixed masks)
                                  K = 6–8, tính 1 lần
                                             │
                                             ▼
                                  Tiny CNN (dùng chung trọng số cho K dải)
                                             │
                                             ▼
                                  K vector đặc trưng
                                             │
                                             ▼
                                  Attention layer (học trọng số α₁…α_K)
                                  ── lớp diễn giải: Hướng 1 ──
                                             │
                                             ▼
                                  Frequency Feature (weighted sum)
            │                               │
            └───────────────┬───────────────┘
                            ▼
                     FEATURE FUSION
                    Concatenate + MLP
                            │
                            ▼
                   PAD CLASSIFIER → REAL / SPOOF

Training-time augmentation (Hướng 3):
  Ảnh gốc ──50%──► JPEG compression (quality random 30–95) ──► đưa vào cả 2 nhánh
```

### 4.3. Khác biệt so với baseline Chen et al. (2020) — điểm mấu chốt để trả lời câu hỏi novelty

| | Chen et al. 2020 | Đề xuất |
|---|---|---|
| Số dải tần số | 2–3 (chỉ high-pass/low-pass) | K = 6–8 dải mịn hơn |
| Trọng số giữa các dải | Không có (fusion đều qua CFSA) | Attention học theo từng ảnh |
| Diễn giải vật lý | Không đề cập | Đối chiếu attention với tần số lý thuyết (moiré/halftone) + inverse-transform visualization |
| Robustness dưới nén | Không đánh giá | Train + eval có JPEG augmentation, đo degradation curve |

### 4.4. Bảng ablation cần chạy (tối thiểu)

1. Spatial-only (baseline gốc của đồ án).
2. Spatial + fixed 2-band (tái hiện kiểu Chen 2020, để so sánh công bằng, cùng backbone/data).
3. Spatial + K-band (không attention, chỉ concat).
4. Spatial + K-band + attention (đề xuất, chưa train compression).
5. Spatial + K-band + attention + JPEG augmentation (đề xuất đầy đủ).

→ Đánh giá cả 5 cấu hình trên **test gốc** và **4 mức JPEG quality** (90/70/50/30) để tách rõ đóng góp riêng của từng hướng.

### 4.5. Dataset và protocol đánh giá

| Mục tiêu | Dataset/protocol | Ghi chú |
|---|---|---|
| So sánh cơ bản, ablation | CelebA-Spoof (subject-disjoint split) | Đã có sẵn; đủ đa dạng attack category (Photo/Poster/A4/Phone/PC/Pad) khớp câu chuyện moiré/halftone |
| Cross-domain generalization (H2) | O&C&I&M leave-one-out (OULU-NPU, CASIA-MFSD, Idiap Replay-Attack, MSU-MFSD) | Protocol chuẩn cộng đồng dùng nhiều nhất; ưu tiên xin quyền truy cập nếu chưa có — thường dễ xin hơn WMCA/SiW-Mv2 |
| Unseen attack type (mạnh hơn, cần cho Q1/Q2) | SiW-Mv2 hoặc WMCA, leave-one-attack-out | **Hiện chưa có quyền truy cập** — để ở mục "hướng mở rộng" hoặc Future Work nếu không xin được |
| Compression robustness (H6) | Tự thiết kế trên CelebA-Spoof/O&C&I&M | Không có benchmark "phổ biến" sẵn có cho việc này — so sánh nội bộ giữa baseline và model đề xuất |

**Về scope**: toàn bộ benchmark trên là **PAD trên ảnh tĩnh (single-frame)** — khác với bài toán deepfake/digital manipulation (FaceForensics++, Celeb-DF...). Giữ đúng scope này, không mở rộng sang video/deepfake trừ khi đổi định hướng đề tài.

### 4.6. Nguyên tắc so sánh với SOTA (tránh lỗi thường gặp)

- Chỉ trích số công bố trực tiếp khi chạy **đúng protocol chuẩn** (OULU-NPU Protocol 1–4, hoặc O&C&I&M với đúng split/face-crop pipeline).
- **Tự re-implement** baseline liên quan trực tiếp đến claim chính (đặc biệt Chen et al. 2020) — train lại trên đúng data/backbone của mình để cô lập đúng biến đang thay đổi.
- Với SOTA rất nặng (CLIP/LLM-based), chỉ dùng làm điểm tham chiếu, ghi rõ cột Params/FLOPs, framing dạng "hiệu năng gần với model nặng gấp 50–100 lần tham số" thay vì claim "vượt trội".
- Chạy nhiều seed (≥3), báo mean ± std cho mọi số tự train.

---

## 5. Roadmap

### Giai đoạn 1 — Hoàn thiện nền (tương đương Q3)

- [ ] Hoàn thiện lại frequency branch: thay DCT-branch hiện tại bằng K dải tần số hình khuyên cố định (K=6–8).
- [ ] Cài lại baseline Chen et al. 2020 (fixed 2-band) trên cùng backbone/data để so sánh công bằng.
- [ ] Thêm attention layer học trọng số K dải.
- [ ] Thêm JPEG augmentation vào training pipeline + tạo 4 bộ test nén sẵn (90/70/50/30).
- [ ] Chạy đủ 5 cấu hình ablation (mục 4.4) trên CelebA-Spoof, subject-disjoint split, ≥3 seed.
- [ ] Đo params/FLOPs/latency cho từng cấu hình (kế thừa từ đồ án gốc).
- [ ] Viết bản thảo đầu tiên, nhắm venue Q3.

### Giai đoạn 2 — Mở rộng generalization (nhắm Q2)

- [ ] Xin quyền truy cập OULU-NPU + CASIA-MFSD + Idiap Replay-Attack + MSU-MFSD (thường miễn phí, ký form với nhóm gốc).
- [ ] Chạy protocol O&C&I&M leave-one-out cho cấu hình tốt nhất từ Giai đoạn 1.
- [ ] Thêm phần visualization: inverse-DCT theo attention weight, đối chiếu định tính với tần số moiré/halftone lý thuyết.
- [ ] Nếu kết quả generalization tốt, cân nhắc nộp Q2.

### Giai đoạn 3 — Mở rộng độ khó (nhắm Q1, tùy chọn)

- [ ] Xin quyền truy cập SiW-Mv2 hoặc WMCA (cần thời gian xử lý thỏa thuận sử dụng dataset).
- [ ] Chạy protocol leave-one-attack-out để chứng minh generalization sang attack chưa từng thấy.
- [ ] Bổ sung so sánh với SOTA CLIP/LLM-based (trích số công bố, ghi rõ Params/FLOPs).
- [ ] Phân tích thống kê đầy đủ (mean ± std nhiều seed, kiểm định ý nghĩa nếu cải thiện nhỏ).

### Rủi ro cần theo dõi

- Nếu không xin được OULU-NPU/CASIA-MFSD/Idiap/MSU-MFSD kịp thời hạn Giai đoạn 2 → giữ nguyên CelebA-Spoof, ghi rõ cross-dataset evaluation chuẩn hóa là "Limitations/Future Work", vẫn có thể nộp Q3 vững.
- Cải thiện ACER giữa các cấu hình ablation quá nhỏ (nằm trong nhiễu random seed) → cần chạy thêm seed và báo cáo trung thực, không ép kết luận.
- Learnable/adaptive phần nào phát sinh lỗi khó debug → luôn có fallback về bản fixed-band (baseline Chen 2020 re-implement) để so sánh, không để cả pipeline phụ thuộc vào phần khó nhất.

---

## 6. Tài liệu tham khảo

Xem chi tiết BibTeX/IEEE citation tại `pad_frequency_domain_references.md` (5 paper nền: Chen et al. 2020, Huang et al. 2020, Roy et al. 2021, Kim et al. 2024, Zhang & Xiang 2020).
