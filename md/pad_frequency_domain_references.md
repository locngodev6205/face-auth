# Tài liệu Tham khảo: Vai trò của Đặc trưng Miền Tần số trong Face Presentation Attack Detection (PAD)

> **Tài liệu tham chiếu lý thuyết & thực nghiệm cho kiến trúc Dual-Branch Spatial–Frequency PAD**  
> Liên kết kiến trúc: [pad_spatial_frequency_architecture.md](file:///Users/coding/PBL6/face_auth/md/pad_spatial_frequency_architecture.md)  
> Tổng quan hệ thống: [README.md](file:///Users/coding/PBL6/README.md)

---

## 1. Đặt vấn đề & Động lực Nghiên cứu

Trong bài toán **Face Presentation Attack Detection (Face PAD)**, các mô hình Deep Learning chỉ dựa trên miền không gian (spatial-only CNNs như ResNet, MobileNet) thường đạt độ chính xác rất cao trên cùng tập dữ liệu huấn luyện (**in-domain**). Tuy nhiên, chúng gặp phải hiện tượng suy giảm hiệu năng nghiêm trọng khi đánh giá trên tập dữ liệu chưa từng thấy (**cross-dataset / cross-domain**).

### Nguyên nhân cốt lõi:
1. **Spatial CNNs dễ bị Overfit vào Domain Bias:**  
   Mạng nơ-ron tích chập truyền thống dễ khai thác các đặc trưng mức cao (high-level semantic) như màu da, phong cách ánh sáng, bối cảnh nền (background), hoặc profile quang học của camera trong tập huấn luyện thay vì học bản chất của hành vi giả mạo.
2. **Spoof Artifacts tồn tại ở dạng Vi cấu trúc (Micro-patterns):**  
   Các dấu hiệu giả mạo như vân màn hình điện tử (*moiré patterns*), cấu trúc lưới điểm in (*halftone screen dots*), phản xạ gương/kính, vệt sọc quét (*refresh-rate scan lines*) và hiện tượng răng cưa/mờ biên (*aliasing/blurring*) thường phân bố tinh tế trên các dải tần số không gian khác nhau.
3. **Giá trị của Miền Tần số (Frequency Domain):**  
   Các phép biến đổi như **Discrete Cosine Transform (DCT)** hoặc **Fast Fourier Transform (FFT)** nén năng lượng ảnh và tách biệt các tần số biến thiên không gian độc lập với ngữ nghĩa khuôn mặt, cung cấp nguồn thông tin bổ trợ (orthogonal/complementary cues) giúp mô hình chống chịu tốt hơn trước sự thay đổi domain.

---

## 2. Phân tích Chi tiết Các Công trình Tiêu biểu (Đã xác minh DOI & Link)

Dưới đây là 5 công trình nghiên cứu tiêu biểu chứng minh hiệu quả thực nghiệm và cơ sở lý thuyết của đặc trưng miền tần số trong bài toán Face PAD. Toàn bộ DOI và liên kết đã được kiểm chứng trực tiếp từ cơ sở dữ liệu học thuật chính thức (IEEE Xplore, ScienceDirect/Elsevier, Springer, PubMed Central, arXiv).

---

### Paper 1: Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability
- **Tác giả:** Baoliang Chen, Wenhan Yang, Shiqi Wang
- **Hội nghị / Xuất bản:** 2020 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR), Shenzhen, China, 2020, pp. 199–204
- **Mã định danh DOI:** `10.1109/MIPR49039.2020.00048`
- **Liên kết chính thức:**
  - DOI Resolver: [https://doi.org/10.1109/MIPR49039.2020.00048](https://doi.org/10.1109/MIPR49039.2020.00048)
  - IEEE Xplore Record: [IEEE Document #9175520](https://ieeexplore.ieee.org/document/9175520)

#### Ý tưởng & Cơ chế cốt lõi:
* Bài báo đề xuất mô hình **hai luồng (two-stream network)** để dung hợp đặc trưng tần số cao (High-Frequency - HF) và tần số thấp (Low-Frequency - LF) của ảnh khuôn mặt.
* Tác giả sử dụng 3 bộ lọc High-pass filter và Low-pass filter để bóc tách rõ ràng 2 thành phần tần số trước khi đưa vào các nhánh mạng để trích xuất đặc trưng.
* **LF Stream:** Nắm bắt cấu trúc tổng thể khuôn mặt, ánh sáng vĩ mô và hình khối.
* **HF Stream:** Tập trung vào các chi tiết biên sắc nhọn, nhiễu vi mô và các hiện tượng giả tạo bề mặt (texture artifacts).
* Hai nhánh giao tiếp với nhau qua mô-đun *Cross-Frequency Spatial Attention (CFSA)* trước khi fusion đưa vào bộ phân loại.

#### Kết quả & Đóng góp đối với đồ án:
* **Chứng minh kiến trúc Dual-Stream:** Xác nhận tính đúng đắn của thiết kế mạng 2 nhánh tách biệt (Spatial/LF và Frequency/HF) thay vì chỉ dùng một mạng nơ-ron đơn luồng.
* **Tăng cường Generalization:** Việc tách bạch tần số giúp mạng giảm thiểu phụ thuộc vào phong cách hình ảnh (style) của một camera nhất định, nâng cao khả năng tổng quát hóa trên tập dữ liệu lạ.
* **Ánh xạ vào đồ án:** Trực tiếp củng cố thiết kế nhánh `MobileNetV3` (Spatial - giữ cấu trúc khuôn mặt) kết hợp song song cùng `DCT + Tiny CNN` (Frequency - trích xuất biến thiên tần số).

---

### Paper 2: Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing
- **Tác giả:** Ying Huang, Wenwei Zhang, Jinzhuo Wang
- **Xuất bản:** arXiv preprint, arXiv:2002.03723 [cs.CV], 2020
- **Mã định danh arXiv:** `arXiv:2002.03723`
- **Liên kết chính thức:**
  - arXiv Abstract: [https://arxiv.org/abs/2002.03723](https://arxiv.org/abs/2002.03723)
  - arXiv PDF: [https://arxiv.org/pdf/2002.03723.pdf](https://arxiv.org/pdf/2002.03723.pdf)

#### Ý tưởng & Cơ chế cốt lõi:
* Đề xuất kiến trúc **FreqSpatialTemporalNet**, khai thác đồng thời 3 chiều thông tin: **Không gian (Spatial)**, **Thời gian (Temporal)**, và **Tần số (Frequency)**.
* Khác với các phương pháp spatial thuần túy, mô hình bổ sung một luồng đầu vào là **ảnh phổ tần số nhiều khung hình (multi-frame spectrum images)**.
* Giới thiệu quy trình *frequent augmentation pipeline* nhằm sinh thêm dữ liệu huấn luyện trong miền tần số, khắc phục tình trạng thiếu hụt dữ liệu giả mạo.

#### Kết quả & Đóng góp đối với đồ án:
* **Học đặc trưng tần số bằng Deep Learning:** Chứng minh rằng đưa phổ tần số vào một mạng CNN riêng biệt có thể giúp mạng tự động học được các biểu diễn đặc trưng mức cao (deep representations) có tính phân biệt mạnh mẽ giữa mặt thật và mặt giả.
* **Bổ trợ cho Spatial Branch:** Thực nghiệm trên 3 bộ dữ liệu công khai khẳng định mô hình kết hợp Spatial + Frequency vượt trội hơn hẳn mô hình chỉ chạy trên ảnh RGB thông thường.
* **Ánh xạ vào đồ án:** Hỗ trợ luận điểm rằng phổ tần số (như phổ 2D DCT) hoàn toàn có thể xem như một feature map 2D độc lập để nạp vào một Convolutional Neural Network nhẹ.

---

### Paper 3: Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra (Bi-FPNFAS)
- **Tác giả:** Koushik Roy, Md. Hasan, Labiba Rupty, Md. Sourave Hossain, Shirshajit Sengupta, Shehzad Noor Taus, Nabeel Mohammed
- **Tạp chí / Xuất bản:** MDPI *Sensors*, Tập 21, Số 8, Bài báo 2799, 2021
- **Mã định danh DOI:** `10.3390/s21082799`
- **Liên kết chính thức:**
  - DOI Resolver: [https://doi.org/10.3390/s21082799](https://doi.org/10.3390/s21082799)
  - MDPI Open Access: [https://www.mdpi.com/1424-8220/21/8/2799](https://www.mdpi.com/1424-8220/21/8/2799)
  - PubMed Central (PMC): [PMC8071536](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8071536/)

#### Ý tưởng & Cơ chế cốt lõi:
* Xây dựng mạng đa thang đo Bi-Directional Feature Pyramid Network kết hợp kỹ thuật giám sát cấp độ điểm ảnh (**pixel-wise supervision**).
* Tích hợp nhánh tự giám sát phụ (**auxiliary self-supervision branch**) yêu cầu mạng tái tạo phổ Fourier (Fourier Spectra) của ảnh đầu vào song song với việc phân loại liveness.
* Mục tiêu: Ép các tầng biểu diễn ẩn của mạng phải nhận biết và mã hóa thông tin phổ tần số của ảnh giả mạo.

#### Kết quả & Đóng góp đối với đồ án:
* **Con số thực nghiệm định lượng ấn tượng:** Đạt **ACER 2.92%** trên **Protocol IV của tập dữ liệu OULU-NPU** — đây là giao thức đánh giá khắc nghiệt nhất (kiểm thử chéo trên các loại cảm biến điện thoại và điều kiện môi trường hoàn toàn khác nhau).
* **Minh chứng vững chắc cho Cross-Dataset Generalization:** Cho thấy khi mạng được trang bị khả năng nhận biết phổ tần số, mô hình có thể thích ứng qua nhiều thiết bị thu nhận khác nhau mà không cần fine-tune.
* **Ánh xạ vào đồ án:** Đây là bằng chứng thực nghiệm quan trọng nhất để bảo vệ **Giả thuyết H2** (Frequency representation giúp giảm độ suy giảm hiệu năng khi kiểm thử cross-dataset).

---

### Paper 4: Advancing Cross-Domain Generalizability in Face Anti-Spoofing: Insights, Design, and Metrics
- **Tác giả:** Hyojin Kim, Jiyoon Lee, Yonghyun Jeong, Haneol Jang, YoungJoon Yoo
- **Hội nghị / Xuất bản:** IEEE/CVF Computer Vision and Pattern Recognition Workshops (CVPRW), 2024
- **Mã định danh arXiv:** `arXiv:2406.12258`
- **Liên kết chính thức:**
  - arXiv Abstract: [https://arxiv.org/abs/2406.12258](https://arxiv.org/abs/2406.12258)
  - arXiv PDF: [https://arxiv.org/pdf/2406.12258.pdf](https://arxiv.org/pdf/2406.12258.pdf)

#### Ý tưởng & Cơ chế cốt lõi:
* Nghiên cứu bài bản về bản chất của hiện tượng kém tổng quát hóa trong Zero-shot Domain Generalization của Face PAD.
* **Phát hiện quan trọng:** Các tín hiệu (cues) để phân biệt thật/giả chủ yếu cư trú ở **dải tần số cao vi mô (subtle high-frequency domain)**. Tuy nhiên, sự sai lệch giữa các dataset (dataset bias do ánh sáng phòng, profile cảm biến, hậu cảnh) lại chi phối mạnh miền không gian, khiến mạng tích chập thông thường bị "đánh lạc hướng".
* Đề xuất các thước đo và cơ chế thiết kế (ECLIPS framework) để tách rời domain bias khỏi các spoof cues tần số cao.

#### Kết quả & Đóng góp đối với đồ án:
* **Cơ sở lý luận bảo vệ slide / báo cáo:** Giải thích trực diện câu hỏi của hội đồng: *"Tại sao mô hình chỉ nhìn ảnh RGB toàn thể lại phân biệt kém khi đổi camera/bối cảnh?"*.
* **Lý giải nguyên nhân:** Mô hình spatial-only dễ hội tụ vào các đặc trưng "dễ học" nhưng là shortcut/spurious features (như tone màu nền). Nhánh tần số giúp cung cấp các spoof cues vi mô bền vững hơn trước thay đổi bối cảnh.
* **Ánh xạ vào đồ án:** Đóng vai trò làm luận cứ lý thuyết trung tâm cho phần *"Động lực nghiên cứu miền tần số"* trong đồ án tốt nghiệp.

---

### Paper 5: Face Anti-Spoofing Detection Based on DWT-LBP-DCT Features
- **Tác giả:** Wanling Zhang, Shijun Xiang
- **Tạp chí / Xuất bản:** *Signal Processing: Image Communication* (Elsevier / ScienceDirect), Tập 89, Tháng 11, 2020, Bài báo 115990
- **Mã định danh DOI:** `10.1016/j.image.2020.115990`
- **Liên kết chính thức:**
  - DOI Resolver: [https://doi.org/10.1016/j.image.2020.115990](https://doi.org/10.1016/j.image.2020.115990)
  - ScienceDirect Article: [ScienceDirect Article #S0923596520301533](https://www.sciencedirect.com/science/article/pii/S0923596520301533)
- *(Công trình hội nghị tiền thân: Ye Tian & Shijun Xiang, "Detection of video-based face spoofing using LBP and multiscale DCT", Springer LNCS 10082, DOI: [10.1007/978-3-319-53465-7_32](https://doi.org/10.1007/978-3-319-53465-7_32))*

#### Ý tưởng & Cơ chế cốt lõi:
* Hướng tiếp cận đặc trưng kết hợp kinh điển: Phối hợp **Discrete Wavelet Transform (DWT)**, **Local Binary Patterns (LBP)**, và **Discrete Cosine Transform (DCT)** với bộ phân loại SVM.
* DWT phân rã ảnh theo các khối tần số đa phân giải (8×8 blocks) để lọc nhiễu và cô lập dải tần; LBP trích xuất hoa văn kết cấu không gian; DCT được áp dụng để nén và biểu diễn các biến thiên động theo chiều dọc.

#### Kết quả & Đóng góp đối với đồ án:
* **Hiệu năng cực cao trên benchmark chuẩn:** Đạt Equal Error Rate (EER) lần lượt là **0.1% (0.001)** trên REPLAY-ATTACK, **3.0% (0.03)** trên CASIA-FASD, và **0.5% (0.005)** trên MSU-MFSD.
* **Khẳng định vai trò của DCT:** Biến đổi Cosin rời rạc (DCT) có khả năng cô đọng năng lượng thông tin (energy compaction) cực tốt, gom các hệ số tần số quan trọng về một góc của phổ mà không tiêu tốn tham số huấn luyện nào ($0$ trainable parameters).
* **Ánh xạ vào đồ án:** Giải thích lý do lựa chọn **2D DCT** làm bước tiền xử lý cố định cho Frequency Branch: xác định (deterministic), gọn nhẹ, không cần học, tính toán nhanh và cực kỳ phù hợp với thiết bị biên (Edge devices).

---

## 3. Bảng So sánh & Tổng hợp Các Nghiên cứu

| Công trình | Năm & Nguồn xuất bản | Công cụ Tần số | Kiến trúc / Cách tiếp cận | Benchmark & Kết quả chính | Đóng góp & Ý nghĩa với Đồ án |
|---|:---:|---|---|---|---|
| **Chen et al.** | 2020<br>*(IEEE MIPR)*<br>[DOI](https://doi.org/10.1109/MIPR49039.2020.00048) | High-pass & Low-pass Filters | Two-stream CNN (HF stream + LF stream) + CFSA Attention | Cross-database: OULU-NPU, CASIA-FASD, Replay-Attack | **Cơ sở cho Dual-Branch:** Chứng minh việc phân tách và dung hợp tần số cao/thấp cải thiện rõ rệt độ tổng quát hóa. |
| **FreqSpatialTemporalNet** | 2020<br>*(arXiv:2002.03723)*<br>[Link](https://arxiv.org/abs/2002.03723) | 2D Fourier Spectrum Images | Spatial-Temporal-Frequency multi-stream CNN | OULU-NPU, SiW, CASIA-FASD | **Spectrum CNN:** Chứng minh ảnh phổ tần số có thể học trực tiếp qua các khối tích chập độc lập. |
| **Bi-FPNFAS** | 2021<br>*(Sensors / PMC)*<br>[DOI](https://doi.org/10.3390/s21082799) | 2D Fourier Spectra | Bi-FPN + Frequency auxiliary self-supervision | OULU-NPU (**Protocol IV: ACER 2.92%**) | **Minh chứng H2 (Cross-dataset):** Số liệu thực nghiệm thuyết phục nhất cho thấy tần số giúp mô hình khái quát hóa qua nhiều sensor. |
| **Advancing Cross-Domain** | 2024<br>*(CVPRW)*<br>[Link](https://arxiv.org/abs/2406.12258) | High-Frequency Cue Analysis | Domain Disentanglement & Frequency Variance Modeling | Cross-dataset (CelebA-Spoof, SiW-Mv2) | **Cơ sở Lý thuyết cho Slide:** Giải thích tại sao domain gap nằm ở spatial semantics và tại sao spoof cues nằm ở tần số cao vi mô. |
| **Zhang & Xiang (DWT-LBP-DCT)** | 2020<br>*(Signal Processing: Image Comm.)*<br>[DOI](https://doi.org/10.1016/j.image.2020.115990) | DWT + Multiscale 2D DCT | Handcrafted Multi-resolution features + SVM | Replay-Attack (EER: 0.1%), CASIA (3.0%), MSU (0.5%) | **Lý do chọn DCT:** Minh chứng năng lực nén năng lượng và phân biệt spoofing của phép biến đổi DCT với chi phí tính toán tối thiểu. |

---

## 4. Đối chiếu với Thiết kế Kiến trúc Đồ án (SmartFace Dual-Branch PAD)

Kiến trúc PAD đề xuất trong tài liệu [pad_spatial_frequency_architecture.md](file:///Users/coding/PBL6/face_auth/md/pad_spatial_frequency_architecture.md) được xây dựng dựa trên sự kế thừa và tinh gọn từ các nghiên cứu trên:

```text
               224×224 RGB Face Image
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
  SPATIAL BRANCH                  FREQUENCY BRANCH
  MobileNetV3-Large                     │
  (ImageNet Pretrained)                 ▼
        │                             2D DCT  (Deterministic, 0 params)
        ▼                               │
  Spatial Feature Map                   ▼
        │                         Frequency Map (224×224)
        ▼                               │
  Global Avg Pooling                    ▼
        │                         Lightweight Tiny CNN
        ▼                         (Depthwise Separable Convolutions)
  256-D Spatial Vector                  │
        │                               ▼
        │                         64-D Frequency Vector
        │                               │
        └────────────────┬──────────────┘
                         ▼
                  FEATURE FUSION
              Concatenate (320-D)
                         │
                         ▼
                   MLP / Classifier
                         │
                         ▼
                    REAL / SPOOF
```

### 3 Giả thuyết Nghiên cứu (Hypotheses) được bảo đảm bởi tài liệu tham khảo:

1. **Giả thuyết H1 (Thông tin Bổ trợ - Complementary Evidence):**
   * *Nghiên cứu bảo trợ:* Chen et al. (2020) và FreqSpatialTemporalNet (2020).
   * *Luận cứ:* Nhánh MobileNetV3 học hình thái không gian khuôn mặt (256-D), trong khi nhánh Tiny CNN học đặc trưng phân bố tần số (64-D). Hai luồng thông tin này có tính chất trực giao, bù trừ khuyết điểm cho nhau.

2. **Giả thuyết H2 (Cải thiện Tổng quát hóa Cross-Dataset):**
   * *Nghiên cứu bảo trợ:* Bi-FPNFAS (2021) và Kim et al. (CVPRW 2024).
   * *Luận cứ:* Khi kiểm thử từ `CelebA-Spoof` sang `OULU-NPU` hoặc `SiW`, các bias về ánh sáng phòng và background ở nhánh Spatial sẽ ít làm sụp đổ toàn bộ mô hình vì nhánh DCT tập trung vào các vi cấu trúc vật lý của phương tiện tấn công (màn hình, giấy in).

3. **Giả thuyết H3 (Tối ưu Triển khai Biên - Edge Friendly Efficiency):**
   * *Nghiên cứu bảo trợ:* Zhang & Xiang (2020) và nguyên lý Depthwise Separable Convolution.
   * *Luận cứ:* 2D DCT là phép biến đổi toán học thuần túy (không tốn tham số), kết hợp cùng Tiny CNN chỉ gồm 2 block Depthwise Separable Conv giúp nhánh tần số chỉ tốn thêm một lượng rất nhỏ FLOPs và tham số, đảm bảo latency thời gian thực cho hệ thống điểm danh AIoT.

---

## 5. Trích dẫn Mẫu Phục vụ Báo cáo & Khóa luận

### Định dạng IEEE (Dùng cho bài báo / báo cáo kỹ thuật)
```text
[1] B. Chen, W. Yang, and S. Wang, "Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability," in Proc. IEEE Conf. Multimedia Information Processing and Retrieval (MIPR), Shenzhen, China, 2020, pp. 199–204, doi: 10.1109/MIPR49039.2020.00048.
[2] Y. Huang, W. Zhang, and J. Wang, "Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing," arXiv preprint arXiv:2002.03723, 2020. [Online]. Available: https://arxiv.org/abs/2002.03723.
[3] K. Roy, M. Hasan, L. Rupty, M. S. Hossain, S. Sengupta, S. N. Taus, and N. Mohammed, "Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra," Sensors, vol. 21, no. 8, Art. no. 2799, Apr. 2021, doi: 10.3390/s21082799.
[4] H. Kim, J. Lee, Y. Jeong, H. Jang, and Y. Yoo, "Advancing Cross-Domain Generalizability in Face Anti-Spoofing: Insights, Design, and Metrics," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2024, arXiv:2406.12258. [Online]. Available: https://arxiv.org/abs/2406.12258.
[5] W. Zhang and S. Xiang, "Face anti-spoofing detection based on DWT-LBP-DCT features," Signal Processing: Image Communication, vol. 89, Art. no. 115990, Nov. 2020, doi: 10.1016/j.image.2020.115990.
```

### Định dạng BibTeX (Dùng cho LaTeX)
```bibtex
@inproceedings{chen2020face,
  title={Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability},
  author={Chen, Baoliang and Yang, Wenhan and Wang, Shiqi},
  booktitle={2020 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)},
  pages={199--204},
  year={2020},
  organization={IEEE},
  doi={10.1109/MIPR49039.2020.00048},
  url={https://doi.org/10.1109/MIPR49039.2020.00048}
}

@article{huang2020deep,
  title={Deep Frequent Spatial Temporal Learning for Face Anti-Spoofing},
  author={Huang, Ying and Zhang, Wenwei and Wang, Jinzhuo},
  journal={arXiv preprint arXiv:2002.03723},
  year={2020},
  url={https://arxiv.org/abs/2002.03723}
}

@article{roy2021bi,
  title={Bi-Directional Feature Pyramid Network for Pixel-Wise Face Anti-Spoofing by Leveraging Fourier Spectra},
  author={Roy, Koushik and Hasan, Md and Rupty, Labiba and Hossain, Md Sourave and Sengupta, Shirshajit and Taus, Shehzad Noor and Mohammed, Nabeel},
  journal={Sensors},
  volume={21},
  number={8},
  pages={2799},
  year={2021},
  publisher={MDPI},
  doi={10.3390/s21082799},
  url={https://doi.org/10.3390/s21082799}
}

@inproceedings{kim2024advancing,
  title={Advancing Cross-Domain Generalizability in Face Anti-Spoofing: Insights, Design, and Metrics},
  author={Kim, Hyojin and Lee, Jiyoon and Jeong, Yonghyun and Jang, Haneol and Yoo, YoungJoon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024},
  url={https://arxiv.org/abs/2406.12258}
}

@article{zhang2020face,
  title={Face anti-spoofing detection based on DWT-LBP-DCT features},
  author={Zhang, Wanling and Xiang, Shijun},
  journal={Signal Processing: Image Communication},
  volume={89},
  pages={115990},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.image.2020.115990},
  url={https://doi.org/10.1016/j.image.2020.115990}
}
```
