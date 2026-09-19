# Lightweight Spatial–Frequency Face Anti-Spoofing (PAD)


## 0. Định vị nghiên cứu theo literature 2024–2026

Thiết kế của project thuộc nhóm **single-frame RGB Face PAD có dual-domain representation**:

```text
RGB face
 ├─ Spatial branch: MobileNetV3-Large
 └─ Frequency branch: DCT + lightweight CNN
          ↓
      Feature Fusion
          ↓
      REAL / SPOOF
```

Điểm khác biệt cần giữ rõ trong báo cáo:

- không xem frequency branch là một classifier độc lập;
- frequency cue là **complementary evidence** cho spatial cue;
- mục tiêu không chỉ là giảm lỗi in-domain mà còn kiểm tra **cross-domain generalization**;
- kiến trúc phải đủ nhẹ để đo latency/FPS và hướng tới edge deployment;
- **V1 dùng concatenation** để kiểm chứng câu hỏi nghiên cứu cơ bản;
- **V2 dùng adaptive/gated fusion** chỉ sau khi V1 chứng minh frequency branch thực sự có giá trị.

### Vị trí so với các hướng Face PAD hiện đại

| Nhóm phương pháp | Ví dụ tiêu biểu | Điểm mạnh | Hạn chế / khoảng trống liên quan project |
|---|---|---|---|
| Texture / lightweight CNN | MiniFASNet, DeepPixBiS | Nhẹ, dễ deploy | Có thể phụ thuộc domain/camera |
| Auxiliary supervision | CDCN, depth/material supervision | Học spoof cue chi tiết hơn | Training phức tạp hơn |
| Domain Generalization | SSDG, FSDA, frequency-shortcut methods | Tập trung unseen domain | Thường tăng độ phức tạp training |
| Frequency-aware | Fourier/DCT/Wavelet, FSDA, Oculus | Khai thác periodic/micro-texture cues | Cần chứng minh frequency cue bổ trợ thật sự |
| Transformer / Foundation / VLM | ViT, CLIP-based UAD/FAS | Representation mạnh, multi-task | Nặng hơn, khó phù hợp edge |
| Multi-modal | RGB + Depth/IR | Robust hơn trong nhiều điều kiện | Cần sensor bổ sung |

**Định vị của nhóm:** lightweight RGB-only, dual-domain spatial–frequency, ưu tiên ablation rõ ràng và edge efficiency hơn việc dùng backbone rất lớn.

### Ba công trình gần nhất đặc biệt liên quan

1. **Ali et al., Scientific Reports 2026** — dual-branch spatial + Fourier-domain PAD trên Raspberry Pi 3B+, rất gần về triết lý **lightweight + frequency + edge**.
2. **Niu & Lin, Neural Networks 2026** — Frequency Adaptive Enhancement, tích hợp động nhiều dải tần để học feature bền vững hơn theo domain.
3. **de Dravo et al., ICCV Workshops 2025 (Oculus)** — spatial backbone + frequency branch, **feature concatenation rồi post-fusion attention**, là reference rất gần cho roadmap `Concat → Adaptive/Gated Fusion`.

> Không nên tuyên bố kiến trúc của nhóm mới hoàn toàn chỉ vì dùng Spatial + Frequency. Contribution hợp lý hơn là **một thiết kế lightweight có ablation chặt chẽ, đánh giá cross-domain và tiến hóa fusion từ concat sang adaptive gate nếu dữ liệu chứng minh cần thiết**.

---

## 1. Mục tiêu

Thiết kế một mô hình Face Presentation Attack Detection (PAD) nhẹ, xử lý ảnh RGB single-frame, có khả năng chạy trên edge device và có một thành phần nghiên cứu rõ ràng: bổ sung đặc trưng miền tần số vào đặc trưng miền không gian.

Mục tiêu thực nghiệm chính:

1. Xây dựng **Spatial-only baseline** bằng MobileNetV3-Large.
2. Xây dựng **Frequency-aware model** bằng cách bổ sung một nhánh DCT + CNN nhẹ.
3. Huấn luyện hai mô hình **độc lập** để comparison công bằng.
4. Kiểm tra liệu frequency branch có cải thiện **cross-domain generalization** hay không.
5. Đo thêm chi phí triển khai: số tham số, FLOPs và latency.

> Đây là proposed architecture. Không giả định trước rằng frequency branch chắc chắn cải thiện kết quả; đó là giả thuyết cần được kiểm chứng bằng ablation và cross-dataset evaluation.

---

## 2. Kiến trúc tổng thể

```text
                         FACE IMAGE
                           224×224
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
       SPATIAL BRANCH                    FREQUENCY BRANCH
       MobileNetV3-Large                       │
       ImageNet pretrained                     ▼
             │                                2D DCT
             ▼                                  │
      Feature Extraction                       ▼
             │                            Frequency Map
             ▼                                  │
       Spatial Feature                         ▼
          256-D                        Lightweight CNN
                                                │
                                                ▼
                                        Frequency Feature
                                             64-D
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                       FEATURE FUSION
                         Concatenate
                            320-D
                              │
                              ▼
                          MLP / FC
                           128-D
                              │
                              ▼
                         PAD CLASSIFIER
                              │
                       REAL / SPOOF
```

### Các thành phần

| Thành phần | Vai trò | Trainable? |
|---|---|---:|
| MobileNetV3-Large | Trích xuất spatial features | Có, fine-tune |
| 2D DCT | Chuyển ảnh sang miền tần số | Không |
| Lightweight Frequency CNN | Học frequency features | Có |
| Feature Fusion | Kết hợp hai nguồn đặc trưng | Có |
| MLP / FC | Phân loại | Có |

DCT là phép biến đổi xác định (deterministic transform), không có trọng số học được.

---

## 3. Tại sao thêm miền tần số?

Một CNN trong miền không gian có thể học:

- facial structure;
- edges;
- texture;
- illumination;
- local patterns.

Tuy nhiên, PAD còn quan tâm đến các dấu hiệu low-level do quá trình tái tạo khuôn mặt tạo ra. Ví dụ:

- texture của giấy và mực in;
- pattern tuần hoàn từ màn hình;
- pixel/subpixel structure;
- moiré và aliasing;
- một số artifact do compression hoặc reproduction.

Những dấu hiệu này có thể được biểu diễn bổ sung trong frequency domain.

### Ý nghĩa của spatial frequency

Spatial frequency mô tả mức độ nhanh/chậm mà cường độ pixel thay đổi theo không gian.

- **Low frequency:** thay đổi chậm; thường liên quan đến cấu trúc lớn, illumination và vùng mượt.
- **Mid/high frequency:** thay đổi nhanh; thường liên quan đến edge, fine texture và pattern nhỏ/tuần hoàn.

Điểm cần tránh:

> Không được giả định “high frequency = spoof”.

Khuôn mặt thật cũng có high-frequency information. Mục tiêu của frequency branch là học **các pattern phân biệt trên nhiều dải tần số**, không phải chỉ đo lượng high-frequency.

---

## 4. Frequency Branch chi tiết

### 4.1. Input

Input PAD nên là face crop ở kích thước:

```text
224 × 224 × 3
```

Ở phiên bản đầu tiên, sử dụng **face crop đúng theo bounding box**, chưa cần expanded crop.

### 4.2. Chuyển sang luminance / grayscale

Frequency branch có thể bắt đầu từ grayscale hoặc kênh luminance để giảm phụ thuộc vào color semantics:

```text
RGB Face
   ↓
Luminance / Grayscale
   ↓
224 × 224
```

Nên coi đây là một lựa chọn thực nghiệm, không phải yêu cầu bắt buộc. Có thể ablate giữa DCT trên grayscale và DCT trên RGB nếu còn thời gian.

### 4.3. 2D DCT

Áp dụng 2D DCT:

```text
224 × 224 luminance
        ↓
      2D DCT
        ↓
224 × 224 frequency coefficients
```

DCT không làm mất hoàn toàn thông tin hình ảnh theo kiểu “xóa pixel”; nó biểu diễn lại thông tin theo các thành phần tần số.

### 4.4. Dynamic range / normalization

DCT coefficients có thể có dynamic range lớn. Một pipeline thực tế có thể dùng:

```text
C = DCT(image)
C' = sign(C) * log(1 + |C|)
C_norm = normalize(C')
```

Hoặc dùng một normalization ổn định tương đương.

Mục tiêu là giúp CNN phía sau dễ tối ưu hơn.

### 4.5. Lightweight Frequency CNN

Không cần một backbone lớn. Một phương án ban đầu:

```text
Frequency Map
     ↓
3×3 Depthwise Conv
     ↓
1×1 Pointwise Conv
     ↓
Activation + BatchNorm
     ↓
3×3 Depthwise Conv
     ↓
1×1 Pointwise Conv
     ↓
Global Average Pooling
     ↓
Linear
     ↓
64-D Frequency Feature
```

Có thể dùng 32 → 64 channels, tùy thiết bị.

Mục tiêu là giữ frequency branch rất nhỏ để phần lớn computation vẫn nằm ở MobileNetV3.

---

## 5. Spatial Branch

### Backbone đề xuất

**MobileNetV3-Large, ImageNet pretrained**.

Lý do:

- phù hợp edge/mobile;
- đủ capacity cho baseline PAD;
- có pretrained weights tốt;
- dễ export ONNX/TensorRT/TFLite;
- giữ được một backbone phổ biến để làm baseline.

### Flow

```text
224×224×3
    ↓
MobileNetV3-Large
    ↓
Feature map
    ↓
Global Average Pooling
    ↓
Projection Layer
    ↓
256-D Spatial Feature
```

Projection layer có thể là:

```python
nn.Linear(backbone_dim, 256)
```

Sau đó có thể thêm BatchNorm hoặc LayerNorm nếu thực nghiệm cho thấy hữu ích.

---

## 6. Feature Fusion

### 6.1. V1 — Concatenation Fusion (main experiment đầu tiên)

Đây vẫn là lựa chọn **bắt buộc nên làm trước** vì đơn giản, dễ giải thích và tạo ablation sạch.

```text
Spatial Feature      256-D
Frequency Feature     64-D
        │
        └──── Concatenate ────┐
                              ▼
                           320-D
                              ↓
                           MLP
                              ↓
                           128-D
                              ↓
                         Classifier
```

```python
fused = torch.cat([spatial_feat, freq_feat], dim=1)
logits = classifier(fused)
```

Câu hỏi nghiên cứu V1:

> **Explicit frequency features có mang lại thông tin bổ sung cho spatial representation hay không?**

Nếu E3 (Spatial + Frequency + Concat) không cải thiện ổn định so với E1 (Spatial-only), chưa có lý do khoa học để chuyển ngay sang fusion phức tạp.

---

### 6.2. V2 — Gated Fusion / Adaptive Fusion

Sau khi V1 được kiểm chứng, có thể phát triển V2 để trả lời câu hỏi sâu hơn:

> **Đóng góp của spatial và frequency cue có nên thay đổi theo từng input hay không?**

Ví dụ:
- ảnh thiếu sáng có thể làm spatial texture kém tin cậy;
- replay trên màn hình có thể làm frequency cue rõ hơn;
- ảnh thật có texture sắc nét có thể khiến spatial cue đủ mạnh;
- một số frequency bands có thể chứa noise thay vì spoof cue.

Do đó, thay vì luôn ghép hai vector với trọng số ngầm cố định, V2 học một **gate** điều chỉnh mức đóng góp.

#### Kiến trúc đề xuất

Do hai branch đang có số chiều khác nhau (`256-D` và `64-D`), trước tiên project về cùng latent dimension:

```text
Spatial 256-D ── Linear + Norm ──► s' ∈ R^128
Frequency 64-D ─ Linear + Norm ──► f' ∈ R^128
                                      │
                   ┌──────────────────┘
                   ▼
              concat[s', f']
                   │
                 MLP
                   │
               sigmoid
                   │
              gate g ∈ R^128
                   │
                   ▼
     h = g ⊙ s' + (1 - g) ⊙ f'
                   │
                   ▼
               Classifier
```

Công thức:

\[
s' = P_s(s), \qquad f' = P_f(f)
\]

\[
g = \sigma(\mathrm{MLP}([s'; f']))
\]

\[
h = g \odot s' + (1-g)\odot f'
\]

Trong đó:
- `g → 1`: ưu tiên spatial cue;
- `g → 0`: ưu tiên frequency cue;
- gate theo vector cho phép từng latent channel có mức kết hợp khác nhau.

#### PyTorch skeleton

```python
class GatedFusion(nn.Module):
    def __init__(self, spatial_dim=256, freq_dim=64, hidden_dim=128):
        super().__init__()

        self.s_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.f_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, spatial_feat, freq_feat):
        s = self.s_proj(spatial_feat)
        f = self.f_proj(freq_feat)

        g = self.gate(torch.cat([s, f], dim=1))
        fused = g * s + (1.0 - g) * f

        return self.classifier(fused), g
```

### 6.3. Vì sao V2 hợp lý sau V1?

Literature gần đây cho thấy adaptive fusion là một hướng tự nhiên:

- **Oculus (ICCVW 2025):** concatenation spatial + frequency, sau đó dùng **post-fusion SE attention**.
- **Niu & Lin (Neural Networks 2026):** Frequency Adaptive Enhancement Module động tích hợp nhiều dải tần.
- **DEFuseNet (Neurocomputing 2026):** dùng **modulated fusion** để thích nghi đóng góp của RGB và texture cue theo input.

Các paper trên không đồng nghĩa gated fusion của project chắc chắn tốt hơn concat. Chúng chỉ cung cấp **motivation hợp lý để đặt V2 như một ablation tiếp theo**.

### 6.4. Rủi ro của gated fusion

- Gate có thể collapse, luôn ưu tiên một branch.
- Tăng số tham số và nguy cơ overfit.
- Nếu frequency branch chưa học cue hữu ích, gate chỉ che giấu vấn đề thay vì giải quyết.
- Khó giải thích hơn concatenation.

Do đó cần log thêm:

```text
mean(g)
std(g)
gate distribution REAL vs SPOOF
gate distribution in-domain vs cross-domain
```

và luôn so sánh với concat dưới cùng protocol.

---

## 7. Classifier và loss

Output:

```text
REAL / SPOOF
```

Đây là binary classification.

### Loss khuyến nghị ban đầu

```text
Weighted Binary Cross-Entropy
```

Trong PyTorch có thể dùng:

```python
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

hoặc dùng Cross Entropy 2-class. Chọn một cách và giữ cố định giữa baseline/proposed.

---

## 8. Training protocol

### Dataset chính

**CelebA-Spoof**.

Dataset có khoảng 625K ảnh và hơn 10K subjects, gồm real và nhiều dạng presentation attack. Các attack category cần lưu ý trong proposal gồm Photo, Poster, A4, Phone, PC, Pad, 3D Mask, Face Mask, Upper Body Mask và Region Mask.

### Split

Ưu tiên **subject-disjoint split**:

```text
TRAIN subjects ∩ TEST subjects = ∅
```

Không random split theo ảnh nếu protocol của experiment yêu cầu subject-independent evaluation.

### Input

```text
224 × 224
```

### Augmentation

Có thể bắt đầu bằng:

- Random horizontal flip;
- brightness/contrast nhẹ;
- resize/crop;
- blur nhẹ;
- JPEG compression.

Không nên augmentation quá mạnh ở giai đoạn đầu vì có thể tự tạo artifact tương tự spoof hoặc làm frequency branch học noise không mong muốn.

---

## 9. Cách train baseline và proposed model

Hai model phải được **train độc lập** để comparison công bằng.

### Experiment A — Spatial-only baseline

```text
Face Image
    ↓
MobileNetV3-Large
    ↓
256-D Spatial Feature
    ↓
MLP / Classifier
    ↓
REAL / SPOOF
```

### Experiment B — Frequency-aware model

```text
                     Face Image
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
       MobileNetV3-Large             DCT
              ↓                       ↓
        Spatial 256-D          Tiny Frequency CNN
                                      ↓
                                  Frequency 64-D
              └───────────┬───────────┘
                          ↓
                       Fusion
                          ↓
                       Classifier
                          ↓
                     REAL / SPOOF
```

### Fair comparison

Hai experiment phải giữ nguyên:

- dataset split;
- input resolution;
- augmentation policy;
- optimizer;
- training schedule;
- evaluation code;
- decision threshold calibration procedure.

Khác biệt chính cần kiểm chứng là **sự xuất hiện của frequency branch**.

---

## 10. Training schedule đề xuất

Cấu hình ban đầu có thể là:

```text
Backbone: MobileNetV3-Large ImageNet pretrained
Optimizer: AdamW
Initial LR: 1e-4
Batch size: 32–64
Epochs: 20–40
Scheduler: Cosine Annealing
Early stopping: Yes
```

Có thể dùng differential learning rates:

```text
MobileNetV3 backbone: 1e-5
New frequency/fusion/classifier layers: 1e-4
```

Nếu training không ổn định, có thể freeze backbone vài epoch đầu rồi unfreeze.

Quan trọng: phải dùng **cùng protocol** cho baseline và proposed model.

---

## 11. Ablation study

Đây là phần chứng minh contribution quan trọng nhất.

### 11.1. Bộ experiment khuyến nghị

| ID | Spatial | Frequency | Fusion | Câu hỏi nghiên cứu |
|---|:---:|:---:|---|---|
| **E1** | ✓ | — | — | Spatial-only baseline mạnh đến đâu? |
| **E2** | — | ✓ | — | Frequency cue tự thân có discriminative power không? |
| **E3** | ✓ | ✓ | **Concat** | Frequency có bổ sung thông tin cho spatial không? |
| **E4** | ✓ | ✓ | **Gated** | Adaptive weighting có tốt hơn concat cố định không? |

### 11.2. Thứ tự thực hiện

```text
E1 Spatial baseline
      ↓
E3 Spatial + Frequency + Concat
      ↓
Nếu E3 có lợi ích ổn định
      ↓
E4 Spatial + Frequency + Gated Fusion
```

E2 rất hữu ích cho phân tích nhưng không được dùng để thay cho E1/E3.

### 11.3. Fair comparison

Tất cả E1–E4 phải giữ nguyên:

- dataset split;
- input resolution;
- face crop policy;
- augmentation;
- optimizer;
- scheduler;
- số epoch / early stopping;
- evaluation code;
- threshold calibration;
- random seeds nếu có thể.

Khác biệt phải được cô lập ở branch/fusion cần ablate.

### 11.4. Báo cáo thống kê

Nếu tài nguyên cho phép, chạy ít nhất 3 seeds và báo:

```text
mean ± std
```

Điều này đặc biệt quan trọng nếu cải thiện ACER/HTER chỉ ở mức nhỏ.

---

## 12. Evaluation protocol

Không nên chỉ báo `Accuracy`.

### 12.1. PAD metrics cốt lõi

#### APCER — Attack Presentation Classification Error Rate

Tỷ lệ presentation attack bị hệ thống chấp nhận nhầm là bona fide / real.

> Trong bối cảnh authentication, đây là lỗi an ninh quan trọng.

#### BPCER — Bona Fide Presentation Classification Error Rate

Tỷ lệ người thật bị từ chối nhầm là spoof.

> Đây là lỗi ảnh hưởng usability.

#### ACER

\[
ACER = \frac{APCER + BPCER}{2}
\]

ACER phổ biến trong các protocol PAD như OULU-NPU và phù hợp để chọn/checkpoint nếu protocol nghiên cứu dùng metric này.

---

### 12.2. Cross-domain / Domain-Generalization metrics

Trong literature Domain-Generalizable FAS, đặc biệt các protocol kiểu **OCIM** (OULU-NPU, CASIA-FASD, Idiap Replay-Attack, MSU-MFSD), các metric thường gặp là:

- **HTER — Half Total Error Rate**;
- **AUC — Area Under ROC Curve**.

Do đó nên có hai tầng đánh giá:

```text
A. In-domain / dataset protocol
   APCER, BPCER, ACER, AUC

B. Cross-domain / leave-one-domain-out
   HTER, AUC
```

Nếu dùng `CelebA-Spoof → OULU-NPU / SiW`, hãy gọi rõ đây là **cross-dataset stress test của project**, không đồng nhất với OCIM benchmark chuẩn.

---

### 12.3. Threshold calibration

Không tune threshold trên test.

Pipeline:

```text
TRAIN
  ↓
VALIDATION
  ├─ select checkpoint
  └─ calibrate threshold
          ↓
       LOCK threshold
          ↓
TEST / CROSS-DATASET
```

Nếu so sánh E1–E4, procedure calibration phải giống nhau.

---

### 12.4. Efficiency metrics

Vì project có mục tiêu edge/lightweight, accuracy alone là chưa đủ.

Báo thêm:

```text
Parameters
FLOPs / MACs
Model size
Peak memory
Inference latency
FPS
```

Nếu có edge target thật, đo trên chính target đó hoặc ít nhất một CPU reference.

---

### 12.5. Khi nào được so sánh trực tiếp với paper khác?

Chỉ gọi là **direct comparison** nếu các yếu tố chính tương thích:

- cùng dataset;
- cùng protocol/split;
- cùng loại input/modalities;
- cùng định nghĩa metric;
- cùng chính sách threshold;
- không fine-tune trên target trong cross-domain test.

Nếu khác protocol, chỉ trình bày:

> **Reference result / contextual comparison — not directly comparable.**

Đây là điểm rất quan trọng khi đưa số liệu literature lên slide.

---

### 12.6. Ba reference result nên dùng làm mốc thảo luận

| Paper | Evaluation nổi bật | Cách dùng trong báo cáo |
|---|---|---|
| **Ali et al., Sci. Rep. 2026** | Private edge dataset; APCER **1.2%**, BPCER **1.6%**, Raspberry Pi 3B+ | Gần architecture/edge goal nhất, nhưng **không direct compare** vì private dataset và paper không có spatial-only ablation định lượng |
| **Oculus, ICCVW 2025** | Unified physical/digital challenge; official ACER **20.14%** | Reference cho spatial+frequency fusion + post-fusion attention; protocol khác project |
| **Cao & Ma, WACV 2025** | Generalized FAS; paper báo HTER/AUC trên cross-domain benchmarks | Reference tốt cho lập luận “frequency shortcut” và cross-domain evaluation |

Điểm quan trọng hơn việc “thắng paper” ở giai đoạn này là **đặt experiment của nhóm vào đúng protocol để sau đó có thể so sánh hợp lệ**.

---

## 13. Research hypothesis

Không nên claim trước rằng accuracy chắc chắn tăng.

### H1

> Explicit frequency-domain features cung cấp thông tin bổ sung cho spatial representation trong Face PAD.

### H2

> Nếu có lợi ích, lợi ích đó có thể thể hiện rõ hơn ở cross-dataset generalization so với in-domain test.

### H3

> Frequency branch nhỏ giúp tăng chi phí tính toán ở mức thấp và vẫn phù hợp với edge deployment.

### Expected qualitative result

Pattern mong muốn:

```text
                    Baseline      + Frequency
In-domain             tốt            tốt / tốt hơn nhẹ
Cross-domain         giảm            giảm ít hơn
Model cost            thấp           tăng nhẹ
```

Đây là **expected behavior**, không phải kết quả đã biết trước.

---

## 14. Theo dõi efficiency

Sau khi train, benchmark cả hai model.

### Nên đo

```text
Parameters
FLOPs / MACs
Model size
Peak memory
Inference latency
FPS
```

Đo riêng:

```text
MobileNetV3 only
vs
MobileNetV3 + Frequency Branch
```

Latency nên đo sau khi model warm-up và lấy trung bình trên nhiều lần inference.

---

## 15. Edge deployment

Mục tiêu không phải chỉ có accuracy cao mà là:

```text
Security
   ×
Generalization
   ×
Efficiency
```

Sau khi model ổn định, có thể export:

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime / TensorRT / TFLite
```

Quantization INT8 có thể là bước sau, nhưng **không cần đưa vào research scope đầu tiên**.

---

## 16. PyTorch skeleton

### Frequency branch

```python
import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.proj(x)
```

Đây chỉ là skeleton. Phần DCT preprocessing có thể được viết dưới dạng preprocessing function hoặc module không trainable.

### Fusion model

```python
class FrequencyAwarePAD(nn.Module):
    def __init__(self, spatial_dim=256, freq_dim=64):
        super().__init__()

        self.frequency_branch = FrequencyBranch(freq_dim)

        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, spatial_feat, frequency_input):
        freq_feat = self.frequency_branch(frequency_input)
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fusion(fused)
```

Phần MobileNetV3 backbone cần được nối vào để tạo `spatial_feat`.

---

## 17. Data flow trong training

```text
                Image
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
       RGB Face       Luminance Face
          │                │
          ▼                ▼
    MobileNetV3          DCT
          │                │
          ▼                ▼
    Spatial 256-D    Frequency Map
                           │
                           ▼
                      Tiny CNN
                           │
                           ▼
                     Frequency 64-D
          │                │
          └───────┬────────┘
                  ▼
               Concat
                  │
                  ▼
              320-D vector
                  │
                  ▼
               MLP / FC
                  │
                  ▼
              REAL / SPOOF
```

---

## 18. Pseudocode training loop

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.float().to(device)

    # Spatial branch
    spatial_feat = spatial_backbone(images)

    # Frequency input
    luminance = to_luminance(images)
    freq_map = dct_transform(luminance)
    freq_map = normalize_frequency(freq_map)

    # Frequency branch
    freq_feat = frequency_branch(freq_map)

    # Fusion
    fused = torch.cat([spatial_feat, freq_feat], dim=1)
    logits = classifier(fused).squeeze(1)

    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Baseline training chỉ dùng spatial branch và classifier.

---

## 19. Các lỗi thiết kế cần tránh

### 1. Không đưa raw DCT vector trực tiếp vào classifier

Không nên:

```text
Image → DCT → Flatten → Huge FC → Classifier
```

Nên:

```text
Image → DCT → Tiny CNN → compact feature → Fusion
```

### 2. Không cho frequency branch quyết định một mình

Frequency information là complementary evidence.

### 3. Không train baseline bằng checkpoint của proposed model

Baseline và proposed model phải được train độc lập khi làm comparison chính.

### 4. Không random image split nếu có nguy cơ leakage

Ưu tiên subject-disjoint.

### 5. Không chỉ báo accuracy

PAD nên báo APCER, BPCER và ACER.

### 6. Không chỉ đánh giá trên CelebA-Spoof

Nếu mục tiêu là generalization, cross-dataset evaluation rất quan trọng.

---

## 20. Scope khuyến nghị cho project

### Phiên bản chính

```text
MobileNetV3-Large
        +
       DCT
        +
Tiny Frequency CNN
        +
Concatenation Fusion
        ↓
REAL / SPOOF
```

### Không đưa vào phiên bản đầu

- Depth
- IR
- Temporal liveness
- Transformer lớn
- multimodal fusion
- domain adaptation
- LoRA
- gated fusion trước khi concat baseline được kiểm chứng
- quantization research

Những thứ này có thể là future work.

---

## 21. Câu chuyện research hoàn chỉnh

### Baseline

> MobileNetV3-Large được sử dụng như một lightweight spatial-only PAD baseline.

### Vấn đề

> Spatial representations có thể bỏ sót hoặc không khai thác rõ ràng một số low-level frequency cues và có thể chịu ảnh hưởng bởi domain shift.

### Đề xuất

> Bổ sung một Frequency Branch sử dụng DCT và CNN nhẹ để học frequency representation bổ sung cho spatial features.

### Kiểm chứng

> Train baseline và proposed model độc lập, sau đó thực hiện ablation và cross-dataset evaluation.

### Tiêu chí thành công

> Cải thiện hoặc duy trì PAD performance, đặc biệt trên cross-domain evaluation, trong khi computational overhead vẫn thấp và model có khả năng triển khai trên edge.

---

## 22. Architecture đưa vào presentation: V1 hiện tại và V2 roadmap

### V1 — Main proposed model

```text
                      224×224 FACE
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       MobileNetV3-Large                 DCT
             │                           │
             ▼                           ▼
      Spatial Feature             Frequency Map
          256-D                        │
                                      ▼
                                  Tiny CNN
                                      │
                                      ▼
                              Frequency Feature
                                   64-D
             │                           │
             └─────────────┬─────────────┘
                           ▼
                       Concatenate
                           │
                          320-D
                           │
                           ▼
                          MLP
                           │
                           ▼
                     REAL / SPOOF
```

### V2 — Research extension sau khi V1 được kiểm chứng

```text
Spatial 256-D ──► Projection 128-D ──┐
                                     ├─► Learned Gate ─► Weighted Fusion ─► PAD
Frequency 64-D ─► Projection 128-D ──┘
```

V2 không thay thế V1 trong experiment chính; nó là bước tiến hóa tự nhiên để kiểm tra **adaptive fusion**.

### Thông điệp cuối

> **Spatial features tell the model what the face looks like; frequency features provide complementary information about how image patterns vary across space.**

Trong project này, frequency branch là một giả thuyết nghiên cứu có thể kiểm chứng, không phải một đảm bảo rằng mọi spoof attack đều có đặc trưng tần số riêng.

---


## 23. Nội dung cần đưa lên slide để đáp ứng yêu cầu báo cáo

### Slide A — Các phương pháp học máy hiện đại cho Face PAD

Nên thay bảng “các backbone” đơn thuần bằng taxonomy:

| Hướng | Ví dụ | Cue chính | Nhận xét |
|---|---|---|---|
| Lightweight CNN | MiniFASNet | texture/spatial | nhanh, edge-friendly |
| Auxiliary supervision | CDCN / depth-like supervision | local spoof cue | tốt nhưng training phức tạp |
| Domain Generalization | SSDG / FSDA / Frequency Shortcut | domain-invariant cues | phù hợp bài toán unseen domain |
| Frequency-aware | DCT/Fourier/Wavelet, Oculus | periodic/micro-texture | gần hướng nhóm nhất |
| Transformer / VLM | ViT / CLIP-based methods | global/semantic cues | mạnh nhưng nặng |
| Multi-modal | RGB + IR/Depth | complementary sensors | robust nhưng cần phần cứng bổ sung |

### Slide B — Phương pháp nhóm & nhận xét

```text
Spatial-only baseline
MobileNetV3
        ↓
+ DCT Frequency Branch
        ↓
Concat Fusion (V1)
        ↓
Gated Fusion (V2, planned)
```

Nhận xét:

- nhẹ hơn ViT/VLM;
- không cần IR/depth;
- explicit frequency cue dễ ablate;
- cần chứng minh cross-domain benefit;
- concat là baseline khoa học sạch;
- gated fusion chỉ hợp lý nếu concat đã cho thấy frequency cue hữu ích.

### Slide C — Bài báo gần nhất gần với hướng nhóm

**Ali et al., Scientific Reports, 2026 — “AI-enabled smart surveillance system for secure monitoring and authentication.”**

Điểm gần:
- dual-branch spatial + frequency;
- lightweight PAD;
- edge deployment;
- Fourier-domain cue.

Điểm project có thể làm chặt hơn:
- public benchmark;
- spatial-only vs frequency-aware ablation;
- cross-dataset test;
- fair comparison cùng split/protocol;
- V1 concat và V2 adaptive gate.

Có thể đặt thêm 2 paper nhỏ bên cạnh:

- **Niu & Lin, Neural Networks 2026:** adaptive multi-frequency enhancement.
- **Oculus, ICCVW 2025:** concat + post-fusion attention, rất phù hợp để dẫn sang gated fusion.

### Slide D — Evaluation Protocol & Literature Comparison

Bắt buộc ghi:

```text
PAD: APCER / BPCER / ACER
Cross-domain: HTER / AUC
Edge: Params / FLOPs / Latency / FPS
```

Và một dòng:

> Chỉ so sánh trực tiếp khi cùng dataset + protocol; số từ paper khác protocol chỉ dùng làm reference.

---


## 24. Tài liệu tham khảo liên quan

Chi tiết và BibTeX nằm trong `pad_frequency_domain_references_v2.md`.

Các paper nên ưu tiên đọc/cite:

1. F. A. Ali, S. Mali, R. Mahakud, and G. Yadav, **“AI-enabled smart surveillance system for secure monitoring and authentication,”** *Scientific Reports*, vol. 16, 21686, 2026. DOI: `10.1038/s41598-026-52387-w`.
2. Y. Niu and X. Lin, **“Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction,”** *Neural Networks*, vol. 199, 108734, 2026. DOI: `10.1016/j.neunet.2026.108734`.
3. V. W. de Dravo et al., **“Oculus: Hierarchical Face Spoof Detection via Frequency-Enhanced Vision Transformers with Group-Aware Classification and Post-Fusion Attention,”** *ICCV Workshops*, 2025.
4. J. Cao and C. Ma, **“Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View,”** *WACV*, 2025, pp. 1005–1015. DOI: `10.1109/WACV61041.2025.00107`.
5. Y. Yu, Z. Du, H. Luo, C. Xiao, and J. Hu, **“Fourier-Based Frequency Space Disentanglement and Augmentation for Generalizable Face Anti-Spoofing,”** *IEEE Journal of Biomedical and Health Informatics*, vol. 29, no. 8, pp. 5413–5423, 2025. DOI: `10.1109/JBHI.2024.3417404`.
6. R. P. Singh, R. Dash, and R. K. Mohapatra, **“DEFuseNet: A domain-enhanced fusion network for generalizable face anti-spoofing,”** *Neurocomputing*, vol. 696, 134092, 2026. DOI: `10.1016/j.neucom.2026.134092`.
7. B. Chen, W. Yang, and S. Wang, **“Face Anti-Spoofing by Fusing High and Low Frequency Features for Advanced Generalization Capability,”** *IEEE MIPR*, 2020. DOI: `10.1109/MIPR49039.2020.00048`.

---

## 25. Research roadmap chốt lại

```text
E1  Spatial-only MobileNetV3
        ↓
E2  Frequency-only diagnostic
        ↓
E3  Spatial + DCT Frequency + CONCAT      ← V1 / main contribution
        ↓
Cross-domain + efficiency evaluation
        ↓
E4  Spatial + DCT Frequency + GATED       ← V2 / extension
        ↓
Ablation: Does adaptive fusion beat concat?
```

**Không nhảy thẳng tới gated fusion.** V1 concat tạo bằng chứng rằng frequency branch có ích; V2 mới trả lời câu hỏi liệu contribution của hai branch có cần thay đổi theo input hay không.
