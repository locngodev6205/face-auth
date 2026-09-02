# Lightweight Spatial–Frequency Face Anti-Spoofing (PAD)

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

### V1: Concatenation

Đây là lựa chọn nên dùng đầu tiên vì đơn giản, dễ giải thích và dễ ablation.

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

Ví dụ:

```python
fused = torch.cat([spatial_feat, freq_feat], dim=1)
logits = classifier(fused)
```

### Không nên bắt đầu bằng Gated Fusion

Gated fusion có thể là experiment mở rộng:

```text
Spatial ─┐
         ├─ Gate → weighted fusion
Frequency┘
```

nhưng không cần thiết cho phiên bản đầu tiên. Concatenation giúp câu hỏi nghiên cứu rõ hơn:

> “Việc thêm explicit frequency feature có mang lại lợi ích không?”

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

### Tối thiểu

| Experiment | Spatial | Frequency | Mục đích |
|---|:---:|:---:|---|
| E1 | ✓ | — | Baseline |
| E2 | ✓ | ✓ | Kiểm tra lợi ích của frequency |

### Có thể mở rộng

| Experiment | Spatial | Frequency | Fusion |
|---|:---:|:---:|---|
| E1 | ✓ | — | — |
| E2 | — | ✓ | — |
| E3 | ✓ | ✓ | Concat |
| E4 | ✓ | ✓ | Gated |

E1 và E3 là hai experiment quan trọng nhất nếu thời gian hạn chế.

---

## 12. Evaluation protocol

### In-domain

Train và test trên CelebA-Spoof theo split đã định nghĩa.

Metric chính:

- APCER — Attack Presentation Classification Error Rate;
- BPCER — Bona Fide Presentation Classification Error Rate;
- ACER = (APCER + BPCER) / 2.

Trong authentication, APCER đặc biệt quan trọng vì nó mô tả spoof bị chấp nhận là real.

### Cross-dataset

Train:

```text
CelebA-Spoof
```

Test:

```text
OULU-NPU
SiW
```

Không fine-tune trên test dataset khi thực hiện cross-dataset evaluation chính.

Mục tiêu:

> Kiểm tra xem frequency-aware representation có giúp model giảm độ suy giảm performance khi camera, người, môi trường hoặc điều kiện attack thay đổi hay không.

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
- gated fusion phức tạp
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

## 22. Architecture cuối cùng để đưa vào presentation

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

### Thông điệp cuối

> **Spatial features tell the model what the face looks like; frequency features provide complementary information about how image patterns vary across space.**

Trong project này, frequency branch là một giả thuyết nghiên cứu có thể kiểm chứng, không phải một đảm bảo rằng mọi spoof attack đều có đặc trưng tần số riêng.
