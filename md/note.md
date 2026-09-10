# Sổ tay Kiến thức & Giải đáp Kỹ thuật (Q&A Notes)

Tài liệu tổng hợp các câu hỏi - trả lời về kiến trúc Face Anti-Spoofing (PAD), Face Recognition, Tracking và Vector Database trong hệ thống **SmartFace**.

---

## Câu 1: Đặc trưng Spatial và Frequency khác nhau thế nào? Tại sao cần kết hợp cả hai?

**Trả lời:**
- **Đặc trưng Spatial (MobileNetV3):** Học *hình dạng, cấu trúc và ngữ nghĩa thị giác* khuôn mặt theo vị trí không gian (mắt, mũi, miệng, góc cạnh, độ mịn da). Trả lời câu hỏi: *"Cái gì đang xuất hiện, ở đâu trong ảnh?"*.
- **Đặc trưng Frequency (2D DCT):** Học *tốc độ biến thiên của cường độ pixel*, tức là kết cấu vi mô của bề mặt vật liệu:
  - *Tần số thấp:* Độ mượt, ánh sáng tổng thể, color distortion (dấu hiệu replay attack).
  - *Tần số trung/cao:* Lưới điểm ảnh màn hình, hạt mực in, vân moiré, aliasing (dấu hiệu print attack).
  - Trả lời câu hỏi: *"Ảnh này được tạo ra từ chất liệu tự nhiên hay tái tạo (in/màn hình)?"*.
- **Sự bổ sung:** Spatial "nhìn" khuôn mặt theo ngữ nghĩa con người; Frequency "soi" bản chất vật lý của vật liệu hiển thị. Hai góc nhìn trực giao nhau giúp mô hình tổng quát hóa tốt hơn khi gặp các camera hoặc bối cảnh lạ.

---

## Câu 2: Cơ chế Fusion hoạt động thế nào? Vai trò của tầng MLP và Classifier phía sau?

**Trả lời:**
- **Cơ chế Fusion (Concatenation):** Ghép nối tiếp vector Spatial ($256$-D) và Frequency ($64$-D) thành vector liên hợp $320$-D. Phương pháp Late Fusion này bảo toàn $100\%$ thông tin gốc của từng miền, không làm triệt tiêu phân phối giá trị như phép cộng/nhân.
- **Vai trò của tầng MLP (`Linear(320, 128) -> ReLU -> Dropout(0.2)`):**
  - *Tương tác chéo (Cross-modal Interaction):* Học mối tương quan logic giữa 2 miền (ví dụ: *"Nếu Spatial thấy độ tương phản cao ĐỒNG THỜI Frequency thấy năng lượng dải cao bất thường $\rightarrow$ Tăng điểm Spoof"*).
  - *Nén chiều (320 $\rightarrow$ 128):* Loại bỏ các đặc trưng thừa không phục vụ phân biệt liveness.
  - *Dropout(0.2):* Ngăn mô hình ỷ lại vào nhánh Spatial đã được pretrained, buộc MLP phải học cả tín hiệu từ nhánh Frequency.
- **Vai trò của Classifier (`Linear(128, 1)`):**
  - Chiếu vector $128$-D thành scalar logit $z \in \mathbb{R}$, qua hàm Sigmoid ra xác suất giả mạo $P(\text{Spoof}) \in [0, 1]$.
  - Cho phép hiệu chuẩn ngưỡng an ninh ($\tau$) phù hợp với hệ thống điểm danh (cân đối giữa APCER và BPCER).

---

## Câu 3: Vector Search ở Database đã có thứ tự xếp hạng rồi, tại sao vẫn cần bước "Candidate Ranking"?

**Trả lời:**
Vector Search ở DB chỉ là **xếp hạng vector thô (Raw Vector Ranking)**, chưa phải là **xếp hạng danh tính nhân viên (Identity Ranking)**:
1. **1 nhân viên có nhiều vector:** Một người đăng ký $3 - 5$ ảnh. Top-5 vector DB trả về có thể gồm 3 vector của Nhân viên A và 2 vector của Nhân viên B. Bước Ranking sẽ *Group by User ID* và tổng hợp điểm để xác định chính xác ai là người có độ khớp cao nhất.
2. **ANN xấp xỉ vs Exact Cosine:** Vector DB dùng thuật toán ANN (HNSW/IVFFlat) để tìm kiếm siêu tốc trên quy mô lớn nhưng kết quả chỉ là gần đúng. Bước Ranking tính lại Cosine Similarity chính xác $100\%$ từ vector Float32 gốc.
3. **Xử lý người lạ (Unknown):** Vector Search luôn trả về Top-K kể cả với kẻ lạ mặt. Candidate Ranking áp dụng ngưỡng `MATCH_THRESHOLD` ($0.35$) để gán nhãn `UNKNOWN`.
4. **Kiểm tra biên an toàn (Confidence Margin):** So sánh chênh lệch giữa Top-1 và Top-2 ($\Delta = \text{Score}_1 - \text{Score}_2$). Nếu $\Delta$ quá nhỏ (mặt giống nhau hoặc ảnh mờ), hệ thống có thể từ chối để tránh nhận diện nhầm.

---

## Câu 4: Single-Frame PAD có cạnh tranh được với Temporal (Video/Multi-frame) PAD không?

**Trả lời:**
**Có, hoàn toàn cạnh tranh được**, đặc biệt là trong các hệ thống điểm danh AIoT / Edge:
- **Ưu thế vượt trội của Single-Frame:**
  - *Tốc độ & Độ trễ:* Xử lý trong $10 - 25$ ms (ngay tức thì), không cần chờ gom đủ $15 - 30$ frames buffer như Temporal. Phù hợp cho luồng đi bộ check-in liên tục.
  - *Tài nguyên:* Cực nhẹ (MobileNetV3 + DCT tốn ít RAM/FLOPs), chạy mượt trên chip IoT/Raspberry Pi.
  - *Chống Print Attack:* Dấu hiệu in ấn đã bộc lộ $100\%$ trên từng khung hình, không cần yếu tố thời gian.
- **Điểm Temporal mạnh hơn:** Phát hiện replay video chất lượng cao trên màn hình OLED sắc nét, hoặc deepfake chuyển động.
- **Giải pháp tối ưu của SmartFace:** Dùng **AI Single-Frame** (để giữ tốc độ tối đa) kết hợp **Lọc mượt ở tầng Tracker** (EMA / voting qua các frame bám vết của cùng một người) $\rightarrow$ Đạt được cả tốc độ của Single-Frame lẫn độ ổn định của Temporal.

---

## Câu 5: Logic của FaceTracker hiện tại hoạt động ra sao? Tại sao giúp cứu FPS?

**Trả lời:**
- **Bản chất:** Là một **Spatial IoU Tracker** kết hợp **Greedy Bipartite Matching** và **Temporal Recognition Throttling**:
  1. *Tính ma trận IoU:* Đo tỷ lệ đè lên nhau giữa các bounding box mới và các track cũ.
  2. *Greedy Matching:* Ghép cặp ưu tiên IoU cao nhất ($\ge 0.3$). Track đã match được cập nhật bbox mới và reset `missing_frames = 0`.
  3. *Track mới / Xóa track:* Detection không trùng ai sẽ sinh Track ID mới. Track mất dấu quá $10$ frame liên tiếp (`max_missing_frames = 10`) mới bị xóa khỏi RAM.
- **Tác dụng cứu FPS:**
  - Nhận diện khuôn mặt (ArcFace + DB Search) rất nặng, nếu chạy $30$ lần/giây sẽ làm tụt FPS nghiêm trọng.
  - Tracker chỉ chạy AI nhận diện **1 lần ngay khi người đó xuất hiện**, sau đó **cache kết quả (tên, điểm số)** và bám theo toạ độ bbox.
  - Chỉ re-verify lại sau mỗi `RECOGNIZE_INTERVAL_SECONDS = 1.0s`.
  - Giảm tải tính toán AI tới $96\%$, giúp webcam duy trì ổn định $25 - 30$ FPS.

---

## Câu 6: Cơ chế L2 Normalize là gì và có tác dụng gì trong pipeline?

**Trả lời:**
- **Cơ chế:** Chia vector cho độ dài Euclid của chính nó: $\hat{v} = \frac{v}{\|v\|_2}$. Sau khi chuẩn hóa, vector có độ dài đúng bằng $1$ và nằm trên mặt cầu đơn vị siêu không gian 512 chiều.
- **Tác dụng trong pipeline:**
  1. *Tối ưu hóa Cosine Similarity:* Với 2 vector có chuẩn bằng $1$, công thức Cosine rút gọn thành **phép tích vô hướng (Dot Product):** $\text{Cosine}(u, v) = u \cdot v$. Chỉ cần 1 phép nhân ma trận là xong, không cần tính căn bậc hai hay phép chia trong vòng lặp webcam.
  2. *Triệt tiêu ảnh hưởng của độ sáng/chất lượng ảnh:* Độ lớn vector thô thường bị chi phối bởi ảnh sáng hay tối. Chuẩn hóa L2 đưa mọi ảnh về cùng bán kính $r=1$, chỉ giữ lại thông tin góc biểu diễn nhân dạng thuần túy.
  3. *Tương thích chuẩn ArcFace:* ArcFace tối ưu hóa góc trên mặt cầu, L2 Normalize là điều kiện tiên quyết để ngưỡng so khớp (`MATCH_THRESHOLD`) hoạt động chính xác.

---

## Câu 7: Quá trình Enroll một người diễn ra như thế nào? Cách tính Mean và lọc Outlier?

**Trả lời:**
Khi đăng ký nhân viên với $3 - 5$ ảnh, quy trình xử lý qua 4 bước:
1. **Tiền xử lý & Trích xuất:** Detect mốc mặt $\rightarrow$ Align về $112 \times 112 \rightarrow$ ArcFace trích xuất vector 512-D $\rightarrow$ L2-Normalize từng vector.
2. **Tính Trung bình sơ bộ (Preliminary Mean):**
   $$M_{\text{prelim}} = \text{L2\_norm}\left( \frac{1}{N} \sum_{i=1}^N v_i \right)$$
3. **Lọc ảnh rác (Outlier Filtering):**
   - Đo độ tương đồng giữa từng ảnh với $M_{\text{prelim}}$: $s_i = v_i \cdot M_{\text{prelim}}$.
   - Nếu $s_i < 0.35$ (`outlier_threshold`): Ảnh bị coi là outlier (mờ, nhắm mắt, góc quá nghiêng hoặc người khác lọt vào) và bị loại bỏ để tránh kéo lệch tâm đại diện.
4. **Tính Final Mean Embedding & Lưu DB:**
   - Tính lại trung bình từ các ảnh sạch còn lại: $M_{\text{final}} = \text{L2\_norm}(\text{mean}(v_{\text{valid}}))$.
   - Lưu vào PostgreSQL (pgvector):
     - Vector đại diện lưu với `embedding_id = "[user_id]_0000"` và cờ `is_mean = TRUE`.
     - Các vector ảnh thành phần lưu với `is_mean = FALSE`.
   - *Lợi ích:* Khi chấm công (1:N search), hệ thống chỉ quét các vector `is_mean = TRUE`, giảm $10$ lần số phép so khớp trong DB.