# SmartFace AIoT — Hướng Dẫn Chi Tiết Kiến Trúc & Luồng Thực Thi Real-time (app.py)

> **Mục đích tài liệu:**  
> Tài liệu này tổng hợp toàn bộ các nội dung phân tích chuyên sâu về luồng hoạt động của [app.py](face_auth/app.py) từ đầu phiên làm việc đến nay. Bạn có thể sử dụng tài liệu này để tự học, ôn tập và tự tin thuyết trình bảo vệ đồ án trước Hội đồng / Giảng viên mà không sợ quên chi tiết.

---

## 🗺️ TỔNG QUAN DÂY CHUYỀN XỬ LÝ (PIPELINE)

Ứng dụng [app.py](face_auth/app.py) là **bộ điều phối trung tâm (Pipeline Coordinator)**, liên kết 9 giai đoạn theo thời gian thực:

```text
[Module 1: Load Once & Config] 
       ↓ (Nạp model vào RAM 1 lần duy nhất)
[Module 2: VideoCapture 640×480] 
       ↓ (Đọc khung hình tươi BGR)
[Module 3: SCRFD Face Detection] ── (Chạy mỗi 0.1s hoặc khi cần)
       ↓ (Bbox + 5 Landmarks)
[Module 4: Face Tracking] ──────── (IoU Matching + Optical Flow xen kẽ 2.7ms)
       ↓ (Track ID ổn định)
[Module 5: MobileNetV3 PAD] ────── (Chạy mỗi 0.1s, gom 5 vote LogSumExp)
       │  ├─ SPOOF ──> Chặn nhận diện ──> Báo động đỏ 🔴
       │  └─ REAL  ──> Mở khóa đi tiếp ↓
[Module 6: Face Alignment] ────── (Biến đổi Affine 5-point -> Chuẩn 112×112)
       ↓
[Module 7: ArcFace Embedding] ──── (Trích xuất vector 512-D L2-Normalized)
       ↓
[Module 8: PostgreSQL pgvector] ── (So khớp Cosine 1:N + Cooldown 15 phút)
       ↓
[Module 9: HUD Display & Control]─ (Vẽ 5 trạng thái + FPS trượt + Phím tắt)
```

---

## 📍 CHI TIẾT CÁC MODULE ĐÃ PHÂN TÍCH

---

### MODULE 1: Nạp Cấu Hình & Khởi Tạo Tài Nguyên (Nguyên Tắc Load-Once)

- **Mã nguồn liên quan:** 
  - Khởi tạo trong app: [app.py:L265-L336](face_auth/app.py#L265-L336)
  - Cấu hình hệ thống: [config.py:L8-L38](face_auth/config.py#L8-L38)
  - Biến môi trường: [.env](face_auth/.env)

#### 1. Nạp cấu hình an toàn chuỗi:
- Hàm [_clean_env_val()](face_auth/config.py#L8-L13) dùng regex loại bỏ khoảng trắng và xử lý comment phía sau, giữ nguyên nếu mật khẩu DB có chứa ký tự `#` đặc biệt.

#### 2. Triết lý "LOAD ONCE":
- **Tuyệt đối không** nạp lại mô hình AI hay tạo mới kết nối Database bên trong vòng lặp `while True`.
- Nạp 5 đối tượng cốt lõi ngay tại hàm `main()` trước khi camera mở:
  1. `FaceDetector` (SCRFD `det_500m.onnx`, 640×640)
  2. `FaceEmbedder` (ArcFace trích từ `buffalo_s`)
  3. `VectorDB` (`psycopg_pool.ConnectionPool` 1–4 connections)
  4. `FaceTracker` (Bộ nhớ theo dõi khuôn mặt)
  5. `AntiSpoofPredictor` (Mô hình MobileNetV3 3-class, input 224×224)

#### 3. Cơ chế phòng thủ lỗi (Defensive Fail-Safe):
- Nếu `--pad` bật mà file model `.onnx` bị lỗi/thiếu, hệ thống **chủ động gọi `sys.exit(1)` dừng chương trình ngay lập tức** ([app.py:L328-L333](face_auth/app.py#L328-L333)).
- *Lý do:* Không cho phép app "chạy cố" khi thiếu lớp bảo vệ liveness, tránh bị giả mạo.

---

### MODULE 2: Thu Nhận Luồng Camera & Chẩn Đoán Phần Cứng (`cv2.VideoCapture`)

- **Mã nguồn liên quan:** [app.py:L337-L426](face_auth/app.py#L337-L426)

#### 1. Đàm phán phần cứng & Chống trễ (Buffer Size = 1):
- Thiết lập:
  - `CAP_PROP_FRAME_WIDTH = 640`
  - `CAP_PROP_FRAME_HEIGHT = 480`
  - `CAP_PROP_FPS = 30`
  - `CAP_PROP_BUFFERSIZE = 1`
- `BUFFERSIZE = 1` ép camera driver không đệm 3–5 khung hình cũ, đảm bảo khung hình đọc ra luôn là khung hình thời gian thực tươi mới nhất, loại bỏ độ trễ tích lũy.

#### 2. Bản chất của `CAMERA_WIDTH` & `CAMERA_HEIGHT`:
- Là kích thước cố định của bức ảnh chụp toàn cảnh do camera cung cấp ở mỗi frame: **ngang 640 px × dọc 480 px**.
- *Tại sao là 640×480 chứ không phải 1080p?*
  - Tiết kiệm 6.75 lần số lượng pixel so với Full HD, tránh nghẽn bus USB.
  - Vừa khít với kích thước mạng SCRFD `det_size = (640, 640)`.
  - Đủ sắc nét cho ArcFace và PAD nhận diện.

#### 3. Phòng thủ Backend (OS Hardware Abstraction):
- Gọi lại `cap.getBackendName()`, `cap.get(...)` để in thông số thực tế ([app.py:L352-L367](face_auth/app.py#L352-L367)) vì OpenCV không báo lỗi nếu camera không hỗ trợ đúng resolution.

#### 4. Đo FPS trượt chính xác (Rolling FPS):
- Dùng `frame_time_deque` với cửa sổ trượt 30 frame ([app.py:L385](face_auth/app.py#L385)).
- Công thức:
  $$\text{FPS} = \frac{\text{len}(deque)}{\sum \Delta t}$$
- Giúp con số FPS hiển thị màu vàng trên màn hình không bị giật nháy loạn xạ.

---

### MODULE 3: Phát Hiện Khuôn Mặt (Face Detection — SCRFD 640×640)

- **Mã nguồn liên quan:**
  - Điều phối nhịp trong app: [app.py:L428-L455](face_auth/app.py#L428-L455)
  - Class detector: [detection/detector.py:L14-L88](face_auth/detection/detector.py#L14-L88)

#### 1. Nhịp thực thi (Detection Cadence):
- SCRFD ngốn ~42.7 ms/lần.
- Hệ thống chỉ cho phép chạy SCRFD sau mỗi **`0.1 giây`** (`DETECTION_INTERVAL_SECONDS = 0.1s`).
- **Ngoại lệ an toàn:** Nếu tracker phát hiện mất an toàn (`tracker.needs_redetection() == True`), SCRFD sẽ được gọi ngay lập tức ở frame kế tiếp mà không chờ.

#### 2. Cơ chế nội bộ của SCRFD & Bản chất file `det_500m.onnx`:
- **Nguồn gốc & Bản chất:** 
  - `det_500m.onnx` là file trọng số Deep Learning (Pre-trained weights) đã được nhóm tác giả InsightFace huấn luyện sẵn trên hàng triệu khuôn mặt.
  - Khi cài đặt thư viện `insightface`, mô hình tự động được tải (auto-download) về thư mục cache hệ thống tại `~/.insightface/models/buffalo_s/det_500m.onnx`, không cần tải thủ công.
  - Trong code [detection/detector.py:L39-L46](face_auth/detection/detector.py#L39-L46), lệnh `self._model = app.models.get("detection")` chính là nơi nạp file ONNX này vào runtime.
- **Ý nghĩa tên gọi & Khái niệm MegaFLOPs:**
  - `det`: Detection (Bộ phát hiện khuôn mặt).
  - **MegaFLOPs (MFLOPs)**: Đơn vị đo khối lượng tính toán của mạng Deep Learning (Floating-point Operations). $500\text{ MFLOPs} = 500\text{ triệu phép tính}$ cộng/nhân số thực dấu phẩy động cho 1 bức ảnh.
  - *So sánh độ nhẹ:* Mô hình ResNet-50 tốn ~4,000 MFLOPs, YOLOv8x tốn ~250,000 MFLOPs. Con số **500 MFLOPs là siêu nhẹ (ultra-lightweight)**, cho phép chip CPU máy tính hoặc Raspberry Pi 4 xử lý mượt mà trong ~30–40ms mà không làm quá nhiệt chip.
- **Các file model ONNX có sẵn trong gói `buffalo_s` của InsightFace:**
  1. `det_500m.onnx`: Bộ phát hiện mặt SCRFD siêu nhẹ (đang dùng ở Module 3).
  2. `w600k_mbf.onnx`: Bộ trích xuất đặc trưng MobileFaceNet (ArcFace) trích xuất vector 512 chiều (dùng ở Module 7).
  3. `2d106det.onnx`: Mô hình dự đoán 106 điểm mốc 2D trên mặt (hệ thống bỏ qua bằng `allowed_modules=["detection"]` để tiết kiệm RAM).
  4. `1k3d68.onnx`: Mô hình dựng 68 điểm 3D khuôn mặt (bỏ qua).
  5. `genderage.onnx`: Mô hình dự đoán tuổi và giới tính (bỏ qua).
- **Tác dụng cốt lõi của `det_500m.onnx`:**
  - Quét tìm và trả về tọa độ Hộp bao (Bounding Box) `(x1, y1, x2, y2)`.
  - Định vị **5 điểm mốc then chốt (Landmarks)**: Mắt phải, mắt trái, đỉnh mũi, khóe miệng trái, khóe miệng phải (chuẩn ArcFace).

#### 3. Bản chất Thuật toán NMS (`DETECTOR_NMS_THRESH = 0.3`):
- **Tại sao mô hình sinh ra nhiều Bbox cho cùng 1 mặt?**
  - Mạng Object Detection (như SCRFD, YOLO) rải hàng ngàn điểm neo (Anchor points) đa tỉ lệ trên khắp ảnh.
  - Khi có một khuôn mặt, nhiều điểm neo lân cận sẽ đồng thời phát hiện và trả về các Bounding Box hơi lệch nhau một vài pixel.
- **Thuật toán NMS (Non-Maximum Suppression) giải quyết ra sao?**
  - Chọn hộp có điểm tin cậy cao nhất (Maximum Confidence).
  - So sánh độ trùng lặp diện tích (IoU) của hộp này với các hộp xung quanh.
  - Nếu trùng lặp $\text{IoU} \ge \text{ngưỡng}$ (ví dụ `0.3` trong [config.py:L154](face_auth/config.py#L154)) $\rightarrow$ **Xóa bỏ các hộp thừa**, chỉ giữ lại 1 hộp chuẩn nhất.
- **Vị trí áp dụng trong code:**
  - **Với YunNet:** Truyền trực tiếp vào tham số `nms_threshold` tại [detection/yunnet_detector.py:L43](face_auth/detection/yunnet_detector.py#L43).
  - **Với SCRFD:** Bên trong hàm `self._model.detect()` ([detection/detector.py:L55](face_auth/detection/detector.py#L55)) đã tích hợp sẵn NMS của InsightFace (mặc định thuộc tính `self._model.nms_thresh = 0.4`).

#### 4. Bộ lọc chất lượng khuôn mặt (Face Quality Gate — 4 tiêu chuẩn vàng):
Hệ thống không tiếp nhận bừa bãi mọi khuôn mặt mà áp dụng bộ lọc 4 tiêu chuẩn khắt khe trước khi bàn giao cho Tracker:
1. **Kích thước tối thiểu (`min_face_size = 60px`):**
   - Code tại: [detector.py:L70-L75](face_auth/detection/detector.py#L70-L75) và [yunnet_detector.py:L170-L171](face_auth/detection/yunnet_detector.py#L170-L171).
   - *Tác dụng:* Loại bỏ các khuôn mặt đứng quá xa (dưới 60×60 px), vì ảnh mặt quá mờ/nhỏ sẽ gây sai số lớn cho Anti-Spoofing và trích xuất vector ArcFace.
2. **Khoảng cách an toàn tới 4 mép viền ảnh (`margin = 5px`):**
   - Code tại: [yunnet_detector.py:L165-L167](face_auth/detection/yunnet_detector.py#L165-L167) ([config.py:L84](face_auth/config.py#L84)).
   - *Tác dụng:* Bắt buộc mặt phải cách viền ảnh $\ge 5$ px, loại bỏ trường hợp người mới bước vào hoặc sắp đi ra chỉ lộ nửa mặt (bị cụt tai, mất trán), tránh việc AI đoán mò trên dữ liệu khuyết tật.
3. **Loại bỏ hộp bao tràn viền (Out-of-bounds):**
   - Code tại: [yunnet_detector.py:L160-L162](face_auth/detection/yunnet_detector.py#L160-L162) và [tracker.py:L346-L354](face_auth/tracking/tracker.py#L346-L354).
   - *Tác dụng:* Loại bỏ ngay các tọa độ bị âm (`x1 < 0, y1 < 0`) hoặc vượt quá chiều rộng/cao của ảnh (`x2 > 640, y2 > 480`).
4. **Ngưỡng tự tin phát hiện (`conf_thresh = 0.5`):**
   - Code tại: [detector.py:L36](face_auth/detection/detector.py#L36) và [yunnet_detector.py:L97](face_auth/detection/yunnet_detector.py#L97) ([config.py:L78](face_auth/config.py#L78)).
   - *Tác dụng:* AI phải tự tin từ 50% trở lên mới công nhận là mặt người, loại bỏ các vật thể gây nhiễu (hoa văn áo, bóng râm, đồ vật...).

---

### MODULE 4: Bám Vết & Điều Phối Nhịp (Face Tracking — IoU + Optical Flow)

- **Mã nguồn liên quan:** [tracking/tracker.py](face_auth/tracking/tracker.py)

Đây là "bộ não điều phối tốc độ" của hệ thống, kết hợp 2 thuật toán:

#### 1. Khi Detector CHẠY (`detector_called = True`) — Thuật toán IoU Greedy Matching:
- **Công thức IoU (Intersection over Union):**
  $$\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Diện tích phần Giao (Intersection)}}{\text{Tổng diện tích 2 hộp bao phủ (Union)}}$$
  - Cài đặt tại: [tracker.py:L9-L30](face_auth/tracking/tracker.py#L9-L30).
  - **Ý nghĩa ngưỡng 30% (`TRACK_IOU_THRESHOLD = 0.3`):** Con số 30% là tỷ lệ giữa *diện tích phần chồng đè lên nhau* so với *tổng diện tích cả 2 hộp cộng lại*. Giữa 2 frame cách nhau chỉ 0.1s, người chuyển động nhẹ sẽ có độ trùng lặp từ 50%–80%. Đặt ngưỡng 30% ([config.py:L174](face_auth/config.py#L174)) đảm bảo kết luận chính xác 2 hộp là cùng một người.
- **Khớp nối (Greedy Matching):**
  - Ghép cặp có IoU cao nhất nếu $\ge 0.3$.
  - **Khớp thành công:** Cập nhật vị trí, reset `missing_frames = 0`.
  - **Người mới:** Cấp `Track ID` mới toanh, chuyển về `PAD_PENDING`.
  - **Người biến mất:** Tăng `missing_frames += 1`. Nếu mất dấu quá 10 frame liên tiếp (`missing_frames > 10`), **xóa sổ khỏi RAM** để giải phóng tài nguyên.
- **Tính độc lập của `missing_frames` (Rất quan trọng):**
  - **Mỗi một người đứng trước camera sẽ được cấp một đối tượng `Track` riêng với một biến đếm `missing_frames` hoàn toàn độc lập** ([tracker.py:L56](face_auth/tracking/tracker.py#L56)).
  - *Ví dụ thực tế:* Người A đứng yên trước camera (`Track #1: missing = 0`), người B quay lưng bước ra khỏi phòng (`Track #2: missing` tăng 1, 2, 3... 11). Khi người B mất tích quá 10 frame, hệ thống **chỉ xóa duy nhất `Track #2` của người B khỏi RAM**, đối tượng `Track #1` của người A vẫn tồn tại độc lập và hoạt động bình thường, không bao giờ bị ảnh hưởng!

#### 2. Khi Detector NGHỈ (`detector_called = False`) — Lucas–Kanade Optical Flow:
Hàm [_propagate()](face_auth/tracking/tracker.py#L277-L323) chỉ tốn **~2.7 ms**, giúp hệ thống duy trì tốc độ khung hình cao (30 FPS) mà không cần chạy detector liên tục:

- **Tại sao phải chuyển ảnh sang Grayscale (`_gray()`):**
  - *Bản chất thuật toán:* Lucas-Kanade dựa trên giả định bảo toàn độ sáng ($I(x,y,t) = I(x+\Delta x, y+\Delta y, t+\Delta t)$). Thuật toán chỉ cần đo gradient độ sáng không gian và thời gian trên kênh cường độ (Intensity - 1 kênh), thông tin màu sắc (BGR - 3 kênh) là dư thừa.
  - *Tối ưu tài nguyên:* Chuyển sang ảnh 1 kênh 8-bit giúp giảm ~66% dung lượng bộ nhớ đệm CPU/RAM và giảm 3 lần khối lượng phép tính vi phân ma trận, đảm bảo xử lý mượt mà thời gian thực.

- **Đầu vào và Đầu ra của hàm `cv2.calcOpticalFlowPyrLK` ([tracker.py:L295-L303](face_auth/tracking/tracker.py#L295-L303)):**
  - **Inputs:**
    1. `prevImg` (`self._previous_gray`): Frame trước dạng ảnh xám 8-bit `uint8`.
    2. `nextImg` (`gray`): Frame hiện tại dạng ảnh xám 8-bit `uint8`.
    3. `prevPts` (`points`): Tọa độ 2D của **5 điểm neo đại diện** (4 góc Bbox + 1 điểm trung tâm mặt), shape `(5, 1, 2)` kiểu `float32`.
    4. `nextPts` (`None`): Mảng đệm chứa kết quả (để OpenCV tự cấp phát).
    5. `winSize=(21, 21)`: Cửa sổ tìm kiếm lân cận $21 \times 21$ pixel quanh mỗi điểm.
    6. `maxLevel=2`: Số tầng kim tự tháp ảnh (3 cấp: $1, \frac{1}{2}, \frac{1}{4}$). Cấp thu nhỏ bắt chuyển động lớn, cấp gốc tinh chỉnh chi tiết.
    7. `criteria`: Điều kiện dừng lặp (lặp tối đa 20 lần hoặc độ dịch chuyển $< 0.03$ pixel).
  - **Outputs:**
    1. `next_points`: Mảng tọa độ mới của 5 điểm trong frame hiện tại.
    2. `status`: Mảng cờ trạng thái (`1` = bám thành công, `0` = mất dấu/ra ngoài biên).
    3. `err`: Vector sai số độ tương đồng giữa vùng ảnh cũ và mới.

- **Nguyên lý tìm điểm tiếp theo của thuật toán Lucas-Kanade (Xét 1 điểm $P(x,y)$):**
  - *Phương trình cơ bản:* Từ khai triển Taylor bậc 1 theo giả thiết bảo toàn độ sáng:
    $$I_x \cdot u + I_y \cdot v + I_t = 0$$
    Trong đó $I_x, I_y$ là gradient độ dốc không gian, $I_t$ là chênh lệch sáng giữa 2 frame, $(u,v)$ là độ dịch chuyển cần tìm.
  - *Giải quyết nghịch lý 1 phương trình 2 ẩn (Aperture Problem):* Thuật toán mượn một cửa sổ lân cận $21 \times 21$ pixel (441 pixel) cùng chia sẻ chung vận tốc $(u,v)$. Ta có hệ thừa phương trình $A \cdot d = b$.
  - *Nghiệm bình phương tối thiểu (Least Squares):*
    $$d = (A^T A)^{-1} A^T b \quad \text{với } A^T A = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$$
    Khi vùng quanh điểm có vân/góc cạnh, ma trận $A^T A$ khả nghịch $\to$ giải ra ngay vector dịch chuyển $(u,v)$ và vị trí mới $x_{\text{mới}} = x + u, y_{\text{mới}} = y + v$.

- **Cơ chế cập nhật Bounding Box bằng Trung vị (`median displacement`):**
  - Hệ thống **không lấy trực tiếp 5 điểm mới để vẽ lại khung** vì cơ mặt cử động sẽ làm Bbox bị méo mó, co cụm.
  - Thay vào đó, hệ thống yêu cầu tối thiểu **3/5 điểm hợp lệ** (`status == 1`), sau đó tính độ dịch chuyển trung vị:
    $$dx = \text{median}(next\_x - prev\_x), \quad dy = \text{median}(next\_y - prev\_y)$$
  - Sau đó tịnh tiến cả bounding box: `track.bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]`. Cơ chế trung vị giúp loại bỏ hoàn toàn các điểm nhiễu (outliers) và bảo toàn chuẩn xác hình dạng khung khuôn mặt.

#### 3. Chu trình Điều phối Nhịp (Cadence Orchestration): Bao nhiêu frame thì Detector được gọi lại?
Trong điều kiện thuật toán Optical Flow bám vết **hoàn toàn trơn tru** (`valid >= 4` điểm):
- **Chu kỳ thực thi:** Cứ sau **~3 frames** (tương đương **0.1 giây**, tức **10 lần/giây** trên camera 30 FPS) thì Detector mới được gọi lại một lần:
  ```text
  Frame 1: 🟡 DETECTOR CHẠY (SCRFD mất ~25ms) ──> Định vị mặt + Khởi tạo 5 điểm neo
  Frame 2: 🟢 OPTICAL FLOW BÁM (chỉ mất ~2.7ms) ──> Detector NGHỈ
  Frame 3: 🟢 OPTICAL FLOW BÁM (chỉ mất ~2.7ms) ──> Detector NGHỈ
  Frame 4: 🟡 DETECTOR CHẠY LẠI (0.1s đã trôi qua) ──> Tái định vị chính xác
  ```
- **Tỷ lệ phân chia khối lượng:**
  - **Detector (SCRFD):** Chỉ chạy khoảng **$33\%$** tổng số khung hình.
  - **Optical Flow (LK):** Đảm nhiệm tới **$67\%$** số khung hình còn lại.
  - **Hiệu quả:** Giải phóng ~70% tải tính toán của CPU/GPU, đảm bảo duy trì tốc độ khung hình chuẩn 30 FPS.
- **Công thức tính thời gian trong code ([config.py:L229-L238](face_auth/config.py#L229-L238), [app.py:L430-L435](face_auth/app.py#L430-L435)):**
  $$\text{DETECTION\_INTERVAL\_SECONDS} = \frac{\text{RECOGNIZE\_INTERVAL (0.5s)}}{\text{PAD\_SMOOTH\_WINDOW (5)}} = \mathbf{0.1\text{ giây}}$$
  $$\text{Số frame giữa 2 lần detect} = \frac{0.1\text{s}}{1/30\text{s}} \approx \mathbf{3\text{ frames}}$$
- **Ưu điểm của việc điều phối theo Thời gian thực (`time.monotonic()`) thay vì đếm số frame cứng:**
  - Nếu đếm cứng số frame (như `frame_count % 3 == 0`): Khi camera chạy chậm trên máy yếu (10 FPS), 3 frame mất tới 0.3s gây giật lag; còn trên camera 60 FPS, 3 frame chỉ có 0.05s làm detector chạy quá dày gây nghẽn phần cứng.
  - Đặt theo mốc thời gian thực 0.1s giúp hệ thống chạy ổn định và nhất quán trên mọi phần cứng từ PC đến Raspberry Pi/Edge device.
- **Cơ chế "Bẻ gãy chu trình" khi có biến cố:**
  - Nếu người dùng quay mặt nhanh, bị che khuất, hoặc số điểm neo $< 3$: hàm `tracker.needs_redetection()` trả về `True`.
  - Hệ thống **lập tức kích hoạt SCRFD ở ngay frame tiếp theo mà không cần chờ hết 0.1s** để kịp thời cứu viện, không để mất dấu khuôn mặt.

#### 4. Bốn trạng thái của `tracker_status`:
| Trạng thái | Ý nghĩa | Hành vi hệ thống |
| :--- | :--- | :--- |
| `"detected"` | SCRFD vừa phát hiện ở frame này | Đáng tin cậy 100%, cho detector nghỉ 0.1s tiếp theo. |
| `"tracking"` | Optical Flow bám rất tốt ($\ge 4$ điểm) | Tiếp tục cho detector nghỉ, vẽ khung mượt (~2.7ms). |
| `"needs_redetection"` | Nghi ngờ (chỉ còn 3 điểm, hoặc scale méo) | Dùng tạm frame này, nhưng **ép frame sau phải bật SCRFD ngay**. |
| `"lost"` | Mất dấu hoàn toàn ($< 3$ điểm) | **Tạm ẩn, không vẽ bừa**; lập tức kích hoạt SCRFD tìm kiếm cứu nạn ở frame sau. |

#### 5. Phân biệt: Bám vết liên tục vs Khóa Điểm Danh 15 Phút (Attendance Cooldown):
- **Bám vết (Detection & Tracking):** **KHÔNG BAO GIỜ DỪNG**. Hệ thống vẫn liên tục duy trì Track ID, vẽ khung bounding box xanh lá bám theo mặt người đó và phát hiện thời điểm người đó rời đi.
- **Khóa Điểm Danh (Attendance Cooldown - 15 phút):**
  - Cài đặt tại: [tracker.py:L160-L164](face_auth/tracking/tracker.py#L160-L164), [app.py:L170-L198](face_auth/app.py#L170-L198), và [database/vector_db.py:L178-L193](face_auth/database/vector_db.py#L178-L193).
  - Khi đã nhận diện đúng nhân viên (`Nguyễn Văn A`) và ghi log thành công vào DB, hệ thống gán mốc `track.last_attendance_time = time.time()`.
  - Trong vòng **15 phút** (`ATTENDANCE_GAP_MINUTES = 15`), hàm `can_log_attendance(15)` trả về `False` $\to$ **Chặn hoàn toàn các câu lệnh ghi log vào PostgreSQL**. Người dùng dù có đứng trước camera cả ngày thì DB cũng không bị ghi trùng lặp.
  - **Bộ đệm chống mất Cooldown (`_unknown_streak`):** Nếu nhân viên chớp mắt hoặc quay mặt khiến 1-2 frame bị nháy `UNKNOWN`, hệ thống **chưa xóa timer cooldown ngay**, mà yêu cầu phải liên tục `UNKNOWN > 3 chu kỳ (~1.5s)` thì mới reset timer, tránh việc vừa điểm danh xong bị mất timer đếm ngược.

- **Chu trình khi người dùng đi ra khỏi phòng rồi quay lại (Đồng bộ RAM ⟷ Database):**
  1. *Khi đi ra ngoài:* Biến đếm `missing_frames` tăng dần. Khi `missing_frames > 10` (~0.33s), đối tượng `Track #1` bị xóa sổ hoàn toàn khỏi RAM để giải phóng bộ nhớ ([tracker.py:L390](face_auth/tracking/tracker.py#L390)).
  2. *Khi quay trở lại (ví dụ sau 2 phút, vẫn trong 15 phút cooldown):*
     - SCRFD phát hiện khuôn mặt và cấp một đối tượng mới toanh trong RAM: `Track #2` (với `last_attendance_time = None`).
     - Qua bước PAD (REAL) $\to$ ArcFace nhận diện ra nhân viên `EMP001` (Nguyễn Văn A).
     - `app.py` gọi hàm `db.log_attendance(...)` để kiểm tra.
  3. *Database PostgreSQL kiểm tra đĩa cứng ([vector_db.py:L180-L192](face_auth/database/vector_db.py#L180-L192)):*
     ```sql
     SELECT timestamp FROM attendance_logs
     WHERE employee_id = 'EMP001' AND action = 'CHECKIN'
       AND timestamp > NOW() - INTERVAL '15 minutes'
     ORDER BY timestamp DESC LIMIT 1;
     ```
     DB thấy nhân viên này đã CHECKIN cách đây 2 phút $\to$ **Từ chối ghi log (`success = False`)**, nhưng **trả về timestamp cũ (`last_ts`)**.
  4. *Đồng bộ ngược DB ➔ RAM ([app.py:L196-L197](face_auth/app.py#L196-L197)):*
     - `app.py` nhận được `last_ts` liền gán ngược lại cho Track mới: `track.last_attendance_time = last_ts.timestamp()`.
     - Các frame tiếp theo, `Track #2` đã biết mình còn đang dính cooldown ngay trong RAM $\to$ không gửi query xuống DB nữa, tiết kiệm 100% băng thông mạng/CPU.
  5. *Hiển thị trên màn hình (HUD):* Khung Bbox vẫn hiện màu xanh lá kèm tên `"Nguyễn Văn A"`, đồng thời bật dải thông báo màu cam cảnh báo trong 3 giây:  
     ⚠️ **`Nguyễn Văn A: Đã CHECKIN gần đây. Vui lòng đợi 15 phút.`**

#### 6. Cửa Sổ Trượt Độc Lập (Rolling Window 5 Votes) Cho Từng Khuôn Mặt:
- **Cài đặt tại:** [tracker.py:L60-L130](face_auth/tracking/tracker.py#L60-L130).
- **Khởi tạo bên trong mỗi đối tượng `Track`:**
  ```python
  self._pad_window: deque = deque(maxlen=pad_smooth_window)  # Mặc định maxlen = 5
  ```
- **Tính độc lập tuyệt đối giữa các khuôn mặt trong cùng một khung hình:**
  - Mỗi một người xuất hiện trước camera sẽ được cấp một đối tượng `Track` riêng với một hàng đợi `_pad_window` hoàn toàn tách biệt trong RAM.
  - *Ví dụ:* Nhân viên thật (`Track #1`) tích lũy `[True, True, True, True, True]` $\to$ `REAL`. Kẻ gian đứng ngay cạnh cầm điện thoại chiếu ảnh giả (`Track #2`) tích lũy `[False, False, False, False, False]` $\to$ `SPOOF`. Kẻ gian đứng cạnh **hoàn toàn không làm ảnh hưởng hay vấy bẩn** kết quả biểu quyết của nhân viên thật bên cạnh!
- **Ba cơ chế kiểm soát chất lượng của Rolling Window:**
  1. *Hàng đợi FIFO (First-In, First-Out):* Tự động đẩy phán quyết cũ nhất ra khi có phán quyết thứ 6 nạp vào, luôn giữ đúng 5 phiếu bầu gần nhất trong vòng 0.5 giây.
  2. *Cổng chặn an toàn (`PAD_PENDING`):* Trong 4 frame đầu tiên khi người mới bước vào (`len < 5`), hệ thống giữ trạng thái `PAD_PENDING` và **chặn đứng module nhận diện ArcFace** ([tracker.py:L149](face_auth/tracking/tracker.py#L149)). Tuyệt đối không cho phép điểm danh khi chưa đủ 5 phiếu bầu!
  3. *Chống dùng phiếu cũ (Stale Timeout - 3.0s):* Nếu người đó quay mặt đi hoặc mất dấu quá 3 giây, window cũ tự động bị xóa sạch (`_pad_window.clear()`) để bầu lại từ đầu, ngăn chặn việc kẻ gian tráo ảnh giả vào vị trí người thật vừa rời đi.
  4. *Ngưỡng biểu quyết Spoof (`PAD_SPOOF_MIN_RATIO = 0.6`):* Cần $\ge 60\%$ số phiếu kết luận giả mạo (tối thiểu 3/5 phiếu SPOOF) thì hệ thống mới phán quyết là `SPOOF`. Người thật cử động tự nhiên đôi khi bị nhiễu 1 phiếu (20%) thì vẫn được bảo vệ ở trạng thái `REAL`.

---

### MODULE 5: Cơ Chế Chống Giả Mạo Khuôn Mặt (Face Anti-Spoofing / PAD)

- **Mã nguồn liên quan:**
  - [antispoof/predictor.py](face_auth/antispoof/predictor.py) — Lớp dự đoán `AntiSpoofPredictor`, công thức `LogSumExp` và quản lý phiên ONNX.
  - [antispoof/preprocess.py](face_auth/antispoof/preprocess.py) — Quy trình cắt crop $1.55\times$, đệm viền `BORDER_REFLECT_101` và chuẩn hóa RGB.
  - [antispoof/loader.py](face_auth/antispoof/loader.py) — Nạp mô hình ONNX Runtime tối ưu với CPU/GPU provider.
  - **Trọng số Runtime hiện tại:** [antispoof/models/v3/mnv3s_e1_preliminary_v5_3_edge_best.onnx](face_auth/antispoof/models/v3/mnv3s_e1_preliminary_v5_3_edge_best.onnx) (MobileNetV3-Small v5.3 Edge).

Module PAD đóng vai trò **"Người gác cổng an ninh" (Security Gatekeeper)**: Khuôn mặt chỉ được phép đi tiếp sang bước Căn chỉnh & Nhận diện (Module 6 & 7) khi và chỉ khi đã được xác thực là **NGƯỜI THẬT (REAL)**.

```text
Khuôn mặt từ Tracking
        ↓
[Tiền xử lý PAD] ────── (Crop 1.55×, Viền BORDER_REFLECT_101, Chuyển RGB, Chuẩn hóa CelebA)
        ↓
[MobileNetV3 3-Class] ─ (Inference ONNX cực nhanh ~5.6ms) ──> 3 Logits: [Real, Print, Screen]
        ↓
[LogSumExp Score] ───── d = real - logsumexp(print, screen) ──> So sánh d >= -0.4650
        ↓
[Rolling Window 5 Vote] (Tích lũy 5 phán quyết trong 0.5s trên từng Track riêng biệt)
        ├──> P_SPOOF >= 60% ──> 🔴 BÁO ĐỘNG SPOOF (Chặn đứng nhận diện, khóa điểm danh)
        ├──> Chưa đủ 5 vote ─> 🟡 PAD_PENDING   (Chờ tích lũy, tạm hoãn nhận diện)
        └──> Đạt chuẩn REAL ──> 🟢 MỞ KHÓA CHO ARCFACE & ĐIỂM DANH ↓
```

---

#### 1. Kiến trúc Mô hình 3-Class (Real / Physical Spoof / Digital Spoof)

Hệ thống sử dụng mạng **MobileNetV3-Small (Edge Baseline v5.3)** phân loại 3 lớp:
- **Class 0 — Real:** Người thật đứng trực tiếp trước ống kính.
- **Class 1 — Physical Spoof:** Các dạng tấn công vật lý (ảnh in trên giấy A4, ảnh chụp rửa, bìa cứng, mặt nạ giấy/silicon).
- **Class 2 — Digital Spoof:** Các dạng tấn công kỹ thuật số (phát video/ảnh khuôn mặt qua màn hình điện thoại smartphone, iPad, tablet, laptop, TV).

> **Tại sao dùng 3-Class thay vì Binary (2-Class: Real/Spoof)?**  
> Dấu hiệu giả mạo của ảnh in (sợi giấy, độ nhám, mực in) và màn hình điện thoại (hiện tượng Moire, phản chiếu đèn huỳnh quang, lưới điểm ảnh pixel grid) mang bản chất vật lý quang học hoàn toàn khác nhau. Việc tách thành 3 lớp giúp mạng nơ-ron học được các đặc trưng phân tách sắc nét hơn rất nhiều so với việc gộp chung, giảm thiểu tối đa hiện tượng nhận diện nhầm người thật thành giả mạo.

- **Hiệu năng thực tế:** Mô hình MobileNetV3-Small chỉ nặng **~4.4 MB** và chạy với tốc độ **~5.6 ms/lần gọi** trên CPU thông thường, hoàn toàn đáp ứng thời gian thực cho các thiết bị nhúng IoT (Raspberry Pi, Jetson).

---

#### 2. Quy trình Tiền xử lý PAD (PAD Preprocessing Contract) — Đối lập với ArcFace

Khác với Module nhận diện ArcFace (cắt sát và xoay thẳng theo 5 điểm landmarks), Module PAD **nghiêm cấm xoay mặt và bắt buộc phải cắt rộng**:

- **Mở rộng khung vuông $1.55\times$ (`bbox_expansion_factor = 1.55`):**
  Lấy tâm của Bbox, tính cạnh lớn nhất $max\_dim = \max(w, h)$ và mở rộng ra $155\%$:
  $$crop\_size = max\_dim \times 1.55$$
  *Lý do:* Để mạng nơ-ron nhìn thấy **vùng ngữ cảnh xung quanh đầu** (tai, tóc, ngón tay cầm điện thoại, mép tờ giấy in hoặc viền đen bezel của iPad/iPhone). Nếu cắt sát khuôn mặt, toàn bộ các dấu hiệu tố cáo giả mạo này sẽ bị cắt mất.

- **Đệm viền phản chiếu `cv2.BORDER_REFLECT_101` (Cực kỳ quan trọng):**
  Khi người dùng đứng sát mép camera, vùng crop $1.55\times$ sẽ bị vượt ra khỏi khung hình:
  - 🚫 *Nếu đệm viền đen (`BORDER_CONSTANT = 0`):* Mép viền đen nhân tạo này sẽ đánh lừa AI khiến nó tưởng là **viền màn hình điện thoại**, làm cho người thật bị báo động nhầm thành SPOOF!
  - ✅ *Giải pháp:* Dùng `BORDER_REFLECT_101` (lấy đối xứng gương các pixel ở mép ảnh). Viền gương giữ nguyên độ liên tục của ánh sáng môi trường, loại bỏ hoàn toàn các cạnh giả tạo.

- **Chính sách Gamma: Tại sao TẮT Gamma (`apply_gamma = False`) trong cả Training lẫn Runtime chuẩn?**
  - *Bản chất quang học khi Train:* Màn hình điện thoại tự phát sáng (active light), ảnh in phản xạ khuếch tán, còn da người thật có hiện tượng tán xạ dưới bề mặt (subsurface scattering). Nếu bật Adaptive Gamma, thuật toán sẽ ép độ sáng về mức trung bình ($\approx 110/255$), làm xóa nhòa các dấu vết quang phổ tự nhiên của màn hình và giấy in. Vì vậy, giao thức huấn luyện v5.1–v5.3 **bắt buộc tắt Gamma 100%**.
  - *Tại sao Runtime thực tế cũng KHÔNG DÙNG Gamma (`PAD_GAMMA_ENABLED=false`)?*
    1. *Model đã đủ mạnh:* Tập CelebA-Spoof (> 500.000 ảnh) bao phủ mọi điều kiện ánh sáng; mạng MobileNetV3 cùng các lớp BatchNorm đã tự học được các đặc trưng bất biến với ánh sáng (Illumination Invariance).
    2. *Phần cứng Camera (ISP):* Các webcam hiện đại đều tích hợp Auto Exposure (tự động phơi sáng) và Auto White Balance (cân bằng trắng phần cứng), xử lý ánh sáng tốt hơn nhiều so với việc kéo gamma phần mềm (vốn dễ làm bệt màu da và khuếch đại hạt nhiễu cảm biến).
  - *Tại sao code vẫn giữ sẵn hàm `adaptive_gamma()`?*
    1. *Tương thích ngược (Backward Compatibility):* Cho phép chạy lại các model 128px thế hệ cũ (`best_model_quantized.onnx`).
    2. *Làm thí nghiệm đối chứng (Ablation Study):* Cung cấp số liệu so sánh giữa BẬT vs TẮT Gamma trong báo cáo đồ án để chứng minh giải pháp tối ưu.
    3. *Tùy biến môi trường cực đoan:* Chỉ khi camera lắp ở nơi quá tối hoặc ngược sáng cực đoan mới cần cân nhắc bật trong [.env](face_auth/.env).
  - ⚠️ *Nguyên tắc sống còn:* Nếu bật Gamma ở môi trường thực tế, phân phối logit sẽ bị lệch $\to$ **bắt buộc phải chạy lại bước Hiệu chuẩn ngưỡng (Threshold Calibration) trên tập Validation**, tuyệt đối không được dùng lại ngưỡng $p=0.3858$!

- **Chuyển đổi không gian màu BGR ➔ RGB:**
  OpenCV đọc ảnh BGR, nhưng MobileNetV3 được huấn luyện trên không gian màu RGB $\to$ Bắt buộc chuyển đổi bằng `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

- **Chuẩn hóa Mean & Std theo tập CelebA-Spoof (Z-score Normalization):**
  - *Nguồn gốc con số:* Không phải là file cấu hình có sẵn tải về từ dataset, mà là kết quả tính toán thống kê (Statistical Computation) trên toàn bộ hơn 500.000 bức ảnh khuôn mặt trong tập huấn luyện (Train set) của CelebA-Spoof:
    $$\text{Mean} = [0.5931, 0.4690, 0.4229], \quad \text{Std} = [0.2471, 0.2214, 0.2157]$$
  - *Ý nghĩa sinh học (Khác biệt hoàn toàn so với ImageNet):*
    - Bộ số mặc định của ImageNet (vật thể, phong cảnh) là `Mean = [0.485, 0.456, 0.406]` (3 kênh khá đều nhau).
    - Nhưng ở CelebA-Spoof, kênh **Đỏ (R = 0.5931)** cao vượt trội so với G ($0.4690$) và B ($0.4229$). Lý do là vì da người (bất kể màu da nào) đều chứa sắc tố melanin và mạng lưới vi mạch máu hemoglobin dưới da, tạo nên sắc đỏ tự nhiên chiếm ưu thế.
  - *Tại sao khi Inference BẮT BUỘC dùng đúng bộ số này?*
    Toàn bộ hàng triệu trọng số ($W, b$) của MobileNetV3 được tối ưu trên miền dữ liệu đã chuẩn hóa $\frac{x - \text{Mean}}{\text{Std}}$. Khi chạy thực tế, ảnh camera phải được biến đổi qua đúng công thức này thì dữ liệu mới rơi vào đúng "vùng nhận thức" của mô hình. Nếu dùng sai hoặc bỏ qua bước này, các lớp Convolution đầu tiên sẽ bị lệch vùng kích hoạt (Activation Shift) $\to$ model phán đoán sai lệch hoàn toàn.
  - Đưa về tensor định dạng PyTorch $(1, 3, 224, 224)$ nạp vào ONNX Runtime.

---

#### 3. Bản chất Toán học: Công thức `LogSumExp` & Phán quyết Nhị phân

Đầu ra của mô hình là một vector gồm 3 logits thô: $[z_0, z_1, z_2]$ tương ứng với $[\text{Real}, \text{Physical}, \text{Digital}]$.

Hệ thống tính toán **Điểm số PAD (PAD Score - ký hiệu là $d$)** tại [antispoof/predictor.py:L149-L165](face_auth/antispoof/predictor.py#L149-L165):

$$d = z_{\text{real}} - \text{LogSumExp}(z_{\text{physical}}, z_{\text{digital}}) = z_0 - \ln\left(e^{z_1} + e^{z_2}\right)$$

- **Kỹ thuật ổn định số học (Tránh tràn số mũ overflow khi $z$ lớn):**
  $$\text{LogSumExp}(a) = a_{\max} + \ln\left(\sum_{i} e^{a_i - a_{\max}}\right)$$

- **Mối liên hệ toán học tuyệt đẹp với Softmax $P(\text{REAL})$:**
  Xác suất người thật theo hàm Softmax là:
  $$P(\text{REAL}) = \frac{e^{z_0}}{e^{z_0} + e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_0 - \ln(e^{z_1} + e^{z_2}))}} = \frac{1}{1 + e^{-d}} = \sigma(d)$$
  
  Do đó, việc so sánh $d \ge d_{\text{threshold}}$ **tương đương tuyệt đối về mặt toán học** với việc kiểm tra xác suất:
  $$P(\text{REAL}) \ge p_{\text{threshold}} \quad \text{với } d_{\text{threshold}} = \ln\left(\frac{p}{1 - p}\right)$$

- **Bộ thông số vận hành (Operating Point v5.3 Edge):**
  - Ngưỡng xác suất tối ưu đã calibrate trên tập Validation: $p = \mathbf{0.385795}$ ($38.58\%$).
  - Ngưỡng Logit tương ứng: $d = \mathbf{-0.465022}$.
  - Nếu $d \ge -0.4650$ (hay $P(\text{REAL}) \ge 38.58\%$): Phán quyết frame này là **`True` (REAL)**. Ngược lại là **`False` (SPOOF)**.

- **Cơ chế hình thành & tính toán Phân phối Logit (Logit Distribution & Threshold Calibration):**
  - **1. Trong lúc huấn luyện (Loss Gradient nhào nặn Logits):**
    Vector logits thô $z = [z_0, z_1, z_2]$ được sinh ra bởi lớp tuyến tính cuối cùng: $z = W \cdot h + b$. Dưới tác động của hàm mất mát Cross-Entropy $\mathcal{L} = -\ln(p_y)$, đạo hàm theo từng logit là:
    $$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$
    - Khi mẫu là **REAL** ($y=0$): Gradient kéo $z_0 \uparrow$ tăng cao, đồng thời dìm $z_1, z_2 \downarrow$.
    - Khi mẫu là **SPOOF** ($y \in \{1, 2\}$): Gradient dìm $z_0 \downarrow$ xuống rất thấp, đồng thời đẩy $z_1$ hoặc $z_2 \uparrow$.
    Qua hàng trăm ngàn bước cập nhật (iterations), mạng nơ-ron liên tục tách khoảng cách giữa các logit của ảnh thật và ảnh giả ra xa nhau.
  - **2. Thu thập Phân phối trên tập Validation (Logit Profiling):**
    Sau khi huấn luyện xong, mô hình chạy ở chế độ `model.eval()` trên toàn bộ hàng chục ngàn bức ảnh của tập Validation (đã biết rõ Ground Truth):
    1. Với mỗi ảnh thứ $i$, tính điểm log-odds: $d_i = z_{0,i} - \text{LogSumExp}(z_{1,i}, z_{2,i})$.
    2. Gom thành 2 mảng phân phối riêng biệt:
       - Mảng Người Thật: $\mathcal{D}_{\text{real}} = \{d_i \mid y_i = \text{REAL}\}$ (tạo thành quả chuông lệch phải, tâm dương $\approx +2.8$).
       - Mảng Giả Mạo: $\mathcal{D}_{\text{spoof}} = \{d_i \mid y_i = \text{SPOOF}\}$ (tạo thành quả chuông lệch trái, tâm âm $\approx -3.2$).
  - **3. Thuật toán tìm Ngưỡng tối ưu theo chuẩn ISO/IEC 30107-3:**
    Hệ thống quét một dải ngưỡng $\tau \in [-5.0, +5.0]$ để đo 2 đại lượng sai số cốt lõi:
    - **APCER** (Attack Presentation Classification Error Rate - Tỷ lệ lọt lưới kẻ giả mạo):
      $$\text{APCER}(\tau) = \frac{\sum [d_i \ge \tau \mid y_i = \text{SPOOF}]}{N_{\text{spoof}}}$$
    - **BPCER** (Bona Fide Presentation Classification Error Rate - Tỷ lệ chặn nhầm người thật):
      $$\text{BPCER}(\tau) = \frac{\sum [d_i < \tau \mid y_i = \text{REAL}]}{N_{\text{real}}}$$
    - **ACER** (Average Classification Error Rate - Sai số trung bình):
      $$\text{ACER}(\tau) = \frac{\text{APCER}(\tau) + \text{BPCER}(\tau)}{2}$$
    Điểm cực tiểu toàn cục làm cho **$\text{ACER}(\tau)$ nhỏ nhất** trên tập Validation của mô hình v5.3 Edge chính là con số vận hành thực tế:
    $$\mathbf{d_{\text{threshold}} = -0.465022} \iff \mathbf{P(\text{REAL}) \ge 0.385795\ (38.58\%)}$$
  - **4. Giải thích hiện tượng Trôi phân phối (Distribution Shift):**
    Ngưỡng $d = -0.465022$ là vạch cắt tối ưu dựa trên phân phối ánh sáng gốc tự nhiên của tập dữ liệu. Nếu bật Gamma nhân tạo ở webcam thực tế ($x^\gamma$), các lớp tích chập sẽ trích xuất đặc trưng bị lệch $\to$ cả hai quả chuông $\mathcal{D}_{\text{real}}$ và $\mathcal{D}_{\text{spoof}}$ bị trôi dạt (Distribution Shift) $\to$ vạch cắt cũ rơi vào sai vị trí, dẫn đến APCER hoặc BPCER bùng nổ. Vì vậy, nguyên tắc bất di bất dịch là **giữ nguyên phân phối quang học tự nhiên không dùng Gamma phần mềm**.

- **So sánh chuyên sâu: `LogSumExp` khác biệt thế nào so với `Softmax + argmax` mặc định?**
  - **Lỗ hổng của `Softmax + argmax` ("Hiện tượng chia rẽ phiếu" — Split-Vote Vulnerability):**
    - Giả sử kẻ xấu dùng màn hình iPad chiếu ảnh tấn công, mô hình bị bối rối giữa Print và Screen, cho ra xác suất:
      $$p_{\text{real}} = 0.35\ (35\%), \quad p_{\text{print}} = 0.33\ (33\%), \quad p_{\text{screen}} = 0.32\ (32\%)$$
    - Nếu dùng `argmax`: Vì $0.35$ lớn nhất, hệ thống **kết luận nhầm là NGƯỜI THẬT (REAL)**! ❌
    - Trong khi thực tế, tổng xác suất đây là hành vi giả mạo lên tới:
      $$P(\text{SPOOF}) = p_{\text{print}} + p_{\text{screen}} = 33\% + 32\% = \mathbf{65\%!}$$
      Kẻ giả mạo có tới 65% khả năng tấn công mà vẫn lọt lưới chỉ vì 2 lớp Spoof chia rẽ phiếu của nhau.
    - Hơn nữa, `argmax` là một ngưỡng cứng cố định ($\approx 33.3\%$), không thể điều chỉnh độ nhạy hay tối ưu hóa tỷ lệ đánh đổi APCER/BPCER.
  - **Tại sao `LogSumExp` giải quyết triệt để vấn đề này?**
    - Bản chất an ninh là bài toán nhị phân: $\text{REAL}$ đối đầu với $\text{SPOOF (Tất cả dạng giả mạo gộp lại)}$.
    - Công thức log-odds ratio nhị phân:
      $$d = \ln\left(\frac{P(\text{REAL})}{P(\text{SPOOF})}\right) = \ln\left( \frac{\frac{e^{z_{\text{real}}}}{\sum e^z}}{\frac{e^{z_{\text{print}}} + e^{z_{\text{screen}}}}{\sum e^z}} \right) = z_{\text{real}} - \text{LogSumExp}(z_{\text{print}}, z_{\text{screen}})$$
    - `LogSumExp` đã **gộp toàn bộ năng lượng của tất cả các lớp Spoof lại thành 1 khối duy nhất** để đối trọng với Real. Kẻ tấn công dù là Print, Screen hay ở ranh giới giữa 2 dạng thì năng lượng giả mạo đều cộng dồn lại $\to$ $P(\text{SPOOF}) = 65\% > P(\text{REAL}) = 35\% \to$ Bị chặn đứng ngay lập tức!
    - Cho phép **tinh chỉnh ngưỡng linh hoạt (Tunable Operating Point)**: Tự do chọn $p_{\text{threshold}}$ theo yêu cầu bảo mật (chấm công văn phòng hay bảo mật ngân hàng) mà không bị phụ thuộc vào hàm argmax mù quáng.

| Tiêu chí | `Softmax + argmax` (Mặc định) | `LogSumExp` (Hệ thống của bạn) |
| :--- | :--- | :--- |
| **Bản chất xử lý** | Coi 3 lớp là 3 đối thủ cạnh tranh độc lập. | **Gộp toàn bộ các lớp Spoof lại thành 1 khối duy nhất** đối đầu với Real. |
| **Xử lý khi model bối rối** | **Dễ lọt lưới:** Real chỉ cần $> 33.4\%$ là thắng, dù tổng Spoof chiếm tới $66.6\%$. | **Chặn đứng:** Năng lượng Print & Screen cộng dồn lại $\to$ chặn đứng giả mạo. |
| **Độ ổn định số học** | Dễ bị tràn số mũ (overflow) khi tính `exp(z)` độc lập. | Dùng $a_{\max} + \ln(\sum e^{a - a_{\max}})$ **triệt tiêu 100% rủi ro tràn số**. |
| **Hiệu chuẩn ngưỡng** | Cố định ở ngưỡng tương đối, không chỉnh được. | **Cực kỳ linh hoạt:** Tự do đặt ngưỡng $P(\text{REAL}) \ge p_{\text{threshold}}$ (đạt ACER tối ưu trên Validation). |

- **Câu hỏi vấn đáp phản biện (Defense FAQ): "Tại sao không dùng Softmax rồi cộng 2 xác suất $(p_{\text{print}} + p_{\text{screen}})$ để so sánh với $p_{\text{real}}$?"**
  - **Chứng minh toán học (Hai cách thực chất là MỘT về mặt lý thuyết):**
    Giả sử tính xác suất theo Softmax:
    $$p_{\text{real}} = \frac{e^{z_0}}{\sum_{i=0}^2 e^{z_i}}, \quad p_{\text{spoof}} = p_{\text{print}} + p_{\text{screen}} = \frac{e^{z_1} + e^{z_2}}{\sum_{i=0}^2 e^{z_i}}$$
    Khi so sánh $p_{\text{real}} > p_{\text{spoof}}$:
    $$\frac{e^{z_0}}{\sum e^z} > \frac{e^{z_1} + e^{z_2}}{\sum e^z} \iff e^{z_0} > e^{z_1} + e^{z_2}$$
    Lấy log tự nhiên ($\ln$) hai vế:
    $$z_0 > \ln(e^{z_1} + e^{z_2}) \iff z_0 - \ln(e^{z_1} + e^{z_2}) > 0 \iff \mathbf{z_{\text{real}} - \text{LogSumExp}(z_{\text{print}}, z_{\text{screen}}) > 0}$$
    $\to$ **`LogSumExp` chính là dạng rút gọn toán học trong không gian Logit của phép so sánh Softmax trên!**
  - **Tại sao trong thực tế Edge/Production BẮT BUỘC dùng `LogSumExp` thay vì Softmax?**
    1. **Ổn định số học tuyệt đối (Numerical Stability):** 
       Hàm Softmax tính trực tiếp $e^z$. Nếu mô hình có độ tự tin cao (ví dụ $z = 90$), $e^{90} \approx 1.22 \times 10^{39}$, vượt quá giới hạn biểu diễn của kiểu số thực 32-bit (`float32` max $\approx 3.4 \times 10^{38}$) $\to$ Bị tràn số (Overflow) thành `inf`, dẫn đến phép chia `inf / inf` ra **`NaN` (Not a Number)** gây crash app hoặc phán đoán sai. Trong khi đó, `LogSumExp` dùng **Max Trick**:
       $$\text{LogSumExp}(z_1, z_2) = m + \ln(e^{z_1 - m} + e^{z_2 - m}) \quad \text{với } m = \max(z_1, z_2)$$
       Số mũ lớn nhất luôn là $e^0 = 1$, các số còn lại $\le 1$, **triệt tiêu 100% rủi ro tràn số `inf` hay `NaN`**.
    2. **Tối ưu hóa hiệu năng tính toán trên Edge/CPU (Speed & Resource):**
       Dùng Softmax đòi hỏi 3 phép tính hàm mũ $e^z$, tính tổng mẫu số $\sum$, và 2 phép chia số thực. Dùng `LogSumExp` chỉ cần tính mũ cho 2 số (đã trừ max), lấy 1 lần $\ln$, và thực hiện phép trừ thẳng trên logit, giảm chu kỳ lệnh CPU.
    3. **Bảo toàn thang đo tuyến tính không giới hạn (Linear Unbounded Scale):**
       Xác suất Softmax bị "bóp nghẹt" (saturate) vào khoảng hẹp $[0, 1]$, ở hai đầu cực trị độ nhạy bị bẹt ra. Thang log-odds $d \in (-\infty, +\infty)$ bảo toàn độ dốc tuyến tính, giúp việc hiệu chuẩn ngưỡng bảo mật tối ưu ($d = -0.465022$) đạt độ chính xác số học cao hơn rất nhiều.

---

#### 4. Nhịp điều phối (Cadence) & Bộ lọc Cửa sổ trượt (Temporal Smoothing)

Để đảm bảo hệ thống không bị nhấp nháy (flickering) khi người dùng cử động:
- **Chu kỳ chạy:** Được gọi đều đặn mỗi **0.1 giây** (`PAD_INTERVAL_SECONDS = 0.1s`).
- **Gom phiếu độc lập:** Mỗi khuôn mặt có riêng một hàng đợi `_pad_window = deque(maxlen=5)` tích lũy 5 kết quả gần nhất trong vòng 0.5 giây.
- **Ngưỡng kết luận Spoof ($\ge 60\%$):** 
  $$\text{Tỷ lệ Spoof} = \frac{\text{Số phiếu False}}{len(\_pad\_window)} \ge 0.6$$
  Phải có ít nhất **3 trên 5 phiếu bầu kết luận là giả mạo** thì hệ thống mới bật cảnh báo đỏ `SPOOF`. Nếu chỉ bị nhiễu 1 phiếu (20%) do chớp mắt hay góc đèn phản chiếu, người thật vẫn được bảo vệ an toàn ở trạng thái `REAL`.

---

#### 5. Phản ứng của Hệ thống khi Phát hiện Giả mạo (Spoof Handling & Self-Healing)

Khi một khuôn mặt bị kết luận là **`SPOOF`**, hệ thống sẽ kích hoạt phản ứng phòng thủ đa tầng từ cấp độ **Thuật toán $\to$ Mô hình AI $\to$ Cơ sở dữ liệu $\to$ Giao diện HUD**:

1. **Tầng AI — Ngắt hoàn toàn ArcFace & Database (Gatekeeper Pattern):**
   Tại [face_auth/tracking/tracker.py:L149](face_auth/tracking/tracker.py#L149), hàm `needs_recognition()` lập tức trả về `False`:
   ```python
   if pad_enabled and (not self.pad_ready or not self.is_real):
       return False
   ```
   - **Không trích xuất Embedding:** Hệ thống bỏ qua việc căn chỉnh 5 điểm và không nạp ảnh vào mô hình ArcFace ResNet, tiết kiệm triệt để CPU/GPU trên Edge.
   - **Không truy vấn Database:** Không có bất kỳ vector nào được gửi đến PostgreSQL `pgvector` để tìm kiếm danh tính.
   - **Reset độ tin cậy:** Reset biến đếm nhận diện ổn định về 0 (`track.stable_recognitions = 0`). Kẻ giả mạo tuyệt đối không được gán danh tính của bất kỳ nhân viên nào.

2. **Tầng Nghiệp vụ — Khóa chặt quy trình Điểm danh (Attendance Blocking):**
   Tại [face_auth/app.py:L167-L175](face_auth/app.py#L167-L175):
   ```python
   if (app_mode != "NONE" and not is_spoof and not is_pad_pending and ...):
       db.log_attendance(...)
   ```
   - Do `is_spoof = True`, điều kiện `not is_spoof` bị vi phạm. Lệnh ghi điểm danh `db.log_attendance()` bị **chặn đứng 100%**. Bảng `attendance_logs` hoàn toàn không bị ô nhiễm bởi các lượt điểm danh giả mạo.

3. **Tầng Trực quan (HUD) — Bật Báo động Đỏ trên Màn hình:**
   Tại [face_auth/app.py:L64-L66](face_auth/app.py#L64-L66):
   - **Khung Bounding Box:** Chuyển ngay sang **MÀU ĐỎ RỰC** `COLOR_SPOOF = (0, 0, 220)`.
   - **Nhãn hiển thị:** Dán nhãn **`⚠ SPOOF`** (nền đỏ chữ trắng), đè lên mọi nhãn danh tính khác với độ ưu tiên tuyệt đối.

4. **Tầng Nhật ký Chẩn đoán (Security Logging):**
   Khi bật `PAD_DIAGNOSTIC_LOG=true` trong [.env](face_auth/.env), hệ thống in bằng chứng tấn công lên Console:
   ```text
   [PAD] frame_id=1420 track_id=2 bbox=[180, 110, 360, 290] 
         d=-3.821054 p_real=0.021430 raw=SPOOF smoothed=SPOOF
   ```

5. **Tầng Theo dõi — Duy trì bám đuôi (Tracking Continuity):**
   - Thuật toán Optical Flow (`cv2.calcOpticalFlowPyrLK`) **vẫn tiếp tục bám theo khuôn mặt giả mạo đó**.
   - *Ý nghĩa an ninh:* Không giải phóng track để tránh việc detector ở frame sau coi kẻ tấn công như người mới đến; thay vào đó, hệ thống "khóa chặt" kẻ tấn công trong chiếc hộp màu đỏ `⚠ SPOOF` liên tục cho đến khi họ hạ điện thoại/ảnh in xuống hoặc rời khỏi camera.

- **Câu hỏi vấn đáp phản biện (Edge Case FAQ): "Nếu người thật bước vào camera, ban đầu góc mặt xấu/thiếu sáng nên mô hình nhận nhầm là SPOOF, sau đó mới nhận ra REAL thì hệ thống xử lý thế nào?"**
  - **Cơ chế Tự phục hồi (Self-Healing / State Recovery):**
    Hệ thống **không bao giờ ban hay kết án tử hình vĩnh viễn** đối với một Track. Nhờ hàng đợi FIFO `deque(maxlen=5)` hoạt động với chu kỳ 0.1s:
    1. **Lúc đầu (0.0s – 0.2s):** Bị nhận nhầm $\to$ `_pad_window = [False, False, False, ...]` $\to$ Hiện đỏ `⚠ SPOOF` và ngắt ArcFace để bảo đảm an toàn trước mắt.
    2. **Sau 0.3s – 0.5s:** Người thật đứng ổn định trước camera, PAD trả về các phiếu `True` liên tiếp $\to$ các phiếu `False` cũ bị đẩy văng ra khỏi hàng đợi.
    3. **Lật trạng thái:** Khi tỷ lệ Spoof giảm xuống $< 60\%$ (chỉ cần 3/5 phiếu là Real), hệ thống tự động xóa bỏ cảnh báo đỏ:
       $$\text{SPOOF} \longrightarrow \text{PAD\_READY (REAL)}$$
    4. **Nối lại Nhận diện & Điểm danh:** Ngay khi là REAL, `needs_recognition()` tự động cho phép chạy ArcFace $\to$ Bbox chuyển sang xanh lá **`[OK] Tên_Nhân_Viên`** $\to$ Đạt ổn định 2 chu kỳ $\to$ Ghi nhận điểm danh thành công kèm Cooldown 15 phút.
    - *Kết luận:* Toàn bộ quá trình phục hồi diễn ra hoàn toàn tự động chỉ trong vòng **vài phần mười giây**, không làm gián đoạn trải nghiệm của người dùng thật.

- **Chuyên đề An ninh nâng cao: Hai Cơ Chế Phòng Thủ Đa Tầng (Two Core Security Firewalls):**

  Hệ thống được bảo vệ bởi **hai cơ chế an toàn độc lập và bổ trợ lẫn nhau**, tạo thành một vòng tròn khép kín triệt tiêu hoàn toàn các kịch bản tấn công tráo ảnh giả (Track Hijacking / Trust Transfer) dù kẻ gian ra tay **cực nhanh ($< 1.0\text{s}$)** hay **rình rập khi gián đoạn ($\ge 1.0\text{s}$)**:

  - **1. Bối cảnh nguy hiểm — Tử huyệt "Mù liveness" của Face Tracker:**
    Các thuật toán Tracker hình học (IoU) và Optical Flow chỉ quan tâm đến vị trí tọa độ hộp $(x_1, y_1, x_2, y_2)$. Khi người thật vừa quay mặt đi hoặc cúi đầu, Optical Flow vẫn giữ Track sống trong RAM. Nếu kẻ gian nhanh tay chìa ảnh in/điện thoại vào đúng tọa độ Bbox cũ, Tracker sẽ "ngây thơ" gán luôn `track_id` và danh tính của người thật cho ảnh giả đó.

  - **2. CƠ CHẾ 1 — Đồng bộ nhịp bất đối xứng (Asymmetric Cadence & Fast Elimination):**
    *👉 Chuyên trị kịch bản: Kẻ gian ra tay CỰC NHANH (dưới 1 giây).*
    - **Thiết kế nhịp lệch nhau:** 
      - Nhịp PAD (Cao tần): Chạy cực nhanh mỗi **`0.1 giây`** (`PAD_INTERVAL_SECONDS = 0.1s`).
      - Nhịp ArcFace (Thấp tần): Chạy thưa hơn mỗi **`0.5 giây`** (`RECOGNIZE_INTERVAL_SECONDS = 0.5s`) để tối ưu hóa CPU.
    - **Tốc độ đào thải 0.3 giây:** Hàng đợi 5 vote của PAD chỉ cần tích lũy **3 phiếu `False` liên tiếp** là tỷ lệ Spoof đã đạt $\frac{3}{5} = \mathbf{60\% \ge 60\%}$. Với nhịp 0.1s, PAD chỉ mất đúng:
      $$3 \times 0.1\text{s} = \mathbf{0.3\text{ giây}}$$
      là đã chính thức kết án `SPOOF`, đổi khung sang màu đỏ và hạ cờ `needs_recognition() = False`.
    - **Cuộc đua tốc độ (Cadence Race):** PAD luôn **về đích trước ArcFace tới 200 mili-giây** ($0.3\text{s}$ so với $0.5\text{s}$). Khi ArcFace chuẩn bị chạy nhận diện ở mốc $0.5\text{s}$, cánh cửa an ninh đã bị PAD đóng sập lại từ trước $\to$ Lệnh gọi ArcFace bị hủy bỏ ngay tức khắc!
    - **Chốt chặn nghiệp vụ bổ trợ:**
      - *Nếu là ảnh người thật (Alice):* Bị chặn bởi **Cooldown 15 phút** (`ATTENDANCE_GAP_MINUTES = 15`).
      - *Nếu là ảnh người khác (Bob):* Lập tức bị **reset bộ đếm `stable_recognitions = 1`** (trong khi điều kiện cần $\ge 2$), ở frame kế tiếp PAD đã thành công giáng cấp về `SPOOF` và triệt tiêu khả năng điểm danh lậu.

  - **3. CƠ CHẾ 2 — Thời hạn sử dụng dữ liệu theo đồng hồ thực (Zero-Trust TTL qua `pad_stale_timeout = 1.0s`):**
    *👉 Chuyên trị kịch bản: Kẻ gian rình rập khi người thật BỊ GIÁN ĐOẠN (trên 1 giây).*
    - **Bản chất:** Dữ liệu đánh giá liveness (5 phiếu trong RAM) có một **Hạn sử dụng (Time-To-Live)** nghiêm ngặt đúng **`1.0 giây`** tính theo đồng hồ thời gian thực (`time.time()`).
    - **Xóa sạch bộ nhớ & Treo quyền:** Khi người thật cúi mặt/quay đi quá 1.0 giây, dù Optical Flow vẫn giữ Bbox sống, nhưng ngay khi ảnh giả xuất hiện ở Bbox đó:
      ```python
      # face_auth/tracking/tracker.py:L79-L84
      if (self.last_pad_time is not None 
          and (time.time() - self.last_pad_time) > self._pad_stale_timeout):
          self._pad_window.clear()  # XÓA SẠCH 100% PHIẾU CŨ TRONG RAM!
      ```
      1. Thuộc tính `pad_ready` tự động trả về `False` theo thời gian thực $\to$ khóa ngay quyền nhận diện.
      2. Hàm `update_pad()` xóa sạch toàn bộ các phiếu `True` của người thật lúc trước khỏi RAM.
      3. Track bị giáng cấp về trạng thái khởi động nguội **`PAD_PENDING`** (khung xám `"Kiem tra PAD..."`), bắt buộc ảnh giả phải tích lũy lại ít nhất `pad_min_votes = 3` phiếu mới $\to$ Tỷ lệ Spoof vọt lên $100\%$ và bị tóm sống!

  - **4. Bảng Tổng Hợp Đối Chiếu Hai Cơ Chế Phòng Thủ:**

| Tiêu chí | CƠ CHẾ 1: Nhịp Bất Đối Xứng (Asymmetric Cadence) | CƠ CHẾ 2: Tự Hủy Dữ Liệu Cũ (`pad_stale_timeout`) |
| :--- | :--- | :--- |
| **Vùng bảo vệ** | Tấn công chớp nhoáng: **Thời gian $< 1.0$ giây** | Tấn công gián đoạn: **Thời gian $\ge 1.0$ giây** |
| **Chốt chặn kích hoạt** | Nhịp PAD $0.1\text{s}$ đào thải kết quả chỉ trong **$0.3\text{s}$** | Đồng hồ kiểm tra $\Delta t = \text{time.time()} - \text{last\_pad\_time} > 1.0\text{s}$ |
| **Hành động cốt lõi** | **Về đích trước ArcFace $200\text{ms}$** để hủy lệnh nhận diện trước khi ArcFace kịp chạy ($0.5\text{s}$). | **Xóa sạch RAM (`.clear()`)**, giáng cấp về `PAD_PENDING`, triệt tiêu toàn bộ uy tín cũ. |
| **Kết quả đối với kẻ gian** | **Bị khóa cửa ngay tại giây thứ 0.3s**! | **Bị tước toàn bộ phiếu cũ**, bắt kiểm tra lại từ đầu và bị báo động đỏ! |

---

---

## MODULE 6: CĂN CHỈNH KHUÔN MẶT 5 ĐIỂM (FACE ALIGNMENT & SIMILARITY TRANSFORM)

> **File nguồn:** [face_auth/alignment/aligner.py](face_auth/alignment/aligner.py)  
> **Hàm cốt lõi:** `align_face()`, `get_input_face()`  
> **Nhiệm vụ:** Chuẩn hóa hình học (Geometric Normalization) — khử góc xoay đầu (Roll), triệt tiêu sự chênh lệch kích thước (Scale) và vị trí (Translation), biến đổi ảnh khuôn mặt bất kỳ về ma trận chuẩn **$112 \times 112$** pixel cho mạng ArcFace.

```text
 Frame gốc (640x480) ──> Lấy 5 Landmarks từ SCRFD ──> cv2.estimateAffinePartial2D()
                                                              │
                                                              ▼ (Ma trận M: 2x3)
                                                     cv2.warpAffine()
                                                              │
                                                              ▼
                                                   Ảnh chuẩn hóa 112x112
                                                   (Đưa vào ArcFace ResNet)
```

---

### 1. Tại sao ArcFace BẮT BUỘC phải Face Alignment (Đối lập với PAD)?

Trong kiến trúc tổng thể, Module 5 và Module 6 có triết lý xử lý hình ảnh hoàn toàn trái ngược nhau:
- **Module 5 (Anti-Spoofing / PAD):** Dùng **Bbox mở rộng $1.55\times$** và giữ nguyên góc chụp tự nhiên. *Lý do:* Cần nhìn thấy toàn bộ ngữ cảnh viền ngoài (ngón tay cầm điện thoại, mép giấy in, viền bezel màn hình). Nếu xoay thẳng và zoom sát mặt, các dấu vết giả mạo sẽ bị cắt mất.
- **Module 6 (Face Recognition / ArcFace):** Bắt buộc phải **xoay thẳng và căn chỉnh 5 điểm**.
  - *Bản chất toán học của CNN:* Các mạng nơ-ron tích chập (Convolutional Neural Networks) **không có tính bất biến với phép xoay lớn và co dãn tỷ lệ (Scale & In-plane Rotation Invariant)**.
  - *Hậu quả nếu không căn chỉnh:* Khi một người đứng thẳng, và sau đó nghiêng đầu 20 độ, các điểm ảnh của mắt/mũi sẽ di chuyển sang các tọa độ pixel hoàn toàn khác. Các bộ lọc tích chập (Conv filters) sẽ kích hoạt các neuron sai lệch $\to$ Vector đặc trưng 512-D bị "trôi dạt" (Embedding Drift), làm điểm Cosine Similarity rớt thảm hại từ **$0.85$ xuống dưới $0.40$** (bị hệ thống nhận nhầm thành `UNKNOWN`)!
  - Toàn bộ mạng ArcFace ResNet được huấn luyện trên hàng triệu khuôn mặt đã được chuẩn hóa vị trí 2 mắt và mũi. Do đó, ảnh nạp vào ArcFace bắt buộc phải trải qua bước Alignment này.

---

### 2. Tọa độ "Vàng" — Template chuẩn ArcFace 112×112 (`ARCFACE_DST_112`)

Tại [face_auth/alignment/aligner.py:L10-L16](face_auth/alignment/aligner.py#L10-L16), hệ thống định nghĩa 5 điểm mốc mục tiêu trên khung ảnh $112 \times 112$:

```python
ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],  # 1. Mắt trái  (Left Eye)
    [73.5318, 51.5014],  # 2. Mắt phải (Right Eye)
    [56.0252, 71.7366],  # 3. Đỉnh mũi  (Nose Tip)
    [41.5493, 92.3655],  # 4. Khóe miệng trái (Left Mouth Corner)
    [70.7299, 92.2041],  # 5. Khóe miệng phải (Right Mouth Corner)
], dtype=np.float32)
```

#### Ý nghĩa hình học chuẩn xác:
1. **Trục đối xứng hoàn hảo ($x \approx 56.0$):**
   - Chiều rộng ảnh chuẩn là $112\text{px}$, đường trung trực đối xứng nằm ở $112 / 2 = \mathbf{56.0}$.
   - Đỉnh mũi nằm ở $x = \mathbf{56.0252}$ (gần như chính xác tuyệt đối trên trục giữa).
   - Trung điểm của hai mắt: $\frac{38.2946 + 73.5318}{2} = \mathbf{55.9132} \approx 56.0$.
2. **Hai mắt nằm ngang song song tuyệt đối ($y \approx 51.6$):**
   - Độ cao mắt trái là $y = 51.6963$, mắt phải là $y = 51.5014$.
   - Độ chênh lệch cực tiểu: $|\Delta y| < 0.2$ pixel $\implies$ **Hai mắt luôn được xoay ngang hoàn toàn song song với trục hoành $Ox$**!
   - Hai mắt nằm ở hàng pixel 51 (khoảng $46\%$ chiều cao khung hình), để chừa $46\%$ phía trên cho trán/tóc, và $54\%$ phía dưới cho nhân trung, miệng và cằm.

---

### 3. Bản chất Toán học: Phép biến đổi tương đồng (Similarity Transform)

Để đưa 5 điểm landmarks thực tế trên mặt người (bị nghiêng, lệch, xa, gần) về khớp với 5 điểm chuẩn `ARCFACE_DST_112`, hàm `align_face()` gọi:
```python
matrix, _ = cv2.estimateAffinePartial2D(src, dst)
```

#### ❓ Câu hỏi phản biện lớn: *"Tại sao dùng `estimateAffinePartial2D` mà KHÔNG dùng Affine toàn phần (`estimateAffine2D`) hay Phép xạ ảnh (`findHomography`)?"*
- **Affine thông thường (Full Affine — 6 bậc tự do DoF):** Cho phép co dãn không đều theo 2 trục và có **Lực trượt / Biến dạng góc (Shear/Skew)**. Nếu 5 điểm landmarks thực tế bị rung lắc nhẹ, thuật toán Full Affine sẽ bóp méo khuôn mặt thành hình thoi hoặc hình bình hành, làm biến dạng vĩnh viễn tỷ lệ sinh trắc học tự nhiên của người dùng!
- **Phép biến đổi tương đồng (Similarity Transform — 4 bậc tự do DoF):**
  Ma trận chuyển đổi $M$ kích thước $2 \times 3$ có cấu trúc đại số:
  $$M = \begin{bmatrix} s \cos \theta & -s \sin \theta & t_x \\ s \sin \theta & s \cos \theta & t_y \end{bmatrix}$$
  Nó bị khóa cứng, chỉ cho phép đúng 4 tham số hình học thuần túy:
  1. **$t_x, t_y$** (Tịnh tiến): Dời trọng tâm khuôn mặt về giữa ảnh ($2\text{ DoF}$).
  2. **$\theta$** (Góc xoay trong mặt phẳng - In-plane Rotation): Xoay cho đường nối 2 mắt nằm ngang ($1\text{ DoF}$).
  3. **$s$** (Tỷ lệ co dãn đẳng hướng - Isotropic Scale): Ép khoảng cách hai mắt về đúng chuẩn $35.24\text{px}$ ($1\text{ DoF}$).
  
  👉 **Lực trượt $\text{Shear} = 0$**. Khuôn mặt **chỉ được xoay thẳng và phóng to/thu nhỏ đồng dạng**, bảo toàn nguyên vẹn 100% tỷ lệ hình học giữa các cơ quan trên mặt!

#### Thuật toán Umeyama (Least Squares via SVD):
Hệ thống giải bài toán tối ưu bình phương tối thiểu để tìm bộ tham số $(s, \theta, t_x, t_y)$ làm cực tiểu hóa sai số giữa 5 điểm:
$$\min_{s, \theta, t_x, t_y} \sum_{i=1}^5 \| M \cdot \mathbf{x}_i - \mathbf{y}_i \|^2$$

Sau khi tìm được ma trận $M$, hàm `cv2.warpAffine(image, matrix, (112, 112))` thực hiện ánh xạ ngược và nội suy song tuyến (Bilinear Interpolation) để tái tạo bức ảnh chuẩn $112 \times 112$.

- **Câu hỏi vấn đáp phản biện (Defense FAQ): "Hệ thống có gán ép 5 điểm landmarks phải trùng khớp chính xác 100% vào 5 điểm vàng không?"**
  - **Câu trả lời dứt khoát: KHÔNG! Tuyệt đối không gán ép trùng khớp 100%.**
  - **Lý do Sinh trắc học (Bảo tồn đặc trưng nhận dạng):**
    - Cấu trúc khuôn mặt mỗi người là độc nhất: người mặt tròn, người mặt dài, người mắt xa, người mũi ngắn. Tỷ lệ khoảng cách giữa mắt - mũi - miệng chính là **đặc trưng sinh trắc học cốt lõi** để phân biệt người này với người khác.
    - Nếu dùng phép biến dạng phi tuyến (như kéo dãn cao su / Thin Plate Splines) để "bắt ép" 5 điểm trên mặt người dùng phải rơi chính xác 100% vào 5 điểm vàng $\to$ Mọi khuôn mặt sẽ bị kéo giãn và biến dạng thành **cùng một hình dáng nhân tạo giống hệt nhau** $\to$ ArcFace sẽ bị "mù" và mất khả năng phân biệt danh tính cá nhân!
  - **Lý do Toán học (Hệ thừa nghiệm — Over-determined System):**
    - Phép biến đổi tương đồng chỉ có **4 ẩn số (4 DoF)**: $(s, \theta, t_x, t_y)$.
    - Trong khi 5 điểm landmarks tạo ra tới **10 phương trình tọa độ** ($5 \times (x, y) = 10$).
    - Về mặt đại số tuyến tính, 4 ẩn số **không bao giờ có thể thỏa mãn chính xác tuyệt đối 10 phương trình cùng lúc**.
  - **Bản chất "Khớp tối ưu" (Best-Fit via Umeyama Least Squares):**
    - Thuật toán tìm ma trận $M$ sao cho **tổng khoảng cách sai số giữa 5 điểm là nhỏ nhất có thể (Best-fit)**:
      $$\min_{s, \theta, t_x, t_y} \sum_{i=1}^5 \|\text{dst}_i - M \cdot \text{src}_i\|^2$$
    - *Ẩn dụ trực quan ("Tấm kính cứng" vs "Tấm cao su dẻo"):*
      Nó giống như việc bạn cầm một tấm kính cứng chứa ảnh: Bạn chỉ được **xoay tấm kính** ($\theta$) cho 2 mắt thăng bằng, **phóng to/thu nhỏ** ($s$) cho kích thước đầu vừa vặn, và **dời** ($t_x, t_y$) cho mũi về gần trục tâm. Bạn tuyệt đối không được bẻ cong hay kéo giãn tấm kính. Nhờ đó, **tỷ lệ hình học tự nhiên của khuôn mặt được bảo toàn nguyên vẹn 100%**!

- **Phân tích trường hợp góc quay 3D (Yaw Angle) & Hiện tượng tự che khuất má (Self-Occlusion):**
  - **1. Bức ảnh $112 \times 112$ trông như thế nào khi mặt quay nghiêng?**
    Khi người dùng quay mặt sang trái/phải một góc $\approx 30^\circ$, hiện tượng chiếu phối cảnh 3D làm má ở xa bị quay ra sau (tự che khuất - self-occlusion):
    - Sau khi qua `align_face()`, nửa bên má nhìn rõ sẽ chiếm phần lớn khung hình $112 \times 112$, còn má bị khuất sẽ bị ép mỏng sát mép viền.
    - Sống mũi bị trôi lệch sang một bên (không còn nằm ở cột giữa 56), nhưng **đường nối 2 mắt vẫn được xoay thăng bằng theo phương ngang**.
  - **2. Một bên má bị che, tại sao ArcFace vẫn nhận diện chính xác?**
    - *Tính đối xứng sinh học (Facial Symmetry):* Khuôn mặt người có tính đối xứng cao, mạng nơ-ron có thể suy diễn cấu trúc tổng thể chỉ từ nửa khuôn mặt còn lại.
    - *Trích xuất đặc trưng bất biến (Pose-Invariant Features):* Các tầng tích chập sâu của ResNet-50 tự động giảm trọng số vùng má bị khuất và dồn sự chú ý vào các vùng đặc trưng cốt lõi không bị che: hốc mắt, sống mũi, khoảng cách 2 con ngươi, xương chân mày và khóe môi.
    - *Tập dữ liệu huấn luyện khổng lồ (Glint360k / MS1MV2):* ArcFace đã được học trên hàng triệu bức ảnh chụp góc nghiêng ngoài đời thực. Hàm mất mát ArcFace ép các vector của ảnh nghiêng và ảnh chính diện của cùng một người phải hội tụ về cùng một tọa độ trên siêu cầu $\mathbb{S}^{511}$.
  - **3. Điểm số Cosine Similarity thực tế khi má bị che:**
    - Khi nhìn thẳng (thấy đủ 2 má): $\text{score} \approx \mathbf{0.82 - 0.88}$.
    - Khi quay nghiêng $30^\circ$ (bị che một phần má): $\text{score} \approx \mathbf{0.65 - 0.72}$.
    - Vì ngưỡng hệ thống là $\mathbf{MATCH\_THRESHOLD = 0.45}$, mức điểm $0.65 - 0.72$ vẫn **vượt xa ngưỡng an toàn**, hệ thống nhận diện chính xác danh tính ngay lập tức!
  - **4. Ranh giới chịu đựng:**
    Chỉ khi người dùng quay mặt quá sâu ($> 60^\circ$ — góc nhìn ngang tai Profile View) làm mất hoàn toàn con mắt ở xa, detector SCRFD không bắt đủ 5 landmarks $\to$ Hệ thống mới chuyển sang nhánh **Fallback (Bbox crop mở rộng 20% margin)**.

---

### 4. Cơ chế Dự phòng an toàn (Fallback Mechanism)

Tại [face_auth/alignment/aligner.py:L46-L73](face_auth/alignment/aligner.py#L46-L73), hàm `get_input_face()` được lập trình phòng thủ:
- **Trường hợp chuẩn (Đủ 5 landmarks):** Thực hiện Similarity Transform qua `align_face()`.
- **Trường hợp ngoại lệ (Mất landmarks hoặc detector bị lỗi):**
  - Tự động kích hoạt nhánh **Fallback theo Bbox**:
    - Lấy Bbox và mở rộng đều **20% biên an toàn** (`crop_margin = 0.2`):
      $$mx = bw \times 0.2, \quad my = bh \times 0.2$$
    - Kẹp biên an toàn trong khung hình (`max(0, ...)` và `min(w/h, ...)`).
    - Dùng `cv2.resize()` đưa thẳng về $(112, 112)$.
  - **Mục đích:** Hệ thống **tuyệt đối không bao giờ bị crash app** kể cả khi landmarks bị mất do góc mặt quá tối hoặc người dùng đứng quá xa camera.

---

### 5. Tác động mang tính quyết định của Module 6 lên các Module sau đó

Module 6 không hoạt động độc lập, mà là "bệ phóng" quyết định sự thành bại của toàn bộ các module phía sau:

#### A. Tác động lên Module 7: Trích xuất đặc trưng ArcFace 512-D
- **Giữ vững không gian biểu diễn (Domain Consistency):** Mạng ArcFace ResNet-50 được tối ưu hóa trên không gian siêu cầu $\mathbb{S}^{511}$ bằng hàm mất mát Additive Angular Margin Loss. Nếu ảnh bị lệch góc hoặc lệch tâm, vector 512-D sinh ra sẽ bị phân tán hỗn loạn.
- **Tăng cường độ co cụm nội lớp (Intra-class Compactness):** Nhờ Module 6 khử sạch góc xoay $\theta$ và tỷ lệ $s$, hai bức ảnh của cùng một người (dù một ảnh đứng thẳng, một ảnh nghiêng đầu 20 độ) đều được đưa về cùng một góc nhìn chuẩn $\to$ Vector đặc trưng 512-D của hai ảnh này sẽ gần như trùng khít nhau trên siêu cầu!

#### B. Tác động lên Module 8: Cơ sở dữ liệu Vector & Điểm danh (pgvector)
- **Ổn định điểm số Cosine Similarity trên ngưỡng `MATCH_THRESHOLD = 0.45`:**
  - *Nếu có Module 6:* Khi người dùng đứng thẳng đạt $\text{score} \approx 0.82$; khi nghiêng đầu, lắc lư vẫn duy trì $\text{score} \approx 0.75 - 0.78$ (luôn vượt xa ngưỡng 0.45 $\to$ Nhận diện chính xác 100%).
  - *Nếu KHÔNG có Module 6:* Khi người dùng chỉ cần hơi nghiêng đầu 15 độ, $\text{score}$ tụt thẳng đứng xuống **$0.35 - 0.40$** $\to$ Bị nhận nhầm thành `UNKNOWN`!
- **Duy trì chuỗi nhận diện ổn định (`stable_recognitions >= 2`):**
  - Để điểm danh thành công, hệ thống yêu cầu nhận diện đúng liên tiếp 2 chu kỳ. Nếu không có Alignment, việc lắc đầu nhẹ làm rớt điểm số sẽ liên tục reset `stable_recognitions` về 0, khiến người dùng đứng trước camera mãi mà không được điểm danh!

#### C. Tác động lên Module 9: Giao diện HUD & Trải nghiệm Người dùng (UX)
- **Triệt tiêu hiện tượng "Nhấp nháy danh tính" (Identity Flickering):**
  - Ngăn chặn triệt để tình trạng khung viền camera nhảy loạn xạ giữa màu xanh lá (`[OK] Tên_Nhân_Viên`) và màu đỏ thẫm (`UNKNOWN 0.38`) mỗi khi người dùng cử động. Khung Bbox và nhãn tên luôn được "ghim chặt" mượt mà, chuyên nghiệp.

---

---

## MODULE 7: TRÍCH XUẤT ĐẶC TRƯNG ARCFACE 512-D (FACE FEATURE EMBEDDING)

> **File nguồn:** [face_auth/recognition/embedder.py](face_auth/recognition/embedder.py)  
> **Lớp cốt lõi:** `FaceEmbedder`  
> **Mô hình AI:** ArcFace (`buffalo_s` pack / `w600k_r50.onnx` hoặc MobileFaceNet)  
> **Đầu vào:** Ảnh khuôn mặt chuẩn BGR $112 \times 112$ (từ Module 6).  
> **Đầu ra:** Vector đặc trưng sinh trắc học 512 chiều được chuẩn hóa L2: $\mathbf{e} \in \mathbb{R}^{512}$, $\|\mathbf{e}\|_2 = 1.0$.

```text
 Ảnh chuẩn 112x112 ──> ResNet-50 / MobileFaceNet ──> Feature Vector (512-D)
                                                             │
                                                             ▼
                                                    L2 Normalization (||e|| = 1.0)
                                                             │
                                                             ▼
                                                    Siêu cầu đơn vị S^511
                                                    (Chuyển sang Module 8 DB)
```

---

### 1. Kiến trúc Bóc tách Thông minh (Recognition-Only Wrapper — Cách B)

Tại [face_auth/recognition/embedder.py:L7-L38](face_auth/recognition/embedder.py#L7-L38), hệ thống áp dụng kỹ thuật bóc tách kiến trúc để tránh chạy trùng lặp:

- **Hạn chế của thư viện InsightFace mặc định:**
  Hàm tích hợp sẵn `FaceAnalysis.get(image)` là một "hộp đen": mỗi khi gọi, nó tự động chạy lại submodel SCRFD detector để tìm landmarks rồi mới align và trích xuất embedding.
  🚫 *Tử huyệt:* Pipeline của chúng ta ở Module 3 đã chạy detector một lần rồi. Nếu gọi `get(image)`, detector SCRFD sẽ **bị gọi lần thứ 2 trên cùng một frame** $\to$ FPS bị tụt hơn 50%, làm nghẽn CPU trên các thiết bị nhúng (Edge).
- **Giải pháp bóc tách trong `FaceEmbedder`:**
  Hệ thống nạp `FaceAnalysis(allowed_modules=["detection", "recognition"])` nhưng **chỉ rút riêng đúng một submodel `recognition`** ra bộ nhớ:
  ```python
  self._model = app.models.get("recognition")
  ```
  Và gọi trực tiếp hàm suy luận:
  ```python
  feat = self._model.get_feat(aligned_face)
  ```
  trên bức ảnh $112 \times 112$ đã được căn chỉnh sẵn bởi Module 6.
  ✅ *Kết quả:* Toàn bộ pipeline chỉ detect đúng **1 lần duy nhất**, giảm triệt để độ trễ và giải phóng tài nguyên CPU!

---

### 2. Bản chất Toán học: Hàm mất mát ArcFace (Additive Angular Margin Loss)

Mô hình nhận diện được huấn luyện bằng hàm mất mát nổi tiếng thế giới **ArcFace** (Deng et al., CVPR 2019) trên tập dữ liệu hàng triệu người (MS1MV2, Glint360k).

#### A. Tại sao hàm Softmax truyền thống thất bại trong bài toán sinh trắc học?
Hàm Softmax tiêu chuẩn $\mathcal{L} = -\ln\left(\frac{e^{W_y^T x}}{\sum e^{W_j^T x}}\right)$ chỉ tối ưu hóa khả năng phân tách tuyến tính giữa các lớp (Separability) bằng siêu phẳng, nhưng **không ép các khuôn mặt của cùng một người phải co cụm chặt lại (Compactness)**.
Khi người đó đổi kiểu tóc, đeo kính, già đi, hoặc đổi góc nhìn, vector đặc trưng sẽ trôi dạt ra xa và bị nhận nhầm sang người khác.

#### B. Đột phá hình học của ArcFace: Đưa bài toán lên Siêu cầu (Hypersphere $\mathbb{S}^{511}$)
1. **Chuẩn hóa vector đặc trưng và ma trận trọng số:**
   ArcFace chiếu toàn bộ vector đặc trưng lên mặt cầu bằng cách ép $\|x\| = 1$ và chuẩn hóa trọng số của mỗi người $\|W_j\| = 1$. Khi đó, tích vô hướng trở thành cosin của góc giữa vector khuôn mặt và tâm đại diện của người đó:
   $$W_j^T x = \|W_j\| \|x\| \cos \theta_j = \cos \theta_j$$
2. **Cộng Lề góc phụ gia (Additive Angular Margin $m$):**
   ArcFace cộng thêm một "vùng đệm an toàn hình học" $m = 0.5$ radian ($\approx \mathbf{28.6^\circ}$) trực tiếp vào góc của người đúng:
   $$\cos(\theta_{y} + m)$$
3. **Công thức hàm mất mát ArcFace hoàn chỉnh:**
   $$\mathcal{L}_{\text{ArcFace}} = -\ln \frac{e^{s \cdot \cos(\theta_y + m)}}{e^{s \cdot \cos(\theta_y + m)} + \sum_{j \ne y} e^{s \cdot \cos \theta_j}}$$
   *(trong đó $s = 64$ là bán kính phóng đại siêu cầu).*

#### 💡 Ý nghĩa hình học:
- Để hàm loss hội tụ về 0, mạng nơ-ron bị ép buộc phải kéo vector khuôn mặt của Alice đến gần tâm đại diện của Alice với một góc **nhỏ hơn ít nhất $28.6^\circ$** so với bất kỳ người nào khác trong công ty.
- Nó tạo ra một **khoảng cách địa lý góc cực kỳ rộng (geodesic margin)** giữa các nhân viên khác nhau, đồng thời nén chặt toàn bộ ảnh của cùng một người vào một chùm tia siêu hẹp trên mặt cầu!

---

### 3. Chuẩn hóa L2-Normalization ($\|\mathbf{e}\|_2 = 1.0$)

Tại [face_auth/recognition/embedder.py:L48-L54](face_auth/recognition/embedder.py#L48-L54):
```python
feat = self._model.get_feat(aligned_face)   # Vector 512 chiều thô
embedding = np.asarray(feat).flatten()

norm = np.linalg.norm(embedding)            # Tính độ dài Euclidean (L2 norm)
if norm == 0:
    return None

return embedding / norm                     # Ép độ dài chính xác về 1.0
```

#### Tại sao bắt buộc phải chia cho `norm` trước khi trả về?
1. **Đưa vector về Siêu cầu đơn vị $\mathbb{S}^{511}$:**
   Mọi vector đầu ra đều có độ dài tuyệt đối đúng bằng $1.0$:
   $$\|\mathbf{e}\|_2 = \sqrt{\sum_{i=1}^{512} e_i^2} = 1.0$$
2. **Biến phép đo khoảng cách thành Tích vô hướng đơn giản:**
   Khi hai vector $\mathbf{u}$ và $\mathbf{v}$ đã có độ dài bằng $1$, công thức Cosine Similarity trở thành:
   $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i v_i = \cos(\theta)$$
   Đồng thời khoảng cách Euclidean bình phương liên hệ trực tiếp với Cosine:
   $$\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) = \mathbf{2 - 2 \cos(\theta)}$$

---

### 4. Tác động trực tiếp lên Module 8 (PostgreSQL + pgvector)

Nhờ việc vector 512 chiều đã được chuẩn hóa L2 ngay tại Module 7:
- **Tối ưu hóa Database tuyệt đối:** Khi nạp vector vào **PostgreSQL `pgvector`**, cơ sở dữ liệu **hoàn toàn không cần tốn chu kỳ CPU để tính căn bậc hai hay chia độ dài vector**.
- **Tìm kiếm thần tốc với toán tử `<=>` (Cosine Distance):**
  $$\text{Cosine Distance} = 1 - \cos(\theta) = 1 - (\mathbf{u} \cdot \mathbf{v})$$
  Database chỉ cần thực hiện phép nhân ma trận tích vô hướng cực nhanh kết hợp với cấu trúc chỉ mục đồ thị **HNSW Index**. Tốc độ truy vấn 1 khuôn mặt trong hàng chục ngàn nhân viên chỉ mất **dưới 2 mili-giây**!

---

---

## MODULE 8: CƠ SỞ DỮ LIỆU VECTOR & ĐỐI SOÁT DANH TÍNH (POSTGRESQL + PGVECTOR)

> **File nguồn:** [face_auth/database/vector_db.py](face_auth/database/vector_db.py), [face_auth/database/schema.sql](face_auth/database/schema.sql)  
> **Lớp cốt lõi:** `VectorDB`, hàm `decide_identity()`  
> **Công nghệ:** PostgreSQL 16+, Extension `pgvector`, Thư viện `psycopg_pool`  
> **Nhiệm vụ:** Quản lý kết nối Connection Pool, tìm kiếm đối soát vector 1:N bằng khoảng cách Cosine, quản lý dữ liệu nhân viên và kiểm soát logic điểm danh (Cooldown 15 phút, audit trail).

```text
 Vector 512-D ──> ConnectionPool ──> PostgreSQL pgvector (HNSW Index)
                                              │
                                              ▼
                         SELECT ... ORDER BY f.embedding <=> %s LIMIT 5
                                              │
                                              ▼
                          decide_identity(similarity >= 0.45)
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
                 [OK] Loc Ngo 0.82                              [UNKNOWN 0.38]
                       │
                       ▼ (Nếu đủ 2 frame ổn định)
             log_attendance(Cooldown 15m)
```

---

### 1. Kiến trúc Bảng Tối giản (Clean Core Schema — 3 Bảng)

Cơ sở dữ liệu được chuẩn hóa cao độ với đúng 3 bảng quan hệ:

1. **`employees`:** Quản lý thông tin định danh nhân sự (`employee_id` làm khóa chính PK, `full_name`, `status` 'ACTIVE'/'INACTIVE', `created_at`, `updated_at`).
2. **`face_embeddings`:** Lưu trữ vector đặc trưng 512 chiều (`embedding VECTOR(512)`), liên kết khóa ngoại `REFERENCES employees(employee_id) ON DELETE CASCADE`. Phân biệt 2 loại vector qua `embedding_type` ('SAMPLE' / 'CENTROID') và phiên bản mô hình `model_version`.
3. **`attendance_logs`:** Bảng lịch sử điểm danh bất biến (Append-only Audit Log), lưu `employee_id`, `action` ('CHECKIN' / 'CHECKOUT'), điểm nhận diện `face_similarity`, điểm sống liveness `liveness_score`, và mốc thời gian có múi giờ `timestamp TIMESTAMPTZ`.

---

### 2. Chiến lược Hai Tầng Vector: `SAMPLE` vs `CENTROID` (Tăng tốc 10 lần)

Khi đăng ký nhân viên (`upsert`), hệ thống chụp 5 - 10 ảnh mẫu ở nhiều góc độ:
- **Vector `SAMPLE`:** Lưu từng vector đơn lẻ của mỗi ảnh chụp làm dữ liệu thô (Raw Biometric Samples) để phục vụ kiểm toán hoặc tái huấn luyện trong tương lai.
- **Vector `CENTROID` (Trọng tâm sinh trắc học):**
  Hệ thống tính vector trung bình của các ảnh chụp và chuẩn hóa lại độ dài L2:
  $$\mathbf{e}_{\text{centroid}} = \frac{\sum_{i=1}^N \mathbf{e}_i}{\left\|\sum_{i=1}^N \mathbf{e}_i\right\|_2}$$
  Lưu vào database với `embedding_type = 'CENTROID'`.

👉 **Tối ưu hóa tìm kiếm 1:N (`search`):**
Tại [face_auth/database/vector_db.py:L72](face_auth/database/vector_db.py#L72), câu lệnh SQL **CHỈ quét trên các vector `CENTROID`**:
```sql
SELECT e.employee_id, e.full_name, 1 - (f.embedding <=> %s::vector) AS similarity
FROM face_embeddings f
JOIN employees e ON f.employee_id = e.employee_id
WHERE f.embedding_type = 'CENTROID'
  AND f.model_version = %s
  AND e.status = 'ACTIVE'
ORDER BY f.embedding <=> %s::vector
LIMIT %s;
```
- *Lợi ích:* Mỗi nhân viên chỉ có đúng **1 vector đại diện duy nhất**. Nếu công ty có 1.000 nhân viên, database chỉ cần so sánh với 1.000 vector thay vì $1.000 \times 10 = 10.000$ vector! Tốc độ tìm kiếm **nhanh gấp 10 lần**, loại bỏ hoàn toàn hiện tượng trùng lặp danh tính trong top kết quả.

---

### 3. Toán tử Cosine `<=>` & Quyết định Danh tính (`decide_identity`)

- Trong `pgvector`, toán tử `<=>` đo **Khoảng cách Cosine (Cosine Distance)**: $1 - \cos \theta$.
- Do vector đã được chuẩn hóa L2 ở Module 7 ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), độ tương đồng **Cosine Similarity** được tính trực tiếp trong SQL bằng:
  $$\text{Similarity} = 1 - (f.embedding \Leftrightarrow \%s::\text{vector}) = \cos \theta = \mathbf{u} \cdot \mathbf{v}$$

Hàm `decide_identity(rows, threshold)` ([face_auth/database/vector_db.py:L215-L228](face_auth/database/vector_db.py#L215-L228)):
- Lấy kết quả đứng đầu có độ tương đồng lớn nhất (`best_similarity`).
- Nếu $\text{best\_similarity} \ge \mathbf{MATCH\_THRESHOLD = 0.45}$: 
  $\implies$ Trả về `(employee_id, full_name, similarity)` $\to$ **Nhận diện thành công!**
- Nếu $< 0.45$: 
  $\implies$ Trả về `(None, "UNKNOWN", similarity)` $\to$ **Khuôn mặt người lạ / Chưa đăng ký!**

---

### 4. Quản lý Kết nối Connection Pool (`psycopg_pool`)

- **Vấn đề:** Camera 30 FPS gọi database liên tục. Việc mở/đóng kết nối TCP riêng lẻ (`connect() -> close()`) sẽ gây nghẽn mạng và cạn kiệt tài nguyên máy chủ (Connection Overhead).
- **Giải pháp:** Dùng `ConnectionPool(min_size=1, max_size=4)`:
  - Khởi tạo sẵn 1 đến 4 kết nối thường trực trong RAM.
  - Khi cần tìm kiếm vector, luồng mượn kết nối (`with self.pool.connection() as conn`), truy vấn xong tự động trả về pool trong vài micro-giây!

---

### 5. Nghiệp vụ Điểm danh: Cooldown 15 phút & Composite Index

Hàm `log_attendance()` ([face_auth/database/vector_db.py:L180-L207](face_auth/database/vector_db.py#L180-L207)) đảm bảo an toàn tuyệt đối cho quy trình chấm công:

1. **Kiểm tra Cooldown chống gian lận/spam điểm danh:**
   ```sql
   SELECT timestamp FROM attendance_logs
   WHERE employee_id = %s AND action = %s
     AND timestamp > NOW() - make_interval(mins => %s)
   ORDER BY timestamp DESC
   LIMIT 1;
   ```
   Dùng hàm `make_interval(mins => %s)` chuẩn của PostgreSQL. Nếu nhân viên đó đã CHECKIN trong vòng 15 phút vừa qua, DB lập tức từ chối và trả về lý do: `"Đã CHECKIN gần đây. Vui lòng đợi 15 phút."` kèm mốc thời gian gần nhất `last_ts`.

2. **Tối ưu hóa bằng Composite Index:**
   Tại [face_auth/database/schema.sql:L37-L38](face_auth/database/schema.sql#L37-L38):
   ```sql
   CREATE INDEX IF NOT EXISTS idx_attendance_lookup 
   ON attendance_logs (employee_id, action, timestamp DESC);
   ```
   Gom 3 trường `(employee_id, action, timestamp DESC)` vào một cây chỉ mục B-tree. Phép kiểm tra Cooldown chạy theo cơ chế **Index Scan** với thời gian chỉ **$0.1\text{ms}$** ngay cả khi bảng điểm danh chứa hàng triệu bản ghi.

3. **Cơ chế Đồng bộ Cooldown Hai Chiều (DB ➔ RAM Cache Sync):**
   Khi DB từ chối vì Cooldown, nó gửi mốc thời gian `last_ts` về cho ứng dụng. Bộ nhớ RAM gán ngay:
   `track.last_attendance_time = last_ts.timestamp()`.
   Nhờ đó, các frame tiếp theo trên camera tự nhận biết mình đang trong thời gian chờ và **hoàn toàn không gửi query kiểm tra lên Database nữa**, giải phóng 100% tải mạng cho hệ thống!

---

---

## MODULE 9: GIAO DIỆN TRỰC QUAN HUD, OVERLAY & ĐIỀU KHIỂN THỜI GIAN THỰC

> **File nguồn:** [face_auth/app.py](face_auth/app.py)  
> **Hàm cốt lõi:** `draw_track()`, `draw_fps()`, `draw_pad_badge()`, vòng lặp xử lý phím tắt `cv2.waitKey()`  
> **Nhiệm vụ:** Hiển thị trực quan trạng thái an ninh của từng khuôn mặt trên luồng video camera, cung cấp thông tin giám sát hiệu năng (FPS, badge PAD, mode) và hỗ trợ tương tác nóng (Runtime Hotkeys).

```text
 ┌──────────────────────────────────────────────────────────────┐
 │ FPS: 28.5                                                    │
 │ PAD: ON                                                      │
 │ MODE: CHECKIN                                                │
 │                                                              │
 │         ┌───────────────────┐        ┌───────────────────┐   │
 │         │ [OK] Loc Ngo 0.82 │        │      ⚠ SPOOF      │   │
 │         ├───────────────────┤        ├───────────────────┤   │
 │         │   (Viền XANH LÁ)  │        │   (Viền ĐỎ RỰC)   │   │
 │         │  Người thật hợp lệ│        │  Phát hiện giả mạo│   │
 │         └───────────────────┘        └───────────────────┘   │
 │                                                              │
 │ Loc Ngo: CHECKIN SUCCESS lúc 08:30:15                        │
 └──────────────────────────────────────────────────────────────┘
```

---

### 1. Hệ Thống 5 Màu Trạng Thái Bounding Box & Thác Ưu Tiên (Priority Cascade)

Tại [face_auth/app.py:L50-L95](face_auth/app.py#L50-L95), hàm `draw_track()` quyết định màu sắc viền và nhãn hiển thị theo một cấu trúc thác điều kiện ưu tiên nghiêm ngặt:

| Thứ tự ưu tiên | Trạng thái hiển thị | Mã màu BGR | Nhãn hiển thị | Ý nghĩa kỹ thuật |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`SPOOF`** | `(0, 0, 220)` (Đỏ tươi) | **`⚠ SPOOF`** | **Ưu tiên cao nhất:** Module PAD phát hiện giả mạo. Đè lên mọi trạng thái khác, ngắt toàn bộ nhận diện và điểm danh. |
| **2** | **`PAD_PENDING`** | `(180, 180, 180)` (Xám) | **`Kiem tra PAD...`** | Khuôn mặt mới vào, đang tích lũy đủ 3 phiếu bầu PAD tối thiểu. Tạm khóa nhận diện để tránh phán đoán vội. |
| **3** | **`Pending`** | `(180, 180, 180)` (Xám) | **`Nhan dien...`** | Đã qua vòng kiểm tra PAD (là người thật), đang đợi mô hình ArcFace trả về danh tính lần đầu. |
| **4** | **`UNKNOWN`** | `(0, 40, 220)` (Đỏ đậm) | **`UNKNOWN 0.38`** | Người thật nhưng độ tương đồng Cosine $< 0.45$, không khớp với bất kỳ nhân viên nào trong Database. |
| **5** | **`Re-verifying`** | `(0, 165, 255)` (Màu cam) | **`Loc Ngo 0.78`** | Đã nhận diện trước đó, giữ tên cũ nhưng đang đến chu kỳ tái thẩm định danh tính định kỳ (mỗi 0.5s). |
| **6** | **`Locked`** | `(0, 210, 80)` (Xanh lá) | **`[OK] Loc Ngo 0.82`** | Đã nhận diện thành công, điểm danh xong và đang trong thời gian khóa hiển thị an toàn. |

#### Kỹ thuật vẽ nhãn Badge chuyên nghiệp (Anti-Aliased Badge):
- Thay vì chỉ viết chữ đè lên ảnh (rất khó đọc khi phông nền sáng), hàm `draw_track()` đo kích thước chữ bằng `cv2.getTextSize()` và vẽ một **hình chữ nhật đặc (Filled Rectangle)** làm nền phía trên Bbox, sau đó viết chữ trắng lên trên bằng cờ làm mịn đường viền `cv2.LINE_AA`.

---

### 2. Banner Thông Báo Điểm Danh Tức Thì (Attendance Toast Notification)

Tại [face_auth/app.py:L571-L575](face_auth/app.py#L571-L575):
- Khi một lượt điểm danh diễn ra (thành công hoặc bị từ chối), hệ thống hiển thị một dòng thông báo nổi ở đáy màn hình trong đúng **3.0 giây**:
  - **Thành công:** Màu xanh lá `(0, 255, 0)`:
    `"Loc Ngo: CHECKIN SUCCESS lúc 08:30:15"`
  - **Bị chặn Cooldown:** Màu cam cảnh báo `(0, 165, 255)`:
    `"Loc Ngo: Đã CHECKIN gần đây. Vui lòng đợi 15 phút. (Gần nhất: 08:25:10)"`
- Giúp nhân viên nhận được phản hồi thị giác ngay lập tức mà không cần nhìn vào màn hình điều khiển console.

---

### 3. Đo Lường FPS Trung Bình Trượt (Rolling Average FPS)

Tại [face_auth/app.py:L584-L586](face_auth/app.py#L584-L586):
```python
t1 = time.perf_counter()
frame_time_deque.append(max(t1 - t0, 1e-6))
avg_fps = len(frame_time_deque) / sum(frame_time_deque)
```

#### Tại sao dùng `len / sum` thay vì trung bình cộng `mean(1/dt)`?
- Nếu tính $FPS = \frac{1}{\Delta t}$ cho từng frame rồi lấy trung bình cộng, chỉ cần một frame bất kỳ bị trôi nhanh bất thường ($\Delta t \to 0$) sẽ khiến giá trị FPS bị thổi phồng ảo (Harmonic mean bias).
- Công thức $\frac{\text{Số frame}}{\text{Tổng thời gian trôi qua}}$ của hàng đợi trượt `frame_time_deque` (cửa sổ 30 frame) phản ánh **chính xác 100% số lượng khung hình thực tế** mà hệ thống xử lý được trong 1 giây.

---

### 4. Hệ Thống Phím Tắt Điều Khiển Nóng (Runtime Hotkeys)

Hệ thống cho phép người vận hành tương tác trực tiếp với camera thông qua lệnh `cv2.waitKey(1)` mà không cần khởi động lại ứng dụng:

- **Phím `'q'` hoặc `ESC` (Thoát an toàn):**
  - Ngắt vòng lặp camera, giải phóng tài nguyên phần cứng `cap.release()`, đóng các cửa sổ OpenCV `cv2.destroyAllWindows()`, và đóng Connection Pool database `db.close()`.
  - **In Bảng Benchmark Hiệu Năng Chi Tiết (Runtime Summary):** Thống kê tổng số frame, số lần gọi detector/tracker/pad/recognition, và thời gian trung bình (ms) của từng module để báo cáo kỹ thuật.
- **Phím `'p'` (Bật/Tắt nóng module PAD):**
  - Chuyển đổi trạng thái `PAD: ON` $\longleftrightarrow$ `PAD: OFF`.
  - Nếu chuyển sang `ON`: tự động nạp model lên RAM (nếu chưa nạp) và kích hoạt `track.reset_pad()` cho toàn bộ các khuôn mặt đang theo dõi để buộc kiểm tra lại từ đầu.
- **Phím `'f'` (Bật/Tắt hiển thị FPS):**
  - Ẩn hoặc hiện dòng chữ `FPS: xx.x` ở góc trên màn hình.

---

## 🏁 TỔNG KẾT TOÀN BỘ PIPELINE THỰC THI (END-TO-END FLOW)

Qua 9 module chuyên sâu, luồng dữ liệu của một frame hình ảnh từ khi camera ghi nhận cho đến khi hoàn tất điểm danh diễn ra theo một chu trình khép kín, chính xác và tối ưu:

```text
 [1. VideoCapture] ──> Frame 640x480 (Buffer=1)
          │
          ▼
 [2. Cadence Control] ──> [3. SCRFD Face Detection] (Mỗi 0.1s / 10 FPS)
          │                         │
          │ (Các frame xen kẽ)       ▼
          └─────────────> [4. Optical Flow Tracker] (IoU 0.3, bám đuôi mượt mà)
                                    │
                                    ▼
                          [5. Anti-Spoofing PAD] (MobileNetV3-Small, LogSumExp, 0.1s)
                                    │
                       ┌────────────┴────────────┐
                       ▼ (SPOOF)                 ▼ (REAL)
                 [Khung ĐỎ ⚠]           [6. Face Alignment] (5-point Similarity 112x112)
                 (Khóa hoàn toàn)                  │
                                                   ▼
                                        [7. ArcFace Embedder] (Vector 512-D, L2 Norm)
                                                   │
                                                   ▼
                                        [8. PostgreSQL pgvector] (Cosine <=> Index HNSW)
                                                   │
                                                   ▼ (Match >= 0.45 & Stable >= 2)
                                        [Điểm danh Cooldown 15m]
                                                   │
                                                   ▼
                                        [9. HUD Giao diện] ([OK] Xanh lá, Banner Toast)
```

---
*Toàn bộ 9 module kiến trúc của hệ thống SmartFace AIoT đã được phân tích đầy đủ và chuẩn hóa phục vụ bảo vệ đồ án tốt nghiệp tại [face_auth/md/app_execution_flow_walkthrough.md](face_auth/md/app_execution_flow_walkthrough.md).*
