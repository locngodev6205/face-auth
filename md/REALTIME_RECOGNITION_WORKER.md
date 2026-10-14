# Nhận diện realtime với cloud DB và Raspberry Pi 4

`python3 app.py` sử dụng một worker nhận diện riêng mặc định. Luồng camera chạy
detection, tracking, PAD, align ảnh mặt nhỏ và hiển thị. Worker chạy ArcFace rồi
truy vấn cloud hoặc đối chiếu centroid trong cache. Chỉ luồng camera cập nhật
track. `api_service.py` vẫn xử lý API như trước, không sử dụng worker/cache này.

## Cấu hình khởi đầu

Các biến sau có mặc định trong `config.py`; có thể thêm vào `.env` để chỉnh:

```dotenv
RECOGNITION_QUEUE_SIZE=2
RECOGNITION_INTRA_OP_THREADS=1
RECOGNITION_CACHE_TTL_SECONDS=30
RECOGNITION_CACHE_THRESHOLD=0.65
RECOGNITION_DB_POOL_TIMEOUT_SECONDS=3
RECOGNITION_DB_QUERY_TIMEOUT_MS=3000
```

Một worker, tối đa hai tác vụ trong hàng đợi và ba tác vụ chưa được thu kết quả.
Mỗi phiên xác minh của một track chỉ có một tác vụ đang xử lý/chờ kết quả. Queue
chỉ giữ bản sao ảnh đã align (thường 112×112), không giữ toàn bộ frame. Hàng đợi
đầy thì bỏ lần gửi hiện tại, thử lại ở frame sau. Không tạo thread theo mỗi frame.

ArcFace dùng ONNX Runtime sequential, một intra-op thread, tắt spinning. Đây là
giới hạn riêng cho embedder realtime, không đổi model/preprocessing. Detection
và PAD vẫn chạy trên luồng camera với cấu hình hiện có. Một worker không đảm bảo
FPS cụ thể; cần đo trên Pi 4 trước khi tăng số thread ArcFace lên 2.

## Cache centroid

Cloud search giữ nguyên bộ lọc CENTROID, model_version, embedding active và nhân
viên ACTIVE; lấy top-1 vì `decide_identity` hiện chỉ dùng ứng viên đứng đầu.
Worker nhận thêm centroid đã lưu trong DB, chuẩn hóa vector và lưu theo
`(track_id, verification_generation)`. Không cache embedding của ảnh camera.

Chỉ cache khi điểm cloud đạt ít nhất `max(MATCH_THRESHOLD,
RECOGNITION_CACHE_THRESHOLD)`. Trong cùng track còn hợp lệ, mỗi lần nhận diện
vẫn chạy ArcFace trên ảnh mới. Nếu cosine similarity với centroid cache đủ cao,
trả kết quả CACHE; nếu thấp thì truy vấn cloud. Cache hết hạn sau 30 giây tính
từ lúc nhận centroid cloud, không gia hạn bởi cache hit. Đặt TTL bằng 0 để tắt.

TTL này giới hạn thời gian phản ánh thay đổi tên, trạng thái ACTIVE hoặc
embeddings trên cloud: thay đổi có thể chưa được thấy trong cửa sổ cache. Cache
không thay thế việc tìm best match toàn gallery; shortcut chỉ xác minh lại
centroid của track đã được cloud xác nhận, với ngưỡng cao hơn. Giá trị 0.65 là
giá trị khởi đầu, chưa được calibrate cho dữ liệu của bạn. Có thể tắt cache để
đối chiếu kết quả toàn gallery hoặc giảm TTL khi cần cập nhật quyền nhanh hơn.

Track mới/reset không dùng cache của track cũ. Cache chỉ ở RAM, tối đa 128 mục,
không ghi disk. Worker xóa cache của phiên nếu truy vấn mới lỗi hoặc không khớp.
Nếu cloud mất kết nối, cache hợp lệ có thể tiếp tục dùng đến hết TTL; sau đó
hiển thị ERROR, không dùng cache quá hạn làm fallback.

## Kết quả muộn và thông báo

- `WAIT 2.1s`: tác vụ chưa có kết quả; camera tiếp tục chạy. Khi xác minh lại,
  box giữ màu xanh nếu đã có danh tính và PAD hợp lệ; dòng riêng `RECHECK` cho
  biết đang xác minh lại, không phải lần xác minh mới. Lần đầu chờ nhận diện hiện cam.
- `QUEUE FULL`: chưa gửi được tác vụ vì hàng đợi đầy.
- `CLOUD 1.2s` / `CACHE 0.1s`: nguồn kết quả và tổng thời gian từ khi gửi tác vụ,
  bao gồm thời gian chờ queue.
- `ERROR PoolTimeout` / loại lỗi khác: không xác minh được; retry theo
  `RECOGNIZE_INTERVAL_SECONDS`. Không coi lỗi mạng là UNKNOWN.

Kết quả muộn được nhận nếu track vẫn tồn tại, còn liên tục trong cùng phiên
xác minh và PAD hiện tại hợp lệ. Không đặt giới hạn tuổi kết quả để đáp ứng việc
chấp nhận kết quả muộn. Track mất, reset, PAD không hợp lệ/spoof hoặc toggle PAD
sẽ làm kết quả cũ không còn hiệu lực. Tác vụ đang chạy không bị kill; kết quả
được bỏ khi trả về. Kết quả phản ánh ảnh lúc gửi, không phải ảnh hiện tại; lần
nhận diện tiếp theo tiếp tục xác minh ảnh mới theo cadence hiện có.

Khi thoát, camera đợi worker tối đa 2 giây. Worker daemon tự đóng pool sau khi
IO đang chạy kết thúc. Log `stopped=False` nghĩa là IO chưa kết thúc trong thời
gian đó. Pool timeout, connect timeout, SQL timeout và TCP keepalive được đặt
để giảm thời gian kẹt; chúng không tạo một deadline tuyệt đối cho DNS/toàn bộ
network stack. Kiểm tra schema và tải model lúc startup vẫn chạy trước camera.

## Kiểm tra offline và trên Pi

Tên nằm trên header cố định `[ID] Tên`; score và thông tin worker nằm ở dòng
riêng phía dưới box. Chỉ nhãn hiển thị dùng hysteresis: danh tính khác cần 2
kết quả liên tiếp (`RECOGNITION_LABEL_CONFIRMATIONS`), UNKNOWN cần 3 kết quả
(`RECOGNITION_LABEL_UNKNOWN_CONFIRMATIONS`). Tên ban đầu được hiển thị ngay.
Khi matching mới bất đồng, tên trước có thể được giữ tối đa 1.5 giây
(`RECOGNITION_LABEL_HOLD_SECONDS`), kèm `LAST LABEL`. Đây chỉ là tên hiển thị;
`track.name`, `track.employee_id`, score và kết quả worker vẫn cập nhật thực tế.
Không dùng tên hiển thị làm quyết định xác thực. Lỗi/PAD/reset xóa nhãn ngay.
Nếu thời gian giữ hết mà tên khác chưa đủ xác nhận, header trở về UNKNOWN.

```bash
cd /Users/coding/PBL6/face_auth
python3 -m unittest tests.test_recognition_worker tests.test_runtime tests.test_tracker tests.test_vector_db_logic
python3 app.py
```

Trên Pi dùng đường dẫn checkout tương ứng. Thử mặt đã đăng ký, theo dõi nguồn
CLOUD rồi CACHE. Giữ track liên tục hơn 30 giây: phải có truy vấn CLOUD mới.
Ngắt mạng sau khi app đã chạy: detection/PAD/hiển thị phải tiếp tục; track mới
chờ rồi báo ERROR, cache chỉ dùng đến TTL. Rời frame/quay lại và thử SPOOF khi
đang chờ cloud: danh tính từ phiên cũ không được gán lại. Bật/tắt PAD cũng phải
bỏ kết quả đang chờ. Bấm `q` để xem số lần cloud/cache/error và trạng thái đóng
worker. `recognition_submit_mean_ms` chỉ là overhead phía camera, không phải
thời gian ArcFace/DB; độ trễ đầy đủ xuất hiện trên overlay và log kết quả.

Các test offline không chứng minh tốc độ CPU Pi, độ trễ Neon, chất lượng cache
threshold hoặc inference/camera thực tế.
