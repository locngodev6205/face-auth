# Runtime comparison: 74b2785, b70172a và working tree hiện tại

Đọc các file trực tiếp bằng git show; không checkout/reset hoặc sửa runtime. Config ở commit cũ là defaults và .env.example, không phải bằng chứng .env thực tế lúc người dùng chạy camera.

## Kết quả đối chiếu

| Yếu tố | 74b2785 | b70172a | Hiện tại |
|---|---|---|---|
| PAD default/active | mnv3_e1 v5.1 | mnv3_e1 v5.1 | mnv3s v5.3 theo .env/log |
| Backbone | MobileNetV3-Large | MobileNetV3-Large | MobileNetV3-Small |
| P(REAL) threshold | 0.3356796703127529 | Như trước | 0.38579509526467753 |
| Crop factor | Default 1.55 | 1.55 | Active 1.55 |
| RGB / normalization / gamma | RGB, CelebA mean/std, OFF | Như trước | Như trước |
| Detector | Default SCRFD640 | Như trước | Như trước |
| Camera với SCRFD | Không set resolution, dùng mode backend mặc định | Yêu cầu 640×480 | Yêu cầu 640×480 |
| Detector/PAD interval | 0.1s | 0.1s | 0.1s |
| Smoothing | 5 votes, min 5, spoof ratio 0.6 | Như trước | Như trước |
| Flow mất dấu/detector unmatched | Giữ PAD window, stale guard 3s | Như trước | Invalidate PAD và identity |
| Unsafe geometry | Có thể return tracked bbox khi cần redetect, sửa ở frame sau | Như trước | Không infer geometry không an toàn; redetect cùng frame |

Tracker, predictor, preprocessing và SCRFD wrapper ở hai commit cũ giống nhau byte-for-byte. b70172a thay chủ yếu camera properties/startup diagnostics và phép tính FPS; không đổi công thức score hay smoothing. 74b2785 có thể lấy ảnh camera nhiều pixel hơn nhưng không có log camera cũ để chứng minh actual resolution.

## Offline checks

- Hash .onnx và .onnx.data của v5.1 tại b70172a khớp các file v5.1 đang archive.
- Load predictor/preprocess/tracker cũ từ source trích vào /private/tmp, không thay import/runtime chính.
- Với cùng model v5.1, threshold -0.682607114315033, crop1.55, RGB, gamma OFF: bốn bbox synthetic (giữa/mép ảnh, nhỏ/lớn) cho crop và tensor cũ/mới bằng nhau chính xác; score delta=0, verdict bằng nhau.
- Tạo track đủ 5 REAL votes, mock một lần flow failure rồi detector match cùng bbox: b70172a giữ REAL/5 votes, current chuyển PAD_PENDING/0 votes.

Smoke parity không phải phép đo accuracy trên ảnh thật hoặc parity mọi backend. Flow failure reproduction chứng minh khác state, không xác nhận mọi lần pending trong camera log có cùng nguyên nhân. Log hiện có chưa ghi reset reason.

## Nhận định

1. Không thấy regression trong công thức spatial PAD crop/normalization/logit score khi cố định model và contract trên các input đã thử.
2. Đổi model Large→Small và threshold đi kèm là thay đổi thực, không thể gộp với tác động runtime. Không hạ threshold của Small để giả lập threshold Large.
3. Reset PAD ngay khi LK thất bại có thể là quá bảo thủ trong trường hợp detector xác nhận lại continuity cùng frame. Đây là candidate runtime regression về độ ổn định, dù đường geometry-refresh trước đó đã được giảm reset.
4. Bản cũ giữ votes qua mất dấu có thể ổn định UI hơn, nhưng cũng có thể kế thừa verdict cho đối tượng mới ở cùng vị trí. Không rollback toàn bộ semantics đó.
5. Detector refresh và transform/copy frame hiện tại có overhead; chưa có replay/hardware measurement để định lượng. Cần đánh giá full PAD path trên edge cùng settings.

## Thứ tự kiểm chứng đề xuất

- Cố định frame/video input, detector, crop1.55 và threshold riêng của từng model; so Large v5.1/Small v5.3 trong cùng runtime. Đánh giá cả bona-fide và print/replay, không chỉ tỷ lệ hiện xanh.
- Cố định model/contract; so reset hiện tại với candidate giữ votes cho một flow failure được detector xác nhận ngay cùng frame. Chỉ chấp nhận candidate khi continuity guard rõ ràng; detector không match, gap dài hoặc PAD inference error vẫn invalidate. IoU đơn độc không chứng minh danh tính.
- Để loại tác động crop drift, so PAD trên detector bbox và propagated bbox của cùng frame; không đổi preprocessing/calibration trong phép so.
- Camera mode 640×480 so mode cũ chỉ cần thử riêng nếu historical actual resolution có thể xác minh; không tăng resolution mặc định để che regression trên edge.
- Thu reset reason, votes, bbox source, capture/frame timestamps và timings. Xác định failure mode trước khi sửa smoothing hoặc retrain.

Có thể chạy thử model v5.1 bằng runtime hiện tại mà không sửa .env:

```bash
PAD_RUNTIME_CONFIG_PATH=antispoof/models/mnv3_e1_preliminary_v5_1_runtime_config.json PAD_BBOX_EXPANSION_FACTOR=1.55 PAD_DIAGNOSTIC_LOG=true python3 app.py --mode none 2>&1 | tee output/pad_legacy_model_current_runtime.log
```

Override crop1.55 ở lệnh này tái lập default của hai commit cũ; metadata training v5.1 ghi1.50. Đây là diagnostic historical profile, không thay thế kết quả calibrated contract1.50. Model config v5.1 đã tạo có đường dẫn ONNX/threshold đúng; không cần git reset. Camera phải do người dùng chạy; chưa có camera/edge result mới từ audit này.
