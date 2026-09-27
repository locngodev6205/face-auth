# PAD crop smoothing experiment

Flag `PAD_CROP_SMOOTHING=true` bật trong .env cho lần thử; code default và .env.example là false. Terminal override .env.

Chỉ làm mượt center và side của square crop cơ sở PAD ở các lượt PAD đến hạn, trước expansion factor hiện tại. Không sửa track bbox, landmarks, lịch detector/PAD, model contract hoặc voting. Chi phí vài phép toán cho mỗi track đến hạn; không thêm model/inference.

Các tham số thử nghiệm chưa được chọn từ accuracy benchmark: time constant 0.12s, jitter limit 5% cạnh bbox cơ sở, gap limit 0.5s. Alpha phụ thuộc thời gian monotonic để tránh phụ thuộc trực tiếp FPS. Nếu dịch tâm/thay size vượt limit, dt ngoài (0,0.5], crop mở rộng chạm ra ngoài frame, hoặc không có history, dùng geometry mới ngay. Border guard nhằm không làm thay đổi padding do giữ geometry cũ. Mất continuity/PAD error/reset xóa crop history. Các chuyển động nhỏ vẫn có độ trễ smoothing; chưa chứng minh cải thiện PAD.

Chạy từ face_auth, PostgreSQL đang chạy:

```bash
mkdir -p output
PAD_CROP_SMOOTHING=true PAD_DIAGNOSTIC_LOG=true python3 -u app.py --mode none 2>&1 | tee output/pad_crop_smoothing_diagnostic.log
```

Startup có `PAD_CROP_SMOOTHING = True`. `[PAD]` ghi thêm `crop_smoothing` và `pad_crop_bbox` (bbox cơ sở trước expansion), cùng raw tracking bbox. Geometry PAD có thể khác tracking bbox khi smoothing bật.

Baseline/revert một lần:

```bash
PAD_CROP_SMOOTHING=false PAD_DIAGNOSTIC_LOG=true python3 -u app.py --mode none 2>&1 | tee output/pad_crop_baseline_diagnostic.log
```

Tắt lâu dài: sửa .env thành `PAD_CROP_SMOOTHING=false` rồi khởi động lại. Không cần git reset.

Thử cùng camera mode/model/threshold/ratio và ánh sáng: mặt thật gần/trung bình/xa, đứng yên và di chuyển; print/replay tương ứng và chuyển real sang attack. Đánh giá raw scores/verdicts, đổi trạng thái, PENDING/time-to-verdict, attack-switch delay, PAD calls/FPS. Camera runs riêng có khác frames; paired replay là bước xác nhận tốt hơn. Dừng giữ feature nếu attack phản ứng chậm hoặc false acceptance tăng.

Offline: focused runtime/tracker regressions kiểm tra jitter giảm, motion/scale/border/gap snap, reset history, actual PAD crop path và tắt flag về crop gốc. Camera và edge chưa được chạy; thông số này là treatment thử nghiệm, không phải production setting đã xác nhận.
