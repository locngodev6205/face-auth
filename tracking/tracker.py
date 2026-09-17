import time
from collections import deque
from typing import List, Dict, Tuple, Optional


def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Tính IOU giữa 2 bounding box (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = float(bbox1_area + bbox2_area - intersection_area)
    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


class Track:
    """Đối tượng theo dõi một khuôn mặt qua các frame."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        pad_smooth_window: int = 5,
        pad_spoof_min_ratio: float = 0.6,
        pad_min_votes: Optional[int] = None,
        pad_stale_timeout: Optional[float] = None,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.name = "UNKNOWN"
        self.score = 0.0
        self.last_recognition_time: Optional[float] = None  # None = ép nhận diện ngay khi đủ điều kiện
        self.frames_since_recognition = 0
        self.missing_frames = 0
        self.recognized_once = False  # True sau lần update_result() đầu tiên

        # PAD temporal smoothing: lưu N verdict gần nhất (True=real, False=spoof)
        self._pad_window: deque = deque(maxlen=pad_smooth_window)
        self._pad_spoof_min_ratio = pad_spoof_min_ratio
        self.pad_min_votes = pad_min_votes if pad_min_votes is not None else pad_smooth_window
        self._pad_stale_timeout: Optional[float] = pad_stale_timeout
        self.last_pad_time: Optional[float] = None  # None = chưa chạy PAD lần nào
        self.last_pad_score: Optional[float] = None

        # Attendance tracking
        self.employee_id: Optional[str] = None
        self.stable_recognitions = 0
        self.last_attendance_time: Optional[float] = None
        self._unknown_streak = 0

    def update_pad(self, is_real: bool, pad_score: Optional[float] = None) -> None:
        """Ghi nhận verdict PAD của frame hiện tại vào rolling window và cập nhật timestamp.

        Nếu verdict cuối cùng đã quá cũ (>_pad_stale_timeout giây), reset window trước khi thêm
        vote mới để tránh để dữ liệu cũ làm sai lệch kết quả (ví dụ: đối tượng vừa quay lại sau khi mất khỏi frame).
        """
        if (
            self._pad_stale_timeout is not None
            and self.last_pad_time is not None
            and (time.time() - self.last_pad_time) > self._pad_stale_timeout
        ):
            self._pad_window.clear()
        self._pad_window.append(bool(is_real))
        self.last_pad_time = time.time()
        if pad_score is not None:
            self.last_pad_score = float(pad_score)

    def reset_pad(self) -> None:
        """Reset rolling window và đưa track về trạng thái PAD_PENDING."""
        self._pad_window.clear()
        self.last_pad_time = None
        self.last_pad_score = None

    def needs_pad(self, interval_seconds: float = 0.2) -> bool:
        """Kiểm tra xem track có cần chạy PAD inference lại hay không (tính theo giây)."""
        if self.last_pad_time is None:
            return True
        return (time.time() - self.last_pad_time) >= interval_seconds

    @property
    def pad_ready(self) -> bool:
        """True nếu track đã tích lũy đủ số vote tối thiểu pad_min_votes và verdict chưa cũ (stale)."""
        if len(self._pad_window) < self.pad_min_votes:
            return False
        # Nếu đã vượt quá thời gian stale timeout mà chưa có update mới, coi như PAD_PENDING
        if (
            self._pad_stale_timeout is not None
            and self.last_pad_time is not None
            and (time.time() - self.last_pad_time) > self._pad_stale_timeout
        ):
            return False
        return True

    @property
    def is_spoof(self) -> bool:
        """
        Trả về True nếu track đã tích lũy đủ vote VÀ tỉ lệ frame SPOOF trong rolling window >= pad_spoof_min_ratio.
        Khi chưa đủ vote (PAD_PENDING), trả về False.
        """
        if not self.pad_ready or not self._pad_window:
            return False
        spoof_ratio = self._pad_window.count(False) / len(self._pad_window)
        return spoof_ratio >= self._pad_spoof_min_ratio

    @property
    def is_real(self) -> bool:
        """True khi track đã tích lũy đủ vote VÀ phán quyết không phải là SPOOF."""
        return self.pad_ready and not self.is_spoof

    @property
    def pad_status(self) -> str:
        """
        Trạng thái PAD của track:
            - 'PAD_PENDING': chưa đủ số vote tối thiểu.
            - 'SPOOF': đủ vote và tỉ lệ spoof >= ngưỡng.
            - 'REAL': đủ vote và là người thật.
        """
        if not self.pad_ready:
            return "PAD_PENDING"
        return "SPOOF" if self.is_spoof else "REAL"

    def needs_recognition(self, interval_seconds: float = 1.0, pad_enabled: bool = True) -> bool:
        """
        Kiểm tra xem track có cần re-verify/nhận diện lại hay không (tính theo giây).
        Nếu pad_enabled=True: CHẶN nhận diện hoàn toàn nếu track chưa pad_ready hoặc là SPOOF.
        """
        if pad_enabled and (not self.pad_ready or not self.is_real):
            return False
        if self.last_recognition_time is None:
            return True
        return (time.time() - self.last_recognition_time) >= interval_seconds

    @property
    def is_pending(self) -> bool:
        """True khi track chưa được nhận diện lần nào (chưa có kết quả nhận diện đầu tiên)."""
        return not self.recognized_once

    def can_log_attendance(self, gap_minutes: float) -> bool:
        """Kiểm tra xem đã hết cooldown (gap_minutes) để điểm danh lại chưa."""
        if self.last_attendance_time is None:
            return True
        return (time.time() - self.last_attendance_time) >= (gap_minutes * 60)

    def update_result(self, employee_id: Optional[str], name: str, score: float):
        """Cập nhật kết quả nhận diện và ghi nhận mốc thời gian hoàn tất."""
        if employee_id is not None:
            self._unknown_streak = 0
            if self.employee_id == employee_id:
                self.stable_recognitions += 1
            else:
                self.stable_recognitions = 1
                # Reset cooldown điểm danh nếu đổi sang một employee_id khác
                if self.employee_id is not None and self.employee_id != employee_id:
                    self.last_attendance_time = None
            self.employee_id = employee_id
        else:
            self.stable_recognitions = 0
            self._unknown_streak += 1
            # Chỉ xoá employee_id và cooldown nếu liên tục UNKNOWN > 3 chu kỳ (~1.5s)
            # Giúp tránh nháy frame UNKNOWN làm mất timer cooldown của người vừa điểm danh
            if self._unknown_streak > 3:
                self.employee_id = None
                self.last_attendance_time = None

        self.name = name
        self.score = score
        self.last_recognition_time = time.time()
        self.recognized_once = True


class FaceTracker:
    """
    Theo dõi khuôn mặt bằng IOU matching đơn giản giữa các frame.
    Giúp tối ưu FPS: chỉ chạy lại nhận diện sau mỗi recognize_interval_seconds giây.
    """

    def __init__(
        self,
        recognize_interval_seconds: float = 1.0,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 10,
        pad_smooth_window: int = 5,
        pad_spoof_min_ratio: float = 0.6,
        pad_interval_seconds: float = 0.2,
        pad_min_votes: Optional[int] = None,
        pad_stale_timeout: Optional[float] = None,
    ):
        self.recognize_interval_seconds = recognize_interval_seconds
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.pad_smooth_window = pad_smooth_window
        self.pad_spoof_min_ratio = pad_spoof_min_ratio
        self.pad_interval_seconds = pad_interval_seconds
        self.pad_min_votes = pad_min_votes if pad_min_votes is not None else pad_smooth_window
        self.pad_stale_timeout = pad_stale_timeout
        self.tracks: List[Track] = []
        self._next_track_id = 1

    def update(self, detections: List[Dict]) -> List[Tuple[Dict, Track]]:
        """
        Khớp các detection mới với tracks hiện có.
        Trả về list các tuple (detection, track).
        """
        matched_results: List[Tuple[Dict, Track]] = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Tăng frame counter cho tất cả track hiện có
        for track in self.tracks:
            track.frames_since_recognition += 1
            track.missing_frames += 1

        if detections and self.tracks:
            # Tính ma trận IOU giữa detections và tracks
            iou_matrix = []
            for det in detections:
                det_bbox = det["bbox"]
                row = [compute_iou(det_bbox, track.bbox) for track in self.tracks]
                iou_matrix.append(row)

            # Greedy matching theo IOU cao nhất
            used_dets = set()
            used_tracks = set()

            matches = []
            for d_idx, row in enumerate(iou_matrix):
                for t_idx, iou_val in enumerate(row):
                    if iou_val >= self.iou_threshold:
                        matches.append((iou_val, d_idx, t_idx))

            matches.sort(key=lambda x: x[0], reverse=True)

            for _, d_idx, t_idx in matches:
                if d_idx in used_dets or t_idx in used_tracks:
                    continue
                used_dets.add(d_idx)
                used_tracks.add(t_idx)

                det = detections[d_idx]
                track = self.tracks[t_idx]
                track.bbox = det["bbox"]
                track.missing_frames = 0
                matched_results.append((det, track))

            unmatched_detections = [i for i in range(len(detections)) if i not in used_dets]
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]

        # Reset chuỗi ổn định nếu track bị mất dấu ở frame này (mất focus / quay mặt)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].stable_recognitions = 0

        # Tạo track mới cho các detection chưa match
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(
                track_id=self._next_track_id,
                bbox=det["bbox"],
                pad_smooth_window=self.pad_smooth_window,
                pad_spoof_min_ratio=self.pad_spoof_min_ratio,
                pad_min_votes=self.pad_min_votes,
                pad_stale_timeout=self.pad_stale_timeout,
            )
            self._next_track_id += 1
            new_track.missing_frames = 0
            self.tracks.append(new_track)
            matched_results.append((det, new_track))

        # Xóa các track đã mất dấu quá max_missing_frames
        self.tracks = [t for t in self.tracks if t.missing_frames <= self.max_missing_frames]

        return matched_results
