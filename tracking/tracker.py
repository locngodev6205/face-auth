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
        pad_smooth_window: int = 8,
        pad_spoof_min_ratio: float = 0.6,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.name = "UNKNOWN"
        self.score = 0.0
        self.last_recognition_time: Optional[float] = None  # None = ép nhận diện ngay frame đầu
        self.frames_since_recognition = 0
        self.missing_frames = 0
        self.recognized_once = False  # True sau lần update_result() đầu tiên

        # PAD temporal smoothing: lưu N verdict gần nhất (True=real, False=spoof)
        self._pad_window: deque = deque(maxlen=pad_smooth_window)
        self._pad_spoof_min_ratio = pad_spoof_min_ratio
        self.last_pad_time: Optional[float] = None  # None = chưa chạy PAD lần nào

    def update_pad(self, is_real: bool) -> None:
        """Ghi nhận verdict PAD của frame hiện tại vào rolling window và cập nhật timestamp."""
        self._pad_window.append(is_real)
        self.last_pad_time = time.time()

    def needs_pad(self, interval_seconds: float = 1.0) -> bool:
        """Kiểm tra xem track có cần chạy PAD inference lại hay không (giống needs_recognition)."""
        if self.last_pad_time is None:
            return True
        return (time.time() - self.last_pad_time) >= interval_seconds

    @property
    def is_spoof(self) -> bool:
        """
        Trả về True nếu tỉ lệ frame SPOOF trong rolling window ≥ pad_spoof_min_ratio.
        Khi window chưa đủ frame (track mới tạo), mặc định coi là REAL để tránh
        hiện đỏ ngay lập tức trước khi có đủ dữ liệu.
        """
        if not self._pad_window:
            return False
        spoof_ratio = self._pad_window.count(False) / len(self._pad_window)
        return spoof_ratio >= self._pad_spoof_min_ratio

    def needs_recognition(self, interval_seconds: float = 1.0) -> bool:
        """Kiểm tra xem track có cần re-verify/nhận diện lại hay không (tính theo giây)."""
        if self.last_recognition_time is None:
            return True
        return (time.time() - self.last_recognition_time) >= interval_seconds

    @property
    def is_pending(self) -> bool:
        """True khi track chưa được nhận diện lần nào (frame đầu tiên sau tạo mới)."""
        return not self.recognized_once

    def update_result(self, name: str, score: float):
        """Cập nhật kết quả nhận diện và ghi nhận mốc thời gian hoàn tất."""
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
        pad_smooth_window: int = 8,
        pad_spoof_min_ratio: float = 0.6,
        pad_interval_seconds: float = 1.0,
    ):
        self.recognize_interval_seconds = recognize_interval_seconds
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.pad_smooth_window = pad_smooth_window
        self.pad_spoof_min_ratio = pad_spoof_min_ratio
        self.pad_interval_seconds = pad_interval_seconds
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

        # Tạo track mới cho các detection chưa match
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(
                track_id=self._next_track_id,
                bbox=det["bbox"],
                pad_smooth_window=self.pad_smooth_window,
                pad_spoof_min_ratio=self.pad_spoof_min_ratio,
            )
            self._next_track_id += 1
            new_track.missing_frames = 0
            self.tracks.append(new_track)
            matched_results.append((det, new_track))

        # Xóa các track đã mất dấu quá max_missing_frames
        self.tracks = [t for t in self.tracks if t.missing_frames <= self.max_missing_frames]

        return matched_results
