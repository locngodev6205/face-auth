from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Dùng detector SCRFD có sẵn trong insightface model pack thay vì
    YOLOv8-face rời rạc.

    Lý do đổi: hầu hết checkpoint YOLOv8-face có landmark (derronqi/yolov8-face,
    yakhyo/yolov8-face-onnx-inference,...) dùng kiến trúc/head tùy biến, không
    tương thích trực tiếp với ultralytics.YOLO() — phải tự viết decode logic
    riêng. SCRFD thì ngược lại: đã là dependency sẵn có (cùng pack với
    recognition model trong embedder.py), cho landmark đúng convention
    ArcFace cần, không phải quản lý checkpoint/license của bên thứ ba khác.

    allowed_modules=["detection"] để chỉ load submodel detection (SCRFD),
    tránh tải kèm genderage/landmark_3d_68 không dùng tới trong pack.

    Output format detect() giữ nguyên như bản YOLOv8-face cũ (list dict có
    "bbox", "score", "landmarks") — phần còn lại của pipeline (aligner.py,
    app.py, run_enroll.py) không cần đổi gì thêm.
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        ctx_id: int = -1,
        det_size=(640, 640),
        conf_thresh: float = 0.5,
    ):
        app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=conf_thresh)

        self._model = app.models.get("detection")
        if self._model is None:
            raise RuntimeError(
                f"Không tìm thấy detection model (SCRFD) trong pack '{model_name}'."
            )

    def detect(self, frame):
        bboxes, kpss = self._model.detect(frame, max_num=0, metric="default")

        detections = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i]

            landmarks = None
            if kpss is not None:
                # 5 điểm: mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải
                landmarks = kpss[i].tolist()

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "score": float(score),
                "landmarks": landmarks,
            })

        return detections