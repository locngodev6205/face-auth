import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    """
    Recognition-only wrapper (Cách B trong tài liệu thiết kế).

    FaceAnalysis.get(image) tự chạy một detector nội bộ (SCRFD) để tìm
    landmark rồi mới align + embed — điều này trùng lặp với FaceDetector
    (detection/detector.py) vốn cũng đã dùng chính submodel SCRFD đó.
    Ở đây ta lấy thẳng submodel recognition (ArcFace) ra khỏi model pack và
    gọi get_feat() trên ảnh ĐÃ ALIGN sẵn 112x112 (xem alignment/aligner.py).
    Cách này chỉ detect một lần duy nhất cho toàn pipeline (ở FaceDetector).

    allowed_modules=["recognition"] để chỉ load submodel recognition, tránh
    tải kèm detection/genderage/landmark_3d_68 không dùng tới ở đây — phần
    detection đã được load riêng trong FaceDetector.
    """

    def __init__(self, model_name: str = "buffalo_s", ctx_id: int = -1):
        # FaceAnalysis bắt buộc phải có 'detection' trong self.models (assert nội bộ),
        # nên ta truyền allowed_modules=["detection", "recognition"] và chỉ lấy submodel recognition.
        app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=ctx_id)

        recognition_model = app.models.get("recognition")
        if recognition_model is None:
            raise RuntimeError(
                f"Không tìm thấy recognition model trong pack '{model_name}'. "
                "Kiểm tra lại tên model hoặc phiên bản insightface đang cài."
            )

        self._model = recognition_model
        # Thường là (112, 112) — dùng giá trị này làm output_size cho aligner.
        self.input_size = self._model.input_size

    def embed_aligned(self, aligned_face):
        """aligned_face: ảnh BGR đã align, kích thước khớp self.input_size."""
        if aligned_face is None:
            return None

        feat = self._model.get_feat(aligned_face)   # vector 512 chiều đại diện cho các nét sinh trắc học
        embedding = np.asarray(feat).flatten()

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        return embedding / norm