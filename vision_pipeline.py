"""Realtime PAD validity, crop and atomic temporal-vote updates."""
import math
import time
import config

def limit_faces(detections):
    """Select largest faces; stable sorting preserves detector order for ties."""
    if len(detections) <= config.MAX_FACES_PER_FRAME:
        return list(detections)
    return sorted(detections, key=lambda d: max(0, d['bbox'][2] - d['bbox'][0])
                  * max(0, d['bbox'][3] - d['bbox'][1]), reverse=True)[:config.MAX_FACES_PER_FRAME]

def _pad_face_skip_reason(detection, frame_shape):
    """Reject low-confidence or incomplete faces near the camera-frame edge."""
    try:
        frame_h, frame_w = frame_shape[:2]
        bbox = detection["bbox"]
        if len(bbox) != 4:
            return "invalid_bbox"
        x1, y1, x2, y2 = (float(value) for value in bbox)
        score = float(detection.get("score"))
    except (KeyError, TypeError, ValueError):
        return "missing_geometry_or_score"

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
        return "non_finite_geometry_or_score"
    if x2 <= x1 or y2 <= y1:
        return "invalid_bbox"
    if score < config.PAD_MIN_DETECTOR_CONFIDENCE:
        return "low_detector_confidence"

    landmarks = detection.get("landmarks")
    try:
        if landmarks is None or len(landmarks) != 5:
            return "incomplete_landmarks"
        points = [(float(point[0]), float(point[1])) for point in landmarks]
    except (TypeError, ValueError, IndexError):
        return "invalid_landmarks"
    if len(points) != 5:
        return "incomplete_landmarks"
    box_w, box_h = x2 - x1, y2 - y1
    landmark_margin = max(1.0, min(box_w, box_h) * 0.02)
    if any(
        not math.isfinite(x) or not math.isfinite(y)
        or x <= x1 + landmark_margin or x >= x2 - landmark_margin
        or y <= y1 + landmark_margin or y >= y2 - landmark_margin
        for x, y in points
    ):
        return "landmark_at_face_edge"

    edge_margin = max(2.0, min(box_w, box_h) * 0.02)
    if (
        x1 <= edge_margin or y1 <= edge_margin
        or x2 >= frame_w - edge_margin
        or y2 >= frame_h - edge_margin
    ):
        return "face_too_close_to_frame_edge"
    return None


def update_track_pad(
    tracked, frame, predictor, interval_seconds,
    frame_id=0, detector_called=False, diagnostic_log=False,
    crop_smoothing=False,
) -> int:
    """Update due tracks atomically; inference errors invalidate cached verdicts."""
    from antispoof.preprocess import crop_for_runtime as pad_crop

    crops, due = [], []
    crop_mode = getattr(predictor, "crop_mode", config.PAD_CROP_MODE)
    if not isinstance(crop_mode, str):
        crop_mode = config.PAD_CROP_MODE
    model_img_size = getattr(predictor, "model_img_size", config.PAD_MODEL_IMG_SIZE)
    if not isinstance(model_img_size, int):
        model_img_size = config.PAD_MODEL_IMG_SIZE

    def fail(track):
        track.invalidate_verification()
        # Pace failed attempts too, while keeping the empty vote window pending.
        track.last_pad_time = time.monotonic()

    for detection, track in tracked:
        skip_reason = _pad_face_skip_reason(detection, frame.shape)
        if skip_reason is not None:
            track.invalidate_verification()
            track.pad_invalid_reason = skip_reason
            if diagnostic_log:
                print(
                    f"[PAD skip] frame_id={frame_id} track_id={track.track_id} "
                    f"reason={skip_reason} score={detection.get('score')} "
                    f"bbox={detection.get('bbox')}"
                )
            continue
        track.pad_invalid_reason = None
        if not track.needs_pad(interval_seconds):
            continue
        try:
            crop_bbox = detection["bbox"]
            if crop_smoothing:
                crop_bbox = track.pad_crop_bbox(
                    crop_bbox, frame.shape, predictor.bbox_expansion_factor,
                )
            crops.append(
                pad_crop(
                    frame,
                    crop_bbox,
                    predictor.bbox_expansion_factor,
                    model_img_size,
                    crop_mode,
                )
            )
            due.append((detection, track, crop_bbox))
        except Exception as exc:
            fail(track)
            print(f"[PAD crop error] track_id={track.track_id}: {exc}")
    if not crops:
        return 0
    try:
        results = predictor.predict_crops(crops)
        if len(results) != len(due):
            raise ValueError("PAD batch result count does not match input count")
        # Validate the entire batch before accepting any vote.
        for res in results:
            if not isinstance(res.get("is_real"), bool):
                raise ValueError("PAD result has no boolean is_real verdict")
            score = float(res.get("pad_score", res.get("logit_diff", float("nan"))))
            if not math.isfinite(score):
                raise ValueError("PAD result has a non-finite score")
    except Exception as exc:
        for _, track, _ in due:
            fail(track)
        print(f"[PAD error] {exc}")
        return 0

    for (detection, track, crop_bbox), res in zip(due, results):
        score = float(res.get("pad_score", res.get("logit_diff")))
        track.update_pad(res["is_real"], score)
        if diagnostic_log:
            exp_score = math.exp(-abs(score))
            p_real = 1.0 / (1.0 + exp_score) if score >= 0 else exp_score / (1.0 + exp_score)
            print(
                f"[PAD] frame_id={frame_id} timestamp={time.time():.6f} "
                f"track_id={track.track_id} detector_called={detector_called} "
                f"bbox_source={detection.get('bbox_source', 'UNKNOWN')} "
                f"bbox={track.bbox} face_width={track.bbox[2]-track.bbox[0]} "
                f"face_height={track.bbox[3]-track.bbox[1]} "
                f"crop_smoothing={crop_smoothing} pad_crop_bbox={tuple(round(float(v), 3) for v in crop_bbox)} "
                f"logits=({res.get('real_logit')},{res.get('spoof_logit')}) "
                f"d={score:.6f} p_real={p_real:.6f} "
                f"threshold_logit={res.get('threshold_logit', predictor.logit_threshold):.6f} "
                f"raw={'REAL' if res['is_real'] else 'SPOOF'} smoothed={track.pad_status}"
            )
    return len(results)

