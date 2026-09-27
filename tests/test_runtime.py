"""Offline runtime regressions: failures, raw frames and same-frame redetection."""

import argparse
import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import app
from alignment.aligner import get_input_face
from tracking.tracker import Track


class TestRuntimeFailures(unittest.TestCase):
    def test_crop_smoothing_changes_only_pad_crop_and_can_be_disabled(self):
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                track = Track(1, (100, 100, 200, 200))
                with patch("tracking.tracker.time.monotonic", return_value=100):
                    track.pad_crop_bbox(track.bbox, (400, 400), 1.55)
                    track.update_pad(True, 2.0)
                raw_bbox = (102, 100, 202, 200)
                predictor = MagicMock(bbox_expansion_factor=1.55)
                predictor.predict_crops.return_value = [{"is_real": True, "pad_score": 2.0}]
                with patch("tracking.tracker.time.monotonic", return_value=100.2), patch(
                    "antispoof.preprocess.crop", return_value=np.zeros((155, 155, 3), np.uint8)
                ) as crop:
                    count = app.update_track_pad(
                        [({"bbox": raw_bbox}, track)], np.zeros((400, 400, 3), np.uint8),
                        predictor, .1, crop_smoothing=enabled,
                    )
                self.assertEqual(count, 1)
                crop_bbox = crop.call_args.args[1]
                if enabled:
                    self.assertGreater(crop_bbox[0], 100)
                    self.assertLess(crop_bbox[0], 102)
                else:
                    self.assertEqual(crop_bbox, raw_bbox)
                self.assertEqual(track.bbox, (100, 100, 200, 200))
                self.assertEqual(len(track._pad_window), 2)

    def test_cli_model_override_cannot_reuse_another_models_contract(self):
        with self.assertRaisesRegex(ValueError, "PAD_RUNTIME_CONFIG_PATH"):
            app.resolve_pad_model_path("some_other_model.onnx")
        self.assertEqual(str(app.resolve_pad_model_path(app.config.PAD_MODEL_PATH)),
                         app.config.PAD_MODEL_PATH)

    def verified_track(self, track_id=1):
        track = Track(track_id, (30, 30, 130, 130))
        for _ in range(5):
            track.update_pad(True, 2.0)
        for _ in range(3):
            track.update_result("A", "Alice", 0.9)
        track.last_recognition_time = 0.0
        track.last_pad_time = 0.0
        return track

    def test_failed_recognition_clears_identity_and_paces_retries(self):
        for mode in ["exception", "empty_alignment", "empty_embedding"]:
            with self.subTest(mode=mode):
                track = self.verified_track()
                embedder, db = MagicMock(), MagicMock()
                embedder.input_size = (112, 112)
                embedder.embed_aligned.return_value = None
                if mode == "exception":
                    embedder.embed_aligned.side_effect = RuntimeError("injected failure")
                aligned = None if mode == "empty_alignment" else np.zeros((112, 112, 3), np.uint8)
                with patch("app.get_input_face", return_value=aligned), patch(
                    "tracking.tracker.time.monotonic", return_value=100
                ), contextlib.redirect_stdout(io.StringIO()):
                    result = app.orchestrate_track_step(
                        track, True, "CHECKIN", db, embedder,
                        np.zeros((180, 180, 3), np.uint8), {"bbox": track.bbox},
                    )
                self.assertFalse(result["recognized"])
                self.assertFalse(result["attendance_attempted"])
                self.assertIsNone(track.employee_id)
                self.assertEqual(track.stable_recognitions, 0)
                self.assertTrue(track.is_pending)
                db.log_attendance.assert_not_called()
                with patch("tracking.tracker.time.monotonic", return_value=100.2):
                    self.assertFalse(track.needs_recognition(0.5))
                with patch("tracking.tracker.time.monotonic", return_value=100.5):
                    self.assertTrue(track.needs_recognition(0.5))

    def test_failed_pad_batch_invalidates_all_due_tracks(self):
        for outcome in [[], [{"is_real": True, "pad_score": 2.0}],
                        [{"is_real": True, "pad_score": float("nan")}] * 2,
                        RuntimeError("injected PAD failure")]:
            with self.subTest(outcome=str(outcome)):
                tracks = [self.verified_track(1), self.verified_track(2)]
                predictor = MagicMock(bbox_expansion_factor=1.55)
                if isinstance(outcome, Exception):
                    predictor.predict_crops.side_effect = outcome
                else:
                    predictor.predict_crops.return_value = outcome
                with contextlib.redirect_stdout(io.StringIO()), patch(
                    "tracking.tracker.time.monotonic", return_value=100
                ):
                    count = app.update_track_pad(
                        [({"bbox": t.bbox}, t) for t in tracks],
                        np.zeros((180, 180, 3), np.uint8), predictor, 0.1,
                    )
                self.assertEqual(count, 0)
                for track in tracks:
                    self.assertFalse(track.pad_ready)
                    self.assertIsNone(track.employee_id)
                    self.assertEqual(track.stable_recognitions, 0)
                    with patch("tracking.tracker.time.monotonic", return_value=100.05):
                        self.assertFalse(track.needs_pad(0.1))

    def test_pad_crop_failure_only_invalidates_failed_track(self):
        good, bad = self.verified_track(1), self.verified_track(2)
        predictor = MagicMock(bbox_expansion_factor=1.55)
        predictor.predict_crops.return_value = [{"is_real": True, "pad_score": 2.0}]
        with contextlib.redirect_stdout(io.StringIO()):
            count = app.update_track_pad(
                [({"bbox": good.bbox}, good), ({"bbox": (10, 10, 0, 0)}, bad)],
                np.zeros((180, 180, 3), np.uint8), predictor, 0.1,
            )
        self.assertEqual(count, 1)
        self.assertTrue(good.pad_ready)
        self.assertFalse(bad.pad_ready)
        self.assertIsNone(bad.employee_id)

    def test_pad_diagnostic_uses_each_tracks_source(self):
        tracks = [self.verified_track(1), self.verified_track(2)]
        predictor = MagicMock(bbox_expansion_factor=1.55, logit_threshold=0.0)
        predictor.predict_crops.return_value = [
            {"is_real": True, "pad_score": 1000.0},
            {"is_real": False, "pad_score": -1000.0},
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            count = app.update_track_pad(
                [({"bbox": t.bbox, "bbox_source": source}, t)
                 for t, source in zip(tracks, ["DETECTOR", "TRACKER"])],
                np.zeros((180, 180, 3), np.uint8), predictor, 0.1, diagnostic_log=True,
            )
        self.assertEqual(count, 2)
        lines = output.getvalue().splitlines()
        self.assertIn("track_id=1 detector_called=False bbox_source=DETECTOR", lines[0])
        self.assertIn("track_id=2 detector_called=False bbox_source=TRACKER", lines[1])


class TestRuntimeFrameLoop(unittest.TestCase):
    def run_frames(self, frames, detections, force_flow_failure=False, force_resolution=None):
        cap, detector, embedder, db = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        cap.get.return_value = 30.0
        detector.detect.return_value = detections
        embedder.input_size = (112, 112)
        embedder.embed_aligned.return_value = np.ones(512) / np.sqrt(512)
        db.search.return_value = [("A", "Alice", 0.9)]
        args = argparse.Namespace(pad_enabled=False, pad_threshold=app.config.PAD_THRESHOLD,
                                  pad_threshold_logit=None, pad_model=app.config.PAD_MODEL_PATH,
                                  camera=0, show_fps=False, mode="none")
        with contextlib.ExitStack() as stack:
            for name, value in [("parse_args", args), ("FaceDetector", detector),
                                ("FaceEmbedder", embedder), ("VectorDB", db),
                                ("cv2.VideoCapture", cap), ("cv2.waitKey", -1)]:
                stack.enter_context(patch("app." + name, return_value=value))
            stack.enter_context(patch("app.cv2.imshow"))
            stack.enter_context(patch("app.cv2.destroyAllWindows"))
            stack.enter_context(patch("app.config.DETECTION_INTERVAL_SECONDS", 1.0))
            stack.enter_context(patch("app.time.monotonic", return_value=100.0))
            if force_resolution is not None:
                stack.enter_context(patch("app.config.CAMERA_FORCE_RESOLUTION", force_resolution))
            if force_flow_failure:
                stack.enter_context(patch("tracking.tracker.cv2.calcOpticalFlowPyrLK",
                                          return_value=(None, None, None)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            app.main()
        self.camera_mock = cap
        return detector, embedder

    def test_camera_default_resolution_and_revert(self):
        for force_resolution in [False, True]:
            with self.subTest(force_resolution=force_resolution):
                self.run_frames([np.zeros((180, 180, 3), np.uint8)], [],
                                force_resolution=force_resolution)
                properties = [c.args[0] for c in self.camera_mock.set.call_args_list]
                self.assertEqual(app.cv2.CAP_PROP_FRAME_WIDTH in properties, force_resolution)
                self.assertEqual(app.cv2.CAP_PROP_FRAME_HEIGHT in properties, force_resolution)
                self.camera_mock.set.assert_any_call(app.cv2.CAP_PROP_FPS, app.config.CAMERA_FPS)
                if force_resolution:
                    self.camera_mock.set.assert_any_call(app.cv2.CAP_PROP_FRAME_WIDTH, app.config.CAMERA_WIDTH)
                    self.camera_mock.set.assert_any_call(app.cv2.CAP_PROP_FRAME_HEIGHT, app.config.CAMERA_HEIGHT)

    def test_overlapping_faces_recognize_from_unannotated_frame(self):
        frame = np.random.default_rng(42).integers(0, 256, (180, 180, 3), np.uint8)
        original = frame.copy()
        detections = [{"bbox": (30, 30, 130, 130)}, {"bbox": (60, 40, 160, 150)}]
        _, embedder = self.run_frames([frame], detections)
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(embedder.embed_aligned.call_count, 2)
        for call, det in zip(embedder.embed_aligned.call_args_list, detections):
            np.testing.assert_array_equal(call.args[0], get_input_face(original, det["bbox"], None))

    def test_failed_flow_redetects_in_same_frame(self):
        frame = np.zeros((180, 180, 3), np.uint8)
        detector, embedder = self.run_frames(
            [frame.copy(), frame.copy()], [{"bbox": (30, 30, 130, 130)}], True,
        )
        self.assertEqual(detector.detect.call_count, 2)
        self.assertEqual(embedder.embed_aligned.call_count, 2)


if __name__ == "__main__":
    unittest.main()
