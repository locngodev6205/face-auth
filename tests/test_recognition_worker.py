"""Offline tests: no camera, ONNX weights or network required."""
import time
import unittest
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np

from recognition.worker import RecognitionWorker, Request
from database.vector_db import VectorDB
from tracking.tracker import Track


class TestRecognitionWorker(unittest.TestCase):
    def make_worker(self, **kwargs):
        vector = np.ones(512, dtype=np.float32)
        embedder, db = MagicMock(), MagicMock()
        embedder.embed_aligned.return_value = vector
        db.search.return_value = [("NV001", "Alice", .9, vector)]
        worker = RecognitionWorker(embedder, db, "buffalo_s", .35, **kwargs)
        self.addCleanup(worker.close)
        return worker, db

    def receive(self, worker):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            results = worker.poll()
            if results:
                return results[0]
            time.sleep(.005)
        self.fail("Worker did not complete")

    def test_slow_cloud_does_not_block_submit_and_work_is_bounded(self):
        worker, db = self.make_worker(queue_size=1)
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        rows = db.search.return_value
        def slow(*args, **kwargs):
            entered.set()
            release.wait(2)
            return rows
        db.search.side_effect = slow
        face = np.zeros((112, 112, 3), np.uint8)
        self.assertTrue(worker.submit((1, 0), face))
        self.assertTrue(entered.wait(1))
        self.assertFalse(worker.submit((1, 0), face))
        self.assertTrue(worker.submit((2, 0), face))
        self.assertFalse(worker.submit((3, 0), face))
        self.assertEqual(len(worker.pending), 2)
        face[:] = 255
        self.assertEqual(worker.pending[(2, 0)].face.max(), 0)
        release.set()
        self.assertEqual(self.receive(worker).source, "CLOUD")

    def test_cache_hit_and_expiry_force_cloud_without_sliding_ttl(self):
        worker, db = self.make_worker(cache_ttl=30)
        request = Request((1, 0), np.zeros((112, 112, 3)), 0, Event())
        # Exercise worker-owned matching deterministically while queue is idle.
        with patch("recognition.worker.time.monotonic", return_value=100):
            self.assertEqual(worker._identify(request)[1], "CLOUD")
        with patch("recognition.worker.time.monotonic", return_value=125):
            self.assertEqual(worker._identify(request)[1], "CACHE")
            self.assertEqual(worker.cache[request.key][2], 130)
        with patch("recognition.worker.time.monotonic", return_value=131):
            self.assertEqual(worker._identify(request)[1], "CLOUD")
        self.assertEqual(db.search.call_count, 2)

    def test_new_track_or_reset_never_uses_another_tracks_cache(self):
        worker, db = self.make_worker()
        for key in [(1, 0), (2, 0), (1, 1)]:
            self.assertEqual(worker._identify(Request(key, np.zeros((112,112,3)), 0, Event()))[1], "CLOUD")
        self.assertEqual(db.search.call_count, 3)

    def test_low_cache_similarity_and_disabled_cache_query_cloud(self):
        for ttl in [0, 30]:
            worker, db = self.make_worker(cache_ttl=ttl)
            request = Request((1, 0), np.zeros((112,112,3)), 0, Event())
            worker._identify(request)
            worker.embedder.embed_aligned.return_value = np.concatenate([np.ones(256), -np.ones(256)])
            self.assertEqual(worker._identify(request)[1], "CLOUD")
            self.assertEqual(db.search.call_count, 2)

    def test_cloud_error_is_reported_not_an_unknown_match(self):
        worker, db = self.make_worker()
        db.search.side_effect = TimeoutError("secret DSN")
        worker.submit((1, 0), np.zeros((112,112,3)))
        result = self.receive(worker)
        self.assertEqual(result.error, "TimeoutError")
        self.assertEqual(result.source, "ERROR")
        self.assertEqual(worker.cache, {})

    def test_cancelled_result_cannot_escape_poll(self):
        worker, db = self.make_worker()
        worker.submit((1, 0), np.zeros((112,112,3)))
        worker.retain(set())
        deadline = time.monotonic() + 2
        while worker.pending and time.monotonic() < deadline:
            self.assertEqual(worker.poll(), [])
            time.sleep(.005)
        self.assertFalse(worker.pending)
        self.assertEqual(worker.stats["discarded"], 1)

    def test_collect_discards_old_generation_and_pad_failure(self):
        import app
        for reset in [True, False]:
            worker, db = self.make_worker()
            track = Track(1, (0, 0, 100, 100), pad_min_votes=1)
            track.update_pad(True)
            worker.submit((1, 0), np.zeros((112,112,3)))
            if reset:
                track.invalidate_verification()
            else:
                track._pad_window.clear()
                track.update_pad(False)
            deadline = time.monotonic() + 2
            while worker.pending and time.monotonic() < deadline:
                app.collect_recognition_results(worker, [({}, track)], True)
                time.sleep(.005)
            self.assertFalse(track.recognized_once)
            self.assertIsNone(track.employee_id)

    def test_collect_applies_late_valid_result_on_main_thread(self):
        import app
        worker, db = self.make_worker()
        release = Event()
        self.addCleanup(release.set)
        rows = db.search.return_value
        def slow_query(*args, **kwargs):
            release.wait(2)
            return rows
        db.search.side_effect = slow_query
        track = Track(1, (0, 0, 100, 100), pad_min_votes=1)
        track.update_pad(True)
        worker.submit((1, 0), np.zeros((112,112,3)))
        # Simulate an old capture timestamp without slowing the offline test.
        worker.pending[(1, 0)].submitted -= 10
        release.set()
        deadline = time.monotonic() + 2
        while not track.recognized_once and time.monotonic() < deadline:
            app.collect_recognition_results(worker, [({}, track)], True)
            time.sleep(.005)
        self.assertEqual(track.employee_id, "NV001")
        self.assertTrue(track.recognition_status.startswith("CLOUD"))
        self.assertGreaterEqual(track.recognition_latency, 10)

    def test_pad_transition_invalidates_cache_key_without_pending_job(self):
        import app
        worker, db = self.make_worker()
        track = Track(1, (0,0,100,100), pad_min_votes=1)
        track.update_pad(True)
        app.collect_recognition_results(worker, [({}, track)], True)
        old_generation = track.verification_generation
        track._pad_window.clear()
        track.update_pad(False)
        app.collect_recognition_results(worker, [({}, track)], True)
        self.assertGreater(track.verification_generation, old_generation)
        # Repeated invalid frames do not continuously increase the generation.
        generation = track.verification_generation
        app.collect_recognition_results(worker, [({}, track)], True)
        self.assertEqual(track.verification_generation, generation)

    def test_weak_cloud_match_is_not_cached(self):
        worker, db = self.make_worker()
        db.search.return_value = [("NV001", "Alice", .4, np.ones(512))]
        request = Request((1,0), np.zeros((112,112,3)), 0, Event())
        identity, source = worker._identify(request)
        self.assertEqual(identity[0], "NV001")  # Original matching threshold is unchanged.
        self.assertEqual(worker.cache, {})
        self.assertEqual(worker._identify(request)[1], "CLOUD")

    @patch("onnxruntime.InferenceSession")
    @patch("recognition.embedder.FaceAnalysis")
    def test_arcface_session_gets_pi_thread_options(self, analysis, session):
        from recognition.embedder import FaceEmbedder
        model = MagicMock(model_file="arcface.onnx", input_size=(112,112))
        analysis.return_value.models = {"recognition": model}
        embedder = FaceEmbedder(intra_op_threads=1)
        options = session.call_args.kwargs["sess_options"]
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.get_session_config_entry("session.intra_op.allow_spinning"), "0")
        self.assertIs(embedder._model, model)

    @patch("database.vector_db.ConnectionPool")
    def test_vector_query_returns_centroid_with_transaction_timeout(self, pool):
        with patch.object(VectorDB, "validate_schema"):
            db = VectorDB("fake", pool_timeout=3, query_timeout_ms=3000)
        conn = pool.return_value.connection.return_value.__enter__.return_value
        db.search(np.ones(512), model_version="buffalo_s", include_vectors=True)
        self.assertIn("f.embedding::text AS centroid", conn.execute.call_args.args[0])
        self.assertEqual(conn.execute.call_args_list[0].args[1], ("3000",))
        self.assertEqual(pool.call_args.kwargs["timeout"], 3)


if __name__ == "__main__":
    unittest.main()
