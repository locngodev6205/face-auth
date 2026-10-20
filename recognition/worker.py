"""Single recognition consumer. Camera thread owns tracks and never waits for IO."""

from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Event, Thread
import time

import numpy as np
from database.vector_db import decide_identity


@dataclass
class Request:
    key: tuple
    face: np.ndarray
    submitted: float
    cancelled: Event


@dataclass
class Result:
    key: tuple
    identity: tuple
    source: str
    latency: float
    error: str = ""


class RecognitionWorker:
    def __init__(self, embedder, db, model_version, match_threshold,
                 queue_size=2, cache_ttl=30.0, cache_threshold=0.65):
        self.embedder = embedder
        self.db = db
        self.model_version = model_version
        self.match_threshold = match_threshold
        self.cache_threshold = max(match_threshold, cache_threshold)
        self.cache_ttl = cache_ttl
        self.requests = Queue(maxsize=queue_size)
        self.results = Queue(maxsize=queue_size + 1)
        self.pending = {}  # Main thread only; includes queued/running/completed jobs.
        self.visible_tracks = {}  # Camera thread only; detects disappearance between frames.
        self.cache = {}  # Worker only; per verification generation, not a partial gallery.
        self.stop = Event()
        self.stats = dict(submitted=0, cloud=0, cache=0, errors=0, discarded=0)
        self.thread = Thread(target=self._run, name="face-recognition", daemon=True)
        self.thread.start()

    def submit(self, key, aligned_face):
        if self.stop.is_set() or key in self.pending:
            return False
        # Also bound completed results waiting for consumption.
        if len(self.pending) >= self.requests.maxsize + 1:
            return False
        request = Request(key, aligned_face.copy(), time.monotonic(), Event())
        try:
            self.requests.put_nowait(request)
        except Full:
            return False
        self.pending[key] = request
        self.stats["submitted"] += 1
        return True

    def retain(self, valid_keys):
        # Do not remove pending jobs until consumed: cancelled work still occupies
        # the actual bounded queues and must not permit unbounded replacement work.
        for key, request in self.pending.items():
            if key not in valid_keys:
                request.cancelled.set()

    def poll(self):
        completed = []
        while True:
            try:
                result = self.results.get_nowait()
            except Empty:
                break
            request = self.pending.pop(result.key, None)
            if request is not None and not request.cancelled.is_set():
                completed.append(result)
            else:
                self.stats["discarded"] += 1
        return completed

    @staticmethod
    def _normalise(vector):
        if isinstance(vector, str):
            vector = np.fromstring(vector.strip("[]"), sep=",", dtype=np.float32)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size != 512 or not np.isfinite(vector).all() or norm <= 0:
            raise ValueError("Invalid recognition vector")
        return vector / norm

    def _identify(self, request):
        embedding = self._normalise(self.embedder.embed_aligned(request.face))
        now = time.monotonic()
        self.cache = {k: v for k, v in self.cache.items() if now < v[2]}
        cached = self.cache.get(request.key)
        if cached is not None:
            identity, centroid, _ = cached
            similarity = float(np.clip(np.dot(embedding, centroid), -1, 1))
            if similarity >= self.cache_threshold:
                self.stats["cache"] += 1
                return (*identity[:2], similarity), "CACHE"
        self.stats["cloud"] += 1
        # decide_identity consumes only the best candidate; send one centroid.
        rows = self.db.search(embedding, top_k=1, model_version=self.model_version,
                              include_vectors=True)
        identity = decide_identity([row[:3] for row in rows], self.match_threshold)
        # A cloud miss/error must never fall back to a previously cached identity.
        self.cache.pop(request.key, None)
        if identity[0] is not None and identity[2] >= self.cache_threshold and self.cache_ttl > 0:
            centroid = self._normalise(rows[0][3])
            # Cap RAM even when many tracks are created during the TTL.
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[request.key] = (identity, centroid, time.monotonic() + self.cache_ttl)
        return identity, "CLOUD"

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    request = self.requests.get(timeout=0.1)
                except Empty:
                    continue
                identity, source, error = (None, "UNKNOWN", 0.0), "CANCELLED", ""
                if not request.cancelled.is_set():
                    try:
                        identity, source = self._identify(request)
                    except Exception as exc:
                        self.cache.pop(request.key, None)
                        self.stats["errors"] += 1
                        # Do not expose DSNs or credentials through exception messages.
                        source, error = "ERROR", type(exc).__name__
                result = Result(request.key, identity, source,
                                time.monotonic() - request.submitted, error)
                while not self.stop.is_set():
                    try:
                        self.results.put(result, timeout=0.1)
                        break
                    except Full:
                        continue
        finally:
            self.cache.clear()
            self.db.close()

    def close(self):
        self.stop.set()
        for request in self.pending.values():
            request.cancelled.set()
        self.thread.join(timeout=2.0)
        # Worker closes DB itself after any in-flight IO, avoiding concurrent close.
        return not self.thread.is_alive()
