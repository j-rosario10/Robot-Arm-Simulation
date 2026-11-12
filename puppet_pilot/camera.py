"""Webcam reader that never falls behind.

The camera is read in its own background thread, and only the NEWEST frame is
kept. If the rest of the program is briefly slow, old frames are simply thrown
away instead of piling up in a queue. That is what stops the "webcam is a
second behind" feeling, and it means a slow camera can never freeze anything
else.
"""

import threading
import time

import cv2


class LatestFrameCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        # CAP_DSHOW opens much faster than the default backend on Windows.
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ask the driver not to queue frames
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam {index}. Is another app using it?")

        self._lock = threading.Lock()
        self._frame = None        # newest frame
        self._stamp = 0.0         # when it was captured (time.time())
        self._count = 0           # goes up by one per new frame
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()      # waits for the next camera frame
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:                  # overwrite: old frame is dropped
                self._frame, self._stamp = frame, time.time()
                self._count += 1

    def read(self):
        """Returns (frame, capture_time, frame_number). frame is None until the
        first frame arrives. The same frame_number twice means 'nothing new'."""
        with self._lock:
            return self._frame, self._stamp, self._count

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
