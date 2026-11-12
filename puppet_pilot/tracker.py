"""Webcam -> hand landmarks -> arm target, with a preview window.

HandPipeline does one job per call to tick():
    newest camera frame -> mirror it -> MediaPipe finds the hand ->
    hand_math reads it -> mapping turns it into an arm target.

It has no ROS in it. Run this file on its own to try the hand tracking
without the robot:

    pixi run hand
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from puppet_pilot.camera import LatestFrameCamera
from puppet_pilot.hand_math import INDEX_TIP, THUMB_TIP, read_hand
from puppet_pilot.mapping import HandToArm

MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
WIDTH, HEIGHT = 640, 480
WINDOW = "Puppet to Pilot - hand"

GREEN, RED, WHITE, ORANGE = (90, 220, 90), (60, 60, 230), (255, 255, 255), (0, 170, 255)
BONES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


class HandPipeline:
    def __init__(self, camera_index: int = 0):
        self.camera = LatestFrameCamera(camera_index, WIDTH, HEIGHT)
        self.mapping = HandToArm()
        # One hand, video mode. Video mode tracks the hand from frame to frame
        # instead of searching the whole image every time, which is much faster.
        self._detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
            )
        )
        self._last_frame_number = -1
        self._last_time = time.perf_counter()
        self._start = time.perf_counter()
        self.fps = 0.0            # hand tracking rate
        self.latency_ms = 0.0     # camera capture -> result ready
        self.hand_visible = False

    def tick(self):
        """Process the newest frame. Returns None if the camera has nothing new
        yet, otherwise (preview_image, target_xyz, gripper_closed, capture_time)."""
        frame, captured_at, number = self.camera.read()
        if frame is None or number == self._last_frame_number:
            return None
        self._last_frame_number = number

        frame = cv2.flip(frame, 1)        # mirror, so it behaves like a mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.perf_counter() - self._start) * 1000)
        result = self._detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

        points, hand = None, None
        if result.hand_landmarks:
            points = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
            hand = read_hand(points, aspect=frame.shape[1] / frame.shape[0])
        self.hand_visible = hand is not None

        target, closed = self.mapping.update(hand, time.perf_counter())

        # timing, lightly averaged so the numbers on screen are readable
        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._last_time, 1e-6)
        self._last_time = now
        self.latency_ms = 0.9 * self.latency_ms + 0.1 * (time.time() - captured_at) * 1000

        self._draw_hand(frame, points, closed)
        return frame, target, closed, captured_at

    def _draw_hand(self, frame, points, closed: bool) -> None:
        if points is None:
            return
        h, w = frame.shape[:2]
        px = [(int(x * w), int(y * h)) for x, y in points]
        for a, b in BONES:
            cv2.line(frame, px[a], px[b], GREEN, 2)
        for p in px:
            cv2.circle(frame, p, 3, WHITE, -1)
        # the pinch line: red when the gripper is closed
        cv2.line(frame, px[THUMB_TIP], px[INDEX_TIP], RED if closed else ORANGE, 3)

    def close(self) -> None:
        self.camera.close()


def draw_status(frame, pipeline: HandPipeline, closed: bool, lines: list) -> None:
    """Text overlay: gripper state, tracking speed, plus any extra lines."""
    text = [
        ("GRIPPER CLOSED" if closed else "GRIPPER OPEN", RED if closed else GREEN),
        (f"hand {pipeline.fps:4.0f} fps   {pipeline.latency_ms:3.0f} ms", WHITE),
    ] + [(line, WHITE) for line in lines]
    if not pipeline.hand_visible:
        text.append(("NO HAND - arm holding", ORANGE))
    for i, (line, colour) in enumerate(text):
        y = 24 + 24 * i
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(frame, "C re-centre   R reset   Q quit", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


def main() -> None:
    """Hand tracking on its own: preview window, no robot."""
    pipeline = HandPipeline()
    try:
        while True:
            out = pipeline.tick()
            if out is not None:
                frame, target, closed, _ = out
                x, y, z = target
                draw_status(frame, pipeline, closed, [f"target x{x:.2f} y{y:+.2f} z{z:.2f} m"])
                cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                pipeline.mapping.recentre()
            elif key == ord("q"):
                break
    finally:
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
