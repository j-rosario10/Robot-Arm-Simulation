"""Windows-side perception: webcam -> MediaPipe -> normalised values over TCP.

This script deliberately does NOT import anything from the WSL-side code.
The Windows/WSL boundary is the project's architectural seam (see
docs/phase0-environment.md): this side owns the camera and publishes
*meaning* — a shoulder-relative wrist vector and a calibrated pinch value —
about 40 bytes per frame, never video.

Wire protocol (newline-delimited JSON, one object per frame):
    {"t": <epoch s>, "pose_ok": bool, "hand_ok": bool,
     "wrist": [x, y, z],   # shoulder-relative, in units of the operator's arm length,
                           # MediaPipe world-frame axes (x right, y down, z away from camera)
     "pinch": 0..1}        # calibrated: 0 = fully pinched, 1 = fully open

Usage:
    .venv\\Scripts\\python pose_publisher.py            # calibrate, then publish
    .venv\\Scripts\\python pose_publisher.py --self-test
Options: --camera N, --arm left|right, --port, --no-window.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_PORT = 8877

# MediaPipe pose landmark indices (shoulder, elbow, wrist) per arm.
_ARM_LANDMARKS = {"left": (11, 13, 15), "right": (12, 14, 16)}
# Hand landmark indices: thumb tip, index tip, wrist, middle-finger MCP.
_THUMB_TIP, _INDEX_TIP, _HAND_WRIST, _MIDDLE_MCP = 4, 8, 0, 9

_CALIBRATION_SECONDS = 2.5  # per stage (hold open / hold pinched)


class FrameClock:
    """Strictly increasing millisecond timestamps for MediaPipe VIDEO mode.

    MediaPipe requires every detect_for_video() call on a detector to carry a
    timestamp larger than the previous one. One shared clock (rather than a
    clock per phase) guarantees that across calibration and the main loop -
    the original bug: calibration used absolute time, the main loop restarted
    from zero, timestamps went backwards, MediaPipe raised, camera died.
    The +2 floor guards against two frames landing in the same millisecond
    (pose is stamped t, hands t+1).
    """

    def __init__(self):
        self._start = time.monotonic()
        self._last = -2

    def next_ms(self) -> int:
        now = int((time.monotonic() - self._start) * 1000)
        self._last = max(now, self._last + 2)
        return self._last


def publish_rate_from_spec() -> float:
    """The pose publish rate lives in SPEC.md, like every task parameter.

    Parsed with a local regex rather than importing the WSL-side parser, to
    keep the two sides decoupled.
    """
    spec = Path(__file__).resolve().parents[1] / "SPEC.md"
    match = re.search(r"\| Pose publish rate \| (\d+(?:\.\d+)?) Hz", spec.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError("SPEC.md: could not find the pose publish rate")
    return float(match.group(1))


# --------------------------------------------------------------------------
# perception
# --------------------------------------------------------------------------

class Perception:
    """Wraps the two MediaPipe Tasks landmarkers and derives our two signals."""

    def __init__(self, arm: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        self.arm = arm

        model_dir = Path(__file__).resolve().parent / "models"
        pose_model = model_dir / "pose_landmarker_lite.task"
        hand_model = model_dir / "hand_landmarker.task"
        for model in (pose_model, hand_model):
            if not model.exists():
                raise FileNotFoundError(f"{model} missing - run download_models.py first")

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
            )
        )
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self._arm_lengths: list = []  # running history for the median
        self.clock = FrameClock()
        self._frame_index = 0
        self._last_hand_result = None

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> dict:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        pose_result = self.pose.detect_for_video(image, timestamp_ms)
        # The hand detector is the expensive half on CPU. Running it every
        # other frame lifts the camera from ~17 to ~25 fps; the pinch value
        # changes far more slowly than that, so nothing is lost.
        self._frame_index += 1
        if self._frame_index % 2 == 0 or self._last_hand_result is None:
            self._last_hand_result = self.hand.detect_for_video(image, timestamp_ms + 1)
        hand_result = self._last_hand_result

        out = {"pose_ok": False, "hand_ok": False, "wrist": None, "pinch_raw": None,
               "arm_image": None, "hand_image": None}

        if pose_result.pose_landmarks:  # image-space (normalised) points, for drawing
            marks_img = pose_result.pose_landmarks[0]
            out["arm_image"] = [(marks_img[i].x, marks_img[i].y) for i in _ARM_LANDMARKS[self.arm]]

        if pose_result.pose_world_landmarks:
            marks = pose_result.pose_world_landmarks[0]
            s_i, e_i, w_i = _ARM_LANDMARKS[self.arm]
            shoulder = np.array([marks[s_i].x, marks[s_i].y, marks[s_i].z])
            elbow = np.array([marks[e_i].x, marks[e_i].y, marks[e_i].z])
            wrist = np.array([marks[w_i].x, marks[w_i].y, marks[w_i].z])

            # Normalising by the operator's own arm length makes the mapping
            # invariant to camera distance and to who is operating. Running
            # median: per-frame length estimates are noisy.
            self._arm_lengths.append(
                float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))
            )
            if len(self._arm_lengths) > 300:
                self._arm_lengths.pop(0)
            arm_length = float(np.median(self._arm_lengths))
            if arm_length > 0.05:
                out["pose_ok"] = True
                out["wrist"] = ((wrist - shoulder) / arm_length).tolist()

        if hand_result.hand_world_landmarks:
            # Pick the hand matching the tracked arm, if handedness is known.
            index = 0
            for i, handed in enumerate(hand_result.handedness or []):
                if handed and handed[0].category_name.lower() == self.arm:
                    index = i
                    break
            marks = hand_result.hand_world_landmarks[index]
            marks_img = hand_result.hand_landmarks[index]
            out["hand_image"] = [(marks_img[_THUMB_TIP].x, marks_img[_THUMB_TIP].y),
                                 (marks_img[_INDEX_TIP].x, marks_img[_INDEX_TIP].y)]

            def point(i):
                return np.array([marks[i].x, marks[i].y, marks[i].z])

            span = float(np.linalg.norm(point(_MIDDLE_MCP) - point(_HAND_WRIST)))
            if span > 1e-4:
                # Thumb-tip to index-tip distance normalised by hand span:
                # invariant to distance from the camera (SPEC's definition).
                out["hand_ok"] = True
                out["pinch_raw"] = float(
                    np.linalg.norm(point(_THUMB_TIP) - point(_INDEX_TIP)) / span
                )
        return out


class PinchCalibration:
    """Maps the raw pinch ratio onto a personal 0..1 range.

    SPEC requires calibration at startup (hold open, then pinched) because
    hand spans and thumb reach differ between people.
    """

    def __init__(self, open_value: float, closed_value: float):
        if open_value - closed_value < 0.1:
            raise RuntimeError(
                f"calibration range too small (open {open_value:.2f}, closed "
                f"{closed_value:.2f}) - redo with clearer open/pinch poses"
            )
        self.open_value = open_value
        self.closed_value = closed_value

    def apply(self, raw: float) -> float:
        return float(np.clip(
            (raw - self.closed_value) / (self.open_value - self.closed_value), 0.0, 1.0
        ))


# --------------------------------------------------------------------------
# TCP publisher
# --------------------------------------------------------------------------

class Publisher:
    """Accepts one client at a time and streams the latest sample to it."""

    def __init__(self, port: int):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", port))
        self.server.listen(1)
        self.client = None
        self.lock = threading.Lock()
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[publisher] listening on 0.0.0.0:{port}")

    def _accept_loop(self) -> None:
        while True:
            client, address = self.server.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                if self.client is not None:
                    self.client.close()
                self.client = client
            print(f"[publisher] client connected from {address[0]}")

    def send(self, sample: dict) -> None:
        line = (json.dumps(sample) + "\n").encode()
        with self.lock:
            if self.client is None:
                return
            try:
                self.client.sendall(line)
            except OSError:
                self.client.close()
                self.client = None
                print("[publisher] client disconnected")


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    import cv2

    rate = publish_rate_from_spec()
    perception = Perception(args.arm)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}")

    streamer = FrameStreamer(args.stream, cv2) if args.stream else None

    if args.skip_calibration:
        # Debug path: raw thumb-index ratio with a nominal range. Good enough
        # to verify the wire; a real session must calibrate (SPEC requires it).
        print("[calibrate] SKIPPED (debug flag) - using a nominal pinch range")
        calibration = PinchCalibration(open_value=1.2, closed_value=0.3)
    else:
        # Retry calibration until it succeeds - the operator may not be in
        # frame yet when the window first opens.
        while True:
            try:
                calibration = _calibrate(capture, perception, args, cv2, streamer)
                break
            except RuntimeError as error:
                print(f"[calibrate] {error} - retrying in 2 s (get hand in frame)")
                time.sleep(2.0)
    publisher = Publisher(args.port)

    period = 1.0 / rate
    last_wrist, last_pinch = [0.0, 0.0, 0.0], 1.0
    print(f"[publisher] streaming at {rate:.0f} Hz - Ctrl+C or 'q' in the window to stop")

    while True:
        tick = time.monotonic()
        ok, frame = capture.read()
        if not ok:
            time.sleep(period)
            continue
        result = _safe_process(perception, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result["pose_ok"]:
            last_wrist = result["wrist"]
        if result["hand_ok"]:
            last_pinch = calibration.apply(result["pinch_raw"])

        publisher.send({
            "t": time.time(),
            "pose_ok": result["pose_ok"],
            "hand_ok": result["hand_ok"],
            "wrist": [round(v, 4) for v in last_wrist],
            "pinch": round(last_pinch, 4),
        })

        # Mirror the DISPLAY only (selfie view - what people expect from a
        # webcam). Perception already ran on the raw frame, so tracking and
        # handedness are unaffected.
        if not args.no_mirror_view:
            frame = cv2.flip(frame, 1)
        _draw_overlay(cv2, frame, result, last_pinch, mirrored=not args.no_mirror_view)
        if streamer:
            streamer.maybe_write(frame)
        if not args.no_window:
            cv2.imshow("puppet_pilot operator view", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


_EMPTY = {"pose_ok": False, "hand_ok": False, "wrist": None, "pinch_raw": None,
          "arm_image": None, "hand_image": None}


def _safe_process(perception: Perception, frame_rgb) -> dict:
    """Run perception; a single bad frame is logged and skipped, never fatal."""
    try:
        return perception.process(frame_rgb, perception.clock.next_ms())
    except Exception as error:  # noqa: BLE001 - MediaPipe raises plain ValueError
        print(f"[perception] frame skipped: {error}")
        return dict(_EMPTY)


class FrameStreamer:
    """Writes the annotated camera frame to <dir>/camera.jpg a few times a
    second (atomic replace) so the web control panel can show it."""

    def __init__(self, directory: str, cv2):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cv2 = cv2
        self._last = 0.0

    def maybe_write(self, frame) -> None:
        now = time.monotonic()
        if now - self._last < 0.2:  # ~5 fps is plenty for a monitor view
            return
        self._last = now
        tmp = self.dir / "camera.tmp.jpg"
        self.cv2.imwrite(str(tmp), frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 70])
        try:
            tmp.replace(self.dir / "camera.jpg")
        except OSError:
            pass


def _calibrate(capture, perception: Perception, args, cv2, streamer=None) -> PinchCalibration:
    """SPEC's required startup routine: hold open, then hold pinched."""
    values = {}
    for stage, instruction in (("open", "hold your hand fully OPEN"),
                               ("closed", "PINCH thumb and index together")):
        print(f"[calibrate] {instruction} ... ", end="", flush=True)
        time.sleep(1.0)  # give the operator a moment to comply
        samples: list = []
        start = time.monotonic()
        while time.monotonic() - start < _CALIBRATION_SECONDS:
            ok, frame = capture.read()
            if not ok:
                continue
            result = _safe_process(perception, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result["hand_ok"]:
                samples.append(result["pinch_raw"])
            if not args.no_mirror_view:
                frame = cv2.flip(frame, 1)
            _draw_arm_nodes(cv2, frame, result, mirrored=not args.no_mirror_view)
            cv2.putText(frame, f"CALIBRATION: {instruction}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            if streamer:
                streamer.maybe_write(frame)
            if not args.no_window:
                cv2.imshow("puppet_pilot operator view", frame)
                cv2.waitKey(1)
        if len(samples) < 10:
            raise RuntimeError(f"calibration failed: hand not visible during '{stage}' stage")
        values[stage] = float(np.median(samples))
        print(f"median {values[stage]:.3f} ({len(samples)} samples)")
    return PinchCalibration(open_value=values["open"], closed_value=values["closed"])


def _draw_arm_nodes(cv2, frame, result: dict, mirrored: bool) -> None:
    """Draw ONLY what the controller uses: shoulder-elbow-wrist and the
    thumb/index pinch line. (The pose model sees the whole body; this makes
    it visible which parts actually matter.)"""
    h, w = frame.shape[:2]

    def px(point):
        x, y = point
        if mirrored:
            x = 1.0 - x
        return int(x * w), int(y * h)

    arm = result.get("arm_image")
    if arm:
        pts = [px(pt) for pt in arm]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 220, 120), 3)
        for label, pt in zip(("shoulder", "elbow", "wrist"), pts):
            cv2.circle(frame, pt, 8, (0, 220, 120), -1)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1)
            cv2.putText(frame, label, (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1)
    hand = result.get("hand_image")
    if hand:
        thumb, index = px(hand[0]), px(hand[1])
        cv2.line(frame, thumb, index, (255, 180, 0), 2)
        cv2.circle(frame, thumb, 6, (255, 180, 0), -1)
        cv2.circle(frame, index, 6, (255, 180, 0), -1)


def _draw_overlay(cv2, frame, result: dict, pinch: float, mirrored: bool = True) -> None:
    _draw_arm_nodes(cv2, frame, result, mirrored)
    h = frame.shape[0]
    pose_colour = (0, 220, 0) if result["pose_ok"] else (0, 0, 255)
    hand_colour = (0, 220, 0) if result["hand_ok"] else (0, 0, 255)
    cv2.putText(frame, "pose", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_colour, 2)
    cv2.putText(frame, "hand", (100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_colour, 2)
    bar_width = int(150 * pinch)
    cv2.rectangle(frame, (20, h - 35), (20 + bar_width, h - 20), (255, 180, 0), -1)
    cv2.rectangle(frame, (20, h - 35), (170, h - 20), (200, 200, 200), 1)
    cv2.putText(frame, f"pinch {pinch:.2f}", (180, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)


def self_test(args: argparse.Namespace) -> int:
    """Verify everything short of a human in frame. Exit 0 on success."""
    import cv2

    print(f"python {sys.version.split()[0]}")
    import mediapipe
    print(f"mediapipe {mediapipe.__version__}")
    print(f"SPEC publish rate: {publish_rate_from_spec():.0f} Hz")

    perception = Perception(args.arm)
    print("models loaded: pose_landmarker_lite, hand_landmarker")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"WARNING: camera {args.camera} did not open (in use, or none present)")
        return 1
    ok, frame = capture.read()
    capture.release()
    if not ok:
        print("WARNING: camera opened but produced no frame")
        return 1
    result = perception.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), perception.clock.next_ms())
    print(f"camera frame {frame.shape[1]}x{frame.shape[0]} processed; "
          f"pose_ok={result['pose_ok']} hand_ok={result['hand_ok']} "
          f"(both False is normal with nobody in frame)")
    print("SELF-TEST PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stream", default=None,
                        help="directory to write camera.jpg into (~5 fps) for the web panel")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="DEBUG ONLY: publish with an uncalibrated pinch mapping")
    parser.add_argument("--no-mirror-view", action="store_true",
                        help="show the raw (un-mirrored) camera image instead of selfie view")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test(args))
    run(args)


if __name__ == "__main__":
    main()
