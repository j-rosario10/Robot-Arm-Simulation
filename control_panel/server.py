"""Puppet to Pilot web control panel.

One small HTTP server (Python standard library only - no Flask, no pip) that
runs on Windows and drives BOTH sides of the project:
  - the Windows camera publisher (a local process it launches)
  - the WSL robot side (bridge, simulator, teleop, training, evaluation),
    each started via `wsl.exe -- bash launch/run.sh <module> ...`

It serves a single control page at http://localhost:5055 with start/stop
buttons, live status, demonstration stats, and one-click train / evaluate.

Run:  C:\\Robot Arm Project\\windows_side\\.venv\\Scripts\\python control_panel\\server.py
(or just double-click control_panel\\OPEN_CONTROL_PANEL.cmd)
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WIN_VENV_PY = PROJECT / "windows_side" / ".venv" / "Scripts" / "python.exe"
LOG_DIR = PROJECT / "control_panel" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISTRO = "Ubuntu-24.04"
RUN_SH = "/mnt/c/Robot Arm Project/launch/run.sh"
CAMERA_PORT = 8877
HTTP_PORT = 5055

# Live-view frames written by the simulator (WSL) and the camera (Windows).
# Both sides see the same folder: one via /mnt/c, one via C:\.
LIVE_DIR = PROJECT / "control_panel" / "live"
LIVE_DIR_WSL = "/mnt/c/Robot Arm Project/control_panel/live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# Live-adjustable teleop settings (axis flips). The teleop node re-reads this
# file every second, so a click in the panel changes the mapping while the
# operator is driving - no restart, no code edit.
TELEOP_CONFIG = PROJECT / "control_panel" / "teleop_config.json"
_DEFAULT_TELEOP_CONFIG = {"flip_x": False, "flip_y": False, "flip_z": False,
                          "smoothness": 50}


def read_teleop_config() -> dict:
    try:
        cfg = json.loads(TELEOP_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    return {**_DEFAULT_TELEOP_CONFIG, **cfg}


def write_teleop_config(cfg: dict) -> None:
    TELEOP_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

# Windows console flag so the camera opens in its own visible window (the
# operator needs to see the calibration prompts and the tracking overlay).
CREATE_NEW_CONSOLE = 0x00000010

# ---- component registry ---------------------------------------------------
# side: where it runs. module: the WSL module string (also how we pgrep/pkill
# it). args: default flags. A Windows component instead names a script.

COMPONENTS = {
    "camera": {
        "side": "windows",
        "label": "Camera (MediaPipe)",
        "script": PROJECT / "windows_side" / "pose_publisher.py",
        # The page shows the camera view (with calibration prompts), so no
        # separate OpenCV window and no console - fewer things to close.
        "args": ["--no-window", "--stream", str(LIVE_DIR)],
        "windowed": False,
    },
    "bridge": {
        "side": "wsl",
        "label": "Socket \u2192 ROS bridge",
        "module": "puppet_pilot.teleop.bridge_node",
        "args": [],
    },
    "session": {
        "side": "wsl",
        "label": "Simulator + recorder",
        "module": "puppet_pilot.sim.session",
        # Headless + streamed to the page. The PyBullet X window was the
        # fragile part: closing it (or WSLg dropping it) killed the whole
        # simulator with "X connection to :0 broken".
        "args": ["--record", "--stream", LIVE_DIR_WSL],
    },
    "teleop": {
        "side": "wsl",
        "label": "Teleop control",
        "module": "puppet_pilot.teleop.teleop_node",
        "args": [],
    },
    "synthetic": {
        "side": "wsl",
        "label": "Synthetic operator (no camera)",
        "module": "puppet_pilot.tools.synthetic_operator",
        "args": [],
    },
}

# Long-running jobs (train / evaluate) are tracked the same way but by a
# unique module marker so status/log work identically.
JOBS = {
    "train": {
        "module": "puppet_pilot.learning.train",
        "args": ["--out", "/mnt/c/Robot Arm Project/data/policies/bc_latest.pt"],
        "label": "Training",
    },
    "evaluate": {
        "module": "puppet_pilot.learning.evaluate",
        "args": ["/mnt/c/Robot Arm Project/data/policies/bc_latest.pt",
                 "--episodes", "20",
                 "--out", "/mnt/c/Robot Arm Project/media/eval.json"],
        "label": "Evaluation",
    },
}

# Handles to Windows-side processes we launched (so we can stop them).
_win_procs: dict = {}
_lock = threading.Lock()


# ---- process control ------------------------------------------------------

def _log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def start_windows(name: str, cfg: dict) -> None:
    log = open(_log_path(name), "w", encoding="utf-8", errors="replace")
    flags = CREATE_NEW_CONSOLE if cfg.get("windowed") else 0
    proc = subprocess.Popen(
        [str(WIN_VENV_PY), str(cfg["script"]), *cfg["args"]],
        cwd=str(cfg["script"].parent),
        stdout=log, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    with _lock:
        _win_procs[name] = proc


def start_wsl(name: str, module: str, args: list, extra: list | None = None) -> None:
    log = open(_log_path(name), "w", encoding="utf-8", errors="replace")
    subprocess.Popen(
        ["wsl.exe", "-d", DISTRO, "--", "bash", RUN_SH, module, *args, *(extra or [])],
        stdout=log, stderr=subprocess.STDOUT,
    )


def stop_component(name: str) -> None:
    cfg = COMPONENTS.get(name) or JOBS.get(name)
    if cfg is None:
        return
    if cfg.get("side") == "windows":
        with _lock:
            proc = _win_procs.pop(name, None)
        if proc and proc.poll() is None:
            proc.terminate()
    else:
        # pkill the module inside WSL; -f matches the full command line.
        subprocess.run(
            ["wsl.exe", "-d", DISTRO, "--", "pkill", "-f", cfg["module"]],
            capture_output=True,
        )


def stop_all() -> None:
    for name in list(COMPONENTS) + list(JOBS):
        stop_component(name)


# ---- status ---------------------------------------------------------------

def camera_streaming() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", CAMERA_PORT)) == 0


def wsl_running(modules: list) -> dict:
    """One WSL call reports which of several module markers are alive."""
    # Two self-match traps to avoid, since the probe runs as one `bash -lc`
    # whose command line contains the module strings we search for:
    #   1. bracket the first char of the pgrep PATTERN ("[p]uppet...") so the
    #      pattern text in the cmdline is not matched by the pattern itself;
    #   2. echo only an INDEX, never the plain module name, so the unbracketed
    #      name is not reintroduced elsewhere on the same command line.
    lines = []
    for i, m in enumerate(modules):
        pat = f"[{m[0]}]{m[1:]}"
        lines.append(f'pgrep -f "{pat}" >/dev/null && echo "{i}=1" || echo "{i}=0"')
    result = subprocess.run(
        ["wsl.exe", "-d", DISTRO, "--", "bash", "-lc", "; ".join(lines)],
        capture_output=True, text=True,
    )
    alive = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            idx, value = line.rsplit("=", 1)
            if idx.strip().isdigit():
                alive[modules[int(idx)]] = value.strip() == "1"
    return alive


def manifest_summary() -> dict:
    path = PROJECT / "data" / "demos" / "manifest.json"
    if not path.exists():
        return {"episodes": 0, "successes": 0, "success_rate": 0, "mean_sorted": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("summary", {})
    except json.JSONDecodeError:
        return {"episodes": 0}


def policies() -> list:
    directory = PROJECT / "data" / "policies"
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.glob("*.pt"))


def eval_report() -> dict | None:
    path = PROJECT / "media" / "eval.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return {k: report[k] for k in
                ("success_rate", "mean_sorted", "mean_cycle_time", "episodes")
                if k in report}
    except json.JSONDecodeError:
        return None


def status() -> dict:
    wsl_modules = [c["module"] for c in COMPONENTS.values() if c["side"] == "wsl"]
    wsl_modules += [j["module"] for j in JOBS.values()]
    alive = wsl_running(wsl_modules)

    components = {}
    for name, cfg in COMPONENTS.items():
        if cfg["side"] == "windows":
            # Track the camera by its own process handle, not by the operator
            # port - the synthetic operator binds the same port, so a port
            # check would light up the camera dot during the no-camera demo.
            with _lock:
                proc = _win_procs.get(name)
            running = proc is not None and proc.poll() is None
        else:
            running = alive.get(cfg["module"], False)
        components[name] = {"label": cfg["label"], "running": running,
                            "side": cfg["side"]}
    jobs = {name: {"label": cfg["label"], "running": alive.get(cfg["module"], False)}
            for name, cfg in JOBS.items()}
    return {
        "components": components,
        "jobs": jobs,
        "demos": manifest_summary(),
        "policies": policies(),
        "eval": eval_report(),
    }


def tail_log(name: str, lines: int = 200) -> str:
    path = _log_path(name)
    if not path.exists():
        return "(no output yet)"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Drop PyBullet's noisy per-link warnings so the useful lines are visible.
    text = [ln for ln in text if "b3Warning" not in ln and "No inertial data" not in ln]
    return "\n".join(text[-lines:])


# ---- HTTP -----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def log_message(self, *args) -> None:
        pass  # keep the console quiet

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (Path(__file__).parent / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json(status())
        elif self.path.startswith("/live/"):
            self._serve_live(self.path[len("/live/"):].split("?")[0])
        elif self.path == "/api/teleop_config":
            self._json(read_teleop_config())
        elif self.path.startswith("/api/log/"):
            self._json({"log": tail_log(self.path.rsplit("/", 1)[1])})
        else:
            self._send(404, b"not found", "text/plain")

    def _serve_live(self, name: str) -> None:
        """Latest streamed frame / status, never cached by the browser."""
        if name not in ("frame.jpg", "camera.jpg", "status.json"):
            self._send(404, b"not found", "text/plain")
            return
        path = LIVE_DIR / name
        if not path.exists():
            self._send(404, b"no frame yet", "text/plain")
            return
        content_type = "image/jpeg" if name.endswith(".jpg") else "application/json"
        try:
            body = path.read_bytes()
        except OSError:  # mid-replace; the client will retry in 200 ms
            self._send(503, b"retry", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parts = self.path.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "api":
            self._send(404, b"not found", "text/plain")
            return
        action = parts[1]
        target = parts[2] if len(parts) > 2 else None

        if action == "start" and target in COMPONENTS:
            cfg = COMPONENTS[target]
            if cfg["side"] == "windows":
                start_windows(target, cfg)
            else:
                start_wsl(target, cfg["module"], cfg["args"])
            self._json({"ok": True})
        elif action == "stop" and target in COMPONENTS:
            stop_component(target)
            self._json({"ok": True})
        elif action == "stop_all":
            stop_all()
            self._json({"ok": True})
        elif action == "start_all":
            self._start_all(camera=True)
            self._json({"ok": True})
        elif action == "start_demo":
            self._start_all(camera=False)
            self._json({"ok": True})
        elif action == "smoothness" and target is not None and target.isdigit():
            cfg = read_teleop_config()
            cfg["smoothness"] = max(0, min(100, int(target)))
            write_teleop_config(cfg)
            self._json(cfg)
        elif action == "flip" and target in ("x", "y", "z"):
            cfg = read_teleop_config()
            cfg[f"flip_{target}"] = not cfg[f"flip_{target}"]
            write_teleop_config(cfg)
            self._json(cfg)
        elif action == "job" and target in JOBS:
            cfg = JOBS[target]
            start_wsl(target, cfg["module"], cfg["args"])
            self._json({"ok": True})
        else:
            self._send(404, b"unknown action", "text/plain")

    def _start_all(self, camera: bool) -> None:
        """Bring the whole stack up in dependency order."""
        for stale in ("frame.jpg", "camera.jpg", "status.json"):
            try:
                (LIVE_DIR / stale).unlink()
            except OSError:
                pass
        if camera:
            start_windows("camera", COMPONENTS["camera"])
        else:
            start_wsl("synthetic", COMPONENTS["synthetic"]["module"], [])
        time.sleep(1.5)
        # Bridge points at the camera (NAT) or the synthetic operator (loopback).
        start_wsl("bridge", COMPONENTS["bridge"]["module"],
                  COMPONENTS["bridge"]["args"], [] if camera else ["--host", "127.0.0.1"])
        start_wsl("session", COMPONENTS["session"]["module"], COMPONENTS["session"]["args"])
        time.sleep(4.0)  # let the sim finish loading before teleop seeds from it
        start_wsl("teleop", COMPONENTS["teleop"]["module"], COMPONENTS["teleop"]["args"])


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    url = f"http://localhost:{HTTP_PORT}"
    print(f"Puppet to Pilot control panel: {url}")
    print("Leave this window open. Ctrl+C to stop the panel (not the robot).")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ncontrol panel stopped (running components keep running)")


if __name__ == "__main__":
    if not WIN_VENV_PY.exists():
        sys.exit(f"Windows venv python not found at {WIN_VENV_PY}")
    main()
