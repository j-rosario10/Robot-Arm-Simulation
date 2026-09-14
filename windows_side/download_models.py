"""Download the MediaPipe Tasks model bundles the pose publisher needs.

MediaPipe 1.0 removed the legacy mp.solutions API; the Tasks API loads
explicit .task model bundles instead. Run once:

    .venv\\Scripts\\python download_models.py
"""

import urllib.request
from pathlib import Path

MODELS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def main() -> None:
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        target = model_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"already have {name} ({target.stat().st_size // 1024} kB)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, target)
        print(f"  -> {target} ({target.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
