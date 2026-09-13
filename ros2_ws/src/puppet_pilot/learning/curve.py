"""Success rate vs number of training demonstrations (Phase 7's key plot).

Trains a policy on growing subsets of the demonstrations and evaluates each.
Plots BOTH partial credit (parts correctly sorted / 8) and binary success -
SPEC prefers partial credit for exactly this plot, because binary success is
too coarse to show learning progress early on.

Run:  python -m puppet_pilot.learning.curve --sizes 5 10 20 35 50 --episodes 15
Output: media/learning_curve.png + media/learning_curve.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from puppet_pilot.learning.evaluate import evaluate
from puppet_pilot.learning.train import train
from puppet_pilot.recording.recorder import demos_dir, load_manifest
from puppet_pilot.spec import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=demos_dir())
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 35, 50])
    parser.add_argument("--episodes", type=int, default=15, help="eval episodes per point")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, default=repo_root() / "media")
    args = parser.parse_args()

    available = sum(
        1 for e in load_manifest(args.demos)["episodes"] if e["outcome"] == "success"
    )
    sizes = [s for s in args.sizes if s <= available]
    if not sizes:
        raise SystemExit(f"only {available} successful demos available; record more first")
    print(f"{available} successful demos; curve points at {sizes}")

    points = []
    with tempfile.TemporaryDirectory() as scratch:
        for size in sizes:
            checkpoint = Path(scratch) / f"bc_{size}.pt"
            print(f"\n=== {size} demonstrations ===")
            train(args.demos, checkpoint, epochs=args.epochs, max_episodes=size, quiet=True)
            report = evaluate(checkpoint, episodes=args.episodes, base_seed=size, quiet=True)
            print(f"  success {report['success_rate'] * 100:.0f}%, "
                  f"partial {report['mean_sorted']}/8")
            points.append({
                "demos": size,
                "success_rate": report["success_rate"],
                "mean_sorted": report["mean_sorted"],
            })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "learning_curve.json").write_text(json.dumps(points, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, left = plt.subplots(figsize=(7, 4.5))
    demos = [p["demos"] for p in points]
    left.plot(demos, [p["mean_sorted"] for p in points], "o-", color="#2b6cb0",
              label="parts correctly sorted (of 8)")
    left.set_xlabel("training demonstrations")
    left.set_ylabel("parts correctly sorted (of 8)", color="#2b6cb0")
    left.set_ylim(0, 8)
    right = left.twinx()
    right.plot(demos, [100 * p["success_rate"] for p in points], "s--", color="#c05621",
               label="episode success rate")
    right.set_ylabel("episode success rate [%]", color="#c05621")
    right.set_ylim(0, 100)
    left.set_title("Policy performance vs demonstration count")
    left.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.out / "learning_curve.png", dpi=150)
    print(f"\nwrote {args.out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
