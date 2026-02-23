from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "pointmass_notebook.py"


def run_step(args: list[str]) -> None:
    cmd = [sys.executable, str(CLI), *args]
    print(f"[run_visualizations] Running: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all three visualization commands: "
            "visualize-dataset -> visualize-stage1 -> visualize-stage2."
        )
    )
    parser.add_argument("--config", type=str, default="config/default.json", help="Path to config JSON.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override --num-episodes for all three commands.")
    parser.add_argument("--fps", type=int, default=None, help="Override --fps for all three commands.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override --max-steps for stage1/stage2 visualizations.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset path for visualize-dataset.")
    parser.add_argument("--stage1-ckpt", type=str, default=None, help="Checkpoint path for visualize-stage1.")
    parser.add_argument("--stage2-ckpt", type=str, default=None, help="Checkpoint path for visualize-stage2.")
    parser.add_argument("--dataset-video", type=str, default=None, help="Output video path for visualize-dataset.")
    parser.add_argument("--stage1-video", type=str, default=None, help="Output video path for visualize-stage1.")
    parser.add_argument("--stage2-video", type=str, default=None, help="Output video path for visualize-stage2.")
    parser.add_argument(
        "--deterministic-action",
        action="store_true",
        help="Use mean action for stage1/stage2 visualization.",
    )
    return parser


def _append_common_vis_args(step: list[str], args: argparse.Namespace) -> None:
    if args.num_episodes is not None:
        step.extend(["--num-episodes", str(args.num_episodes)])
    if args.fps is not None:
        step.extend(["--fps", str(args.fps)])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_arg = ["--config", args.config]

    step = [*config_arg, "visualize-dataset"]
    _append_common_vis_args(step, args)
    if args.dataset_path is not None:
        step.extend(["--dataset-path", args.dataset_path])
    if args.dataset_video is not None:
        step.extend(["--output-video", args.dataset_video])
    run_step(step)

    step = [*config_arg, "visualize-stage1"]
    _append_common_vis_args(step, args)
    if args.max_steps is not None:
        step.extend(["--max-steps", str(args.max_steps)])
    if args.stage1_ckpt is not None:
        step.extend(["--ckpt", args.stage1_ckpt])
    if args.stage1_video is not None:
        step.extend(["--output-video", args.stage1_video])
    if args.deterministic_action:
        step.append("--deterministic-action")
    run_step(step)

    step = [*config_arg, "visualize-stage2"]
    _append_common_vis_args(step, args)
    if args.max_steps is not None:
        step.extend(["--max-steps", str(args.max_steps)])
    if args.stage2_ckpt is not None:
        step.extend(["--ckpt", args.stage2_ckpt])
    if args.stage2_video is not None:
        step.extend(["--output-video", args.stage2_video])
    if args.deterministic_action:
        step.append("--deterministic-action")
    run_step(step)

    print("[run_visualizations] Done.")


if __name__ == "__main__":
    main()

