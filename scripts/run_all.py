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
    print(f"[run_all] Running: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full pointmass pipeline: "
            "generate-data -> visualize-dataset -> train-sft -> visualize-stage1 -> "
            "train-self-improve -> visualize-stage2 -> evaluate."
        )
    )
    parser.add_argument("--config", type=str, default="config/default.json", help="Path to config JSON.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override generate-data --num-episodes.")
    parser.add_argument("--sft-updates", type=int, default=None, help="Override train-sft --num-updates.")
    parser.add_argument(
        "--self-improve-iters",
        type=int,
        default=None,
        help="Override train-self-improve --num-iterations.",
    )
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override evaluate --num-episodes.")
    parser.add_argument("--vis-episodes", type=int, default=None, help="Override visualize-* --num-episodes.")
    parser.add_argument("--vis-fps", type=int, default=None, help="Override visualize-* --fps.")
    parser.add_argument("--vis-max-steps", type=int, default=None, help="Override visualize-stage* --max-steps.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip dataset generation and visualization.")
    parser.add_argument("--skip-sft", action="store_true", help="Skip stage 1 training and visualization.")
    parser.add_argument("--skip-self-improve", action="store_true", help="Skip stage 2 training and visualization.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip final evaluation.")
    return parser


def _extend_visualize_common_args(step: list[str], args: argparse.Namespace) -> None:
    if args.vis_episodes is not None:
        step.extend(["--num-episodes", str(args.vis_episodes)])
    if args.vis_fps is not None:
        step.extend(["--fps", str(args.vis_fps)])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_arg = ["--config", args.config]

    if not args.skip_generate:
        step = [*config_arg, "generate-data"]
        if args.num_episodes is not None:
            step.extend(["--num-episodes", str(args.num_episodes)])
        run_step(step)

        vis_step = [*config_arg, "visualize-dataset"]
        _extend_visualize_common_args(vis_step, args)
        run_step(vis_step)

    if not args.skip_sft:
        step = [*config_arg, "train-sft"]
        if args.sft_updates is not None:
            step.extend(["--num-updates", str(args.sft_updates)])
        run_step(step)

        vis_step = [*config_arg, "visualize-stage1"]
        _extend_visualize_common_args(vis_step, args)
        if args.vis_max_steps is not None:
            vis_step.extend(["--max-steps", str(args.vis_max_steps)])
        run_step(vis_step)

    if not args.skip_self_improve:
        step = [*config_arg, "train-self-improve"]
        if args.self_improve_iters is not None:
            step.extend(["--num-iterations", str(args.self_improve_iters)])
        run_step(step)

        vis_step = [*config_arg, "visualize-stage2"]
        _extend_visualize_common_args(vis_step, args)
        if args.vis_max_steps is not None:
            vis_step.extend(["--max-steps", str(args.vis_max_steps)])
        run_step(vis_step)

    if not args.skip_eval:
        step = [*config_arg, "evaluate"]
        if args.eval_episodes is not None:
            step.extend(["--num-episodes", str(args.eval_episodes)])
        run_step(step)

    print("[run_all] Done.")


if __name__ == "__main__":
    main()

