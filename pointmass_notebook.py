from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from src.pointmass.config import load_config
from src.pointmass.utils import ensure_dirs, path_from_cfg, seed_everything


def _resolve_dataset_path(cfg: Dict[str, Any], dataset_path: Optional[str]) -> Path:
    if dataset_path is not None:
        return Path(dataset_path)
    return path_from_cfg(cfg["paths"]["data_dir"], cfg["paths"]["dataset_file"])


def _resolve_sft_ckpt_path(cfg: Dict[str, Any], ckpt_path: Optional[str]) -> Path:
    if ckpt_path is not None:
        return Path(ckpt_path)
    return path_from_cfg(cfg["paths"]["ckpt_dir"], cfg["paths"]["sft_ckpt"])


def _resolve_self_improve_ckpt_path(cfg: Dict[str, Any], ckpt_path: Optional[str]) -> Path:
    if ckpt_path is not None:
        return Path(ckpt_path)
    return path_from_cfg(cfg["paths"]["ckpt_dir"], cfg["paths"]["self_improve_ckpt"])


def cmd_generate_data(args: argparse.Namespace) -> None:
    from src.pointmass.trainer import build_env_from_cfg, generate_pd_dataset, save_dataset

    cfg = load_config(args.config)
    ensure_dirs(cfg["paths"])
    seed_everything(int(cfg["seed"]))

    if args.num_episodes is not None:
        cfg["dataset"]["num_episodes"] = int(args.num_episodes)

    env = build_env_from_cfg(cfg)
    episodes, tuples, episode_lens = generate_pd_dataset(
        env=env,
        num_episodes=int(cfg["dataset"]["num_episodes"]),
        num_waypoints_per_episode=int(cfg["dataset"]["num_waypoints_per_episode"]),
        episode_len_discard_thresh=int(cfg["dataset"]["episode_len_discard_thresh"]),
    )

    out_path = _resolve_dataset_path(cfg, args.dataset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(out_path, episodes=episodes, tuples=tuples, episode_lens=episode_lens)

    print(f"[Data] Saved: {out_path}")
    print(
        "[Data] Stats: "
        f"num_episodes={episode_lens.shape[0]} "
        f"mean_len={episode_lens.mean():.2f} "
        f"std_len={episode_lens.std():.2f} "
        f"min_len={episode_lens.min()} "
        f"max_len={episode_lens.max()}"
    )


def cmd_train_sft(args: argparse.Namespace) -> None:
    from src.pointmass.trainer import run_sft_training

    cfg = load_config(args.config)
    ensure_dirs(cfg["paths"])
    if args.num_updates is not None:
        cfg["stage1"]["num_updates"] = int(args.num_updates)
    if args.batch_size is not None:
        cfg["stage1"]["batch_size"] = int(args.batch_size)

    dataset_path = _resolve_dataset_path(cfg, args.dataset_path)
    ckpt_path = _resolve_sft_ckpt_path(cfg, args.output_ckpt)
    resume_ckpt_path = Path(args.resume_ckpt) if args.resume_ckpt is not None else None

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run `generate-data` first or pass --dataset-path."
        )

    run_sft_training(
        cfg=cfg,
        dataset_path=dataset_path,
        output_ckpt_path=ckpt_path,
        resume_ckpt_path=resume_ckpt_path,
    )


def cmd_train_self_improve(args: argparse.Namespace) -> None:
    from src.pointmass.trainer import run_self_improve_training

    cfg = load_config(args.config)
    ensure_dirs(cfg["paths"])
    if args.num_iterations is not None:
        cfg["stage2"]["num_iterations"] = int(args.num_iterations)
    if args.env_steps_per_batch is not None:
        cfg["stage2"]["env_steps_per_batch"] = int(args.env_steps_per_batch)

    init_ckpt = _resolve_sft_ckpt_path(cfg, args.init_ckpt)
    if not init_ckpt.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found at {init_ckpt}. Run `train-sft` first or pass --init-ckpt."
        )

    output_ckpt = _resolve_self_improve_ckpt_path(cfg, args.output_ckpt)
    run_self_improve_training(cfg=cfg, init_ckpt_path=init_ckpt, output_ckpt_path=output_ckpt)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.pointmass.trainer import evaluate_policy

    cfg = load_config(args.config)
    ensure_dirs(cfg["paths"])
    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt is not None
        else _resolve_self_improve_ckpt_path(cfg, None)
    )
    if not ckpt_path.exists():
        fallback = _resolve_sft_ckpt_path(cfg, None)
        if fallback.exists():
            ckpt_path = fallback
        else:
            raise FileNotFoundError(
                "No checkpoint found. Run `train-sft` (and optionally `train-self-improve`) first."
            )

    num_episodes = int(args.num_episodes) if args.num_episodes is not None else int(cfg["eval"]["num_episodes"])
    deterministic_action = bool(args.deterministic_action) or bool(cfg["eval"].get("deterministic_action", False))
    metrics = evaluate_policy(
        cfg=cfg,
        ckpt_path=ckpt_path,
        num_episodes=num_episodes,
        deterministic_action=deterministic_action,
    )
    print(f"[Eval] checkpoint={ckpt_path}")
    for k, v in metrics.items():
        print(f"[Eval] {k}={v:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pointmass PyTorch pipeline with CLI commands."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.json",
        help="Path to JSON config file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_data = subparsers.add_parser("generate-data", help="Generate PD dataset.")
    p_data.add_argument("--dataset-path", type=str, default=None, help="Output dataset file path.")
    p_data.add_argument("--num-episodes", type=int, default=None, help="Override dataset.num_episodes.")
    p_data.set_defaults(func=cmd_generate_data)

    p_sft = subparsers.add_parser("train-sft", help="Run Stage 1 supervised training.")
    p_sft.add_argument("--dataset-path", type=str, default=None, help="Input dataset file path.")
    p_sft.add_argument("--output-ckpt", type=str, default=None, help="Output SFT checkpoint path.")
    p_sft.add_argument("--resume-ckpt", type=str, default=None, help="Resume training from this checkpoint.")
    p_sft.add_argument("--num-updates", type=int, default=None, help="Override stage1.num_updates.")
    p_sft.add_argument("--batch-size", type=int, default=None, help="Override stage1.batch_size.")
    p_sft.set_defaults(func=cmd_train_sft)

    p_stage2 = subparsers.add_parser("train-self-improve", help="Run Stage 2 REINFORCE training.")
    p_stage2.add_argument("--init-ckpt", type=str, default=None, help="Input SFT checkpoint path.")
    p_stage2.add_argument("--output-ckpt", type=str, default=None, help="Output self-improve checkpoint path.")
    p_stage2.add_argument("--num-iterations", type=int, default=None, help="Override stage2.num_iterations.")
    p_stage2.add_argument(
        "--env-steps-per-batch",
        type=int,
        default=None,
        help="Override stage2.env_steps_per_batch.",
    )
    p_stage2.set_defaults(func=cmd_train_self_improve)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    p_eval.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. Defaults to self-improve ckpt, then SFT.")
    p_eval.add_argument("--num-episodes", type=int, default=None, help="Number of episodes for evaluation.")
    p_eval.add_argument("--deterministic-action", action="store_true", help="Use mean action instead of sampling.")
    p_eval.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ModuleNotFoundError as exc:
        missing = exc.name if exc.name is not None else str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}. "
            "Please install required packages, e.g. `pip install torch numpy matplotlib`."
        ) from exc


if __name__ == "__main__":
    main()
