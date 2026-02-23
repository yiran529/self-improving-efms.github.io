from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical, Independent, Normal
from torch.utils.data import DataLoader, TensorDataset

from .env import Point2D, Point2DConfig, pd_controller
from .model import TimerNet
from .utils import ensure_dirs, resolve_device, seed_everything


def build_env_from_cfg(cfg: Mapping[str, Any]) -> Point2D:
    env_cfg = Point2DConfig(
        bounds_x=tuple(cfg["env"]["bounds_x"]),
        bounds_y=tuple(cfg["env"]["bounds_y"]),
        physics_substeps=int(cfg["env"]["physics_substeps"]),
        success_radius=float(cfg["env"]["success_radius"]),
    )
    return Point2D(env_cfg)


@dataclass
class NormalizationStats:
    cur_pos_mean: np.ndarray
    cur_pos_std: np.ndarray
    cur_vel_mean: np.ndarray
    cur_vel_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @classmethod
    def from_tuples(cls, tuples: Mapping[str, np.ndarray], eps: float = 1e-6) -> "NormalizationStats":
        cur_pos_mean = np.mean(tuples["cur_pos"], axis=0, keepdims=True).astype(np.float32)
        cur_pos_std = np.std(tuples["cur_pos"], axis=0, keepdims=True).astype(np.float32)
        cur_vel_mean = np.mean(tuples["cur_vel"], axis=0, keepdims=True).astype(np.float32)
        cur_vel_std = np.std(tuples["cur_vel"], axis=0, keepdims=True).astype(np.float32)
        act_mean = np.mean(tuples["action"], axis=0, keepdims=True).astype(np.float32)
        act_std = np.std(tuples["action"], axis=0, keepdims=True).astype(np.float32)
        cur_pos_std = np.maximum(cur_pos_std, eps)
        cur_vel_std = np.maximum(cur_vel_std, eps)
        act_std = np.maximum(act_std, eps)
        return cls(
            cur_pos_mean=cur_pos_mean,
            cur_pos_std=cur_pos_std,
            cur_vel_mean=cur_vel_mean,
            cur_vel_std=cur_vel_std,
            act_mean=act_mean,
            act_std=act_std,
        )

    def normalize_obs_np(self, obs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "cur_pos": ((obs["cur_pos"] - self.cur_pos_mean) / self.cur_pos_std).astype(np.float32),
            "cur_vel": ((obs["cur_vel"] - self.cur_vel_mean) / self.cur_vel_std).astype(np.float32),
            "goal_pos": ((obs["goal_pos"] - self.cur_pos_mean) / self.cur_pos_std).astype(np.float32),
        }

    def normalize_action_np(self, action: np.ndarray) -> np.ndarray:
        return ((action - self.act_mean) / self.act_std).astype(np.float32)

    def unnormalize_action_np(self, action: np.ndarray) -> np.ndarray:
        return (action * self.act_std + self.act_mean).astype(np.float32)

    def normalize_obs_single_to_vector(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        n_obs = self.normalize_obs_np(
            {
                "cur_pos": obs["cur_pos"][None, :],
                "cur_vel": obs["cur_vel"][None, :],
                "goal_pos": obs["goal_pos"][None, :],
            }
        )
        return np.concatenate([n_obs["cur_pos"], n_obs["cur_vel"], n_obs["goal_pos"]], axis=-1)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cur_pos_mean": self.cur_pos_mean,
            "cur_pos_std": self.cur_pos_std,
            "cur_vel_mean": self.cur_vel_mean,
            "cur_vel_std": self.cur_vel_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NormalizationStats":
        return cls(
            cur_pos_mean=np.asarray(d["cur_pos_mean"], dtype=np.float32),
            cur_pos_std=np.asarray(d["cur_pos_std"], dtype=np.float32),
            cur_vel_mean=np.asarray(d["cur_vel_mean"], dtype=np.float32),
            cur_vel_std=np.asarray(d["cur_vel_std"], dtype=np.float32),
            act_mean=np.asarray(d["act_mean"], dtype=np.float32),
            act_std=np.asarray(d["act_std"], dtype=np.float32),
        )


@dataclass
class DiscreteDistanceConverter:
    min_distance: float
    max_distance: float
    num_bins: int

    @property
    def bin_size(self) -> float:
        return (self.max_distance - self.min_distance) / float(self.num_bins)

    def distance_to_index(self, distance: Any) -> Any:
        max_val = self.max_distance - self.bin_size / 2.0
        if torch.is_tensor(distance):
            d = torch.clamp(distance, min=self.min_distance, max=max_val)
            idx = torch.floor((d - self.min_distance) / self.bin_size).long()
            return idx
        d_np = np.asarray(distance, dtype=np.float32)
        d_np = np.clip(d_np, self.min_distance, max_val)
        idx = np.floor((d_np - self.min_distance) / self.bin_size).astype(np.int64)
        return idx

    def logits_to_expected_distance(self, logits: Any) -> Any:
        if torch.is_tensor(logits):
            probs = torch.softmax(logits, dim=-1)
            vals = torch.linspace(
                self.min_distance,
                self.max_distance,
                self.num_bins + 1,
                device=logits.device,
                dtype=logits.dtype,
            )[:-1]
            return torch.sum(probs * vals, dim=-1)

        logits_np = np.asarray(logits, dtype=np.float32)
        logits_np = logits_np - np.max(logits_np, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_np)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        vals = np.linspace(self.min_distance, self.max_distance, self.num_bins + 1, endpoint=True, dtype=np.float32)[:-1]
        return np.sum(probs * vals, axis=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_distance": float(self.min_distance),
            "max_distance": float(self.max_distance),
            "num_bins": int(self.num_bins),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DiscreteDistanceConverter":
        return cls(
            min_distance=float(d["min_distance"]),
            max_distance=float(d["max_distance"]),
            num_bins=int(d["num_bins"]),
        )


def _stack_episode(transitions: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = transitions[0].keys()
    return {k: np.stack([x[k] for x in transitions], axis=0).astype(np.float32) for k in keys}


def _concat_episodes(episodes: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = episodes[0].keys()
    return {k: np.concatenate([ep[k] for ep in episodes], axis=0).astype(np.float32) for k in keys}


def generate_pd_dataset(
    env: Point2D,
    num_episodes: int,
    num_waypoints_per_episode: int,
    episode_len_discard_thresh: int,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, np.ndarray], np.ndarray]:
    episodes: List[Dict[str, np.ndarray]] = []

    while len(episodes) < num_episodes:
        traj: List[Dict[str, np.ndarray]] = []
        cur_obs = env.reset()
        succ = env.success()

        waypoint_idx = 0
        if num_waypoints_per_episode == 0:
            cur_waypoint = cur_obs["goal_pos"]
        else:
            cur_waypoint = env.sample_goal()
        waypoint_succ = env.success(waypoint=cur_waypoint)

        reward = 0.0
        while not succ:
            if waypoint_succ:
                waypoint_idx += 1
                waypoint_idx = min(waypoint_idx, num_waypoints_per_episode)
                if waypoint_idx == num_waypoints_per_episode:
                    cur_waypoint = cur_obs["goal_pos"]
                else:
                    cur_waypoint = env.sample_goal()

            action = pd_controller(cur_obs["cur_pos"], cur_obs["cur_vel"], cur_waypoint)
            next_obs, reward, _ = env.step(action)
            traj.append(
                {
                    "cur_pos": cur_obs["cur_pos"],
                    "cur_vel": cur_obs["cur_vel"],
                    "goal_pos": cur_obs["goal_pos"],
                    "action": action,
                    "time_to_success": np.array(0.0, dtype=np.float32),
                    "reward": np.array(reward, dtype=np.float32),
                    "discount": np.array(1.0, dtype=np.float32),
                    "next_cur_pos": next_obs["cur_pos"],
                    "next_cur_vel": next_obs["cur_vel"],
                    "next_goal_pos": next_obs["goal_pos"],
                }
            )

            cur_obs = next_obs
            succ = env.success()
            waypoint_succ = env.success(waypoint=cur_waypoint)

        action = pd_controller(cur_obs["cur_pos"], cur_obs["cur_vel"], cur_waypoint)
        traj.append(
            {
                "cur_pos": cur_obs["cur_pos"],
                "cur_vel": cur_obs["cur_vel"],
                "goal_pos": cur_obs["goal_pos"],
                "action": action,
                "time_to_success": np.array(0.0, dtype=np.float32),
                "reward": np.array(reward, dtype=np.float32),
                "discount": np.array(0.0, dtype=np.float32),
                "next_cur_pos": cur_obs["cur_pos"],
                "next_cur_vel": cur_obs["cur_vel"],
                "next_goal_pos": cur_obs["goal_pos"],
            }
        )

        traj_len = len(traj)
        if traj_len < episode_len_discard_thresh:
            continue

        episode = _stack_episode(traj)
        episode["time_to_success"] = np.arange(traj_len - 1, -1, -1, dtype=np.float32)
        episodes.append(episode)

    all_tuples = _concat_episodes(episodes)
    episode_lens = np.array([ep["cur_pos"].shape[0] for ep in episodes], dtype=np.int32)
    return episodes, all_tuples, episode_lens


def save_dataset(path: Path, episodes: List[Dict[str, np.ndarray]], tuples: Dict[str, np.ndarray], episode_lens: np.ndarray) -> None:
    payload = {"episodes": episodes, "tuples": tuples, "episode_lens": episode_lens}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _build_sft_arrays(
    tuples: Mapping[str, np.ndarray],
    norm_stats: NormalizationStats,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = {
        "cur_pos": tuples["cur_pos"],
        "cur_vel": tuples["cur_vel"],
        "goal_pos": tuples["goal_pos"],
    }
    n_obs = norm_stats.normalize_obs_np(obs)
    obs_vec = np.concatenate([n_obs["cur_pos"], n_obs["cur_vel"], n_obs["goal_pos"]], axis=-1).astype(np.float32)
    act_vec = norm_stats.normalize_action_np(tuples["action"]).astype(np.float32)
    tts = tuples["time_to_success"].astype(np.float32)
    return obs_vec, act_vec, tts


def _make_train_val_loaders(
    obs_vec: np.ndarray,
    act_vec: np.ndarray,
    tts: np.ndarray,
    batch_size: int,
    train_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    n = obs_vec.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_size = int(n * train_ratio)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    if val_idx.size == 0:
        val_idx = train_idx[: min(1024, train_idx.size)]

    train_data = TensorDataset(
        torch.from_numpy(obs_vec[train_idx]),
        torch.from_numpy(act_vec[train_idx]),
        torch.from_numpy(tts[train_idx]),
    )
    val_data = TensorDataset(
        torch.from_numpy(obs_vec[val_idx]),
        torch.from_numpy(act_vec[val_idx]),
        torch.from_numpy(tts[val_idx]),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _infinite(loader: DataLoader) -> Iterable[Tuple[torch.Tensor, ...]]:
    while True:
        for batch in loader:
            yield batch


def _compute_sft_metrics(
    model: TimerNet,
    obs: torch.Tensor,
    action: torch.Tensor,
    tts: torch.Tensor,
    converter: DiscreteDistanceConverter,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loc, scale, logits = model(obs)
    act_dist = Independent(Normal(loc, scale), 1)
    dist_dist = Categorical(logits=logits)

    act_log_prob = act_dist.log_prob(action).mean()
    dist_idx = converter.distance_to_index(tts)
    dist_log_prob = dist_dist.log_prob(dist_idx).mean()
    loss = -1.0 * (act_log_prob + dist_log_prob)

    metrics = {
        "pretrain_loss": loss.detach(),
        "act_log_prob": act_log_prob.detach(),
        "dist_log_prob": dist_log_prob.detach(),
    }
    return loss, metrics


@torch.no_grad()
def _evaluate_sft(
    model: TimerNet,
    loader: DataLoader,
    converter: DiscreteDistanceConverter,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = {"pretrain_loss": 0.0, "act_log_prob": 0.0, "dist_log_prob": 0.0}
    n_batches = 0
    for obs, action, tts in loader:
        obs = obs.to(device)
        action = action.to(device)
        tts = tts.to(device)
        _, metrics = _compute_sft_metrics(model, obs, action, tts, converter)
        for k in total:
            total[k] += float(metrics[k].item())
        n_batches += 1
    if n_batches == 0:
        return {k: float("nan") for k in total}
    return {k: v / n_batches for k, v in total.items()}


def save_checkpoint(
    path: Path,
    model: TimerNet,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: Mapping[str, Any],
    norm_stats: NormalizationStats,
    distance_converter: DiscreteDistanceConverter,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": dict(cfg),
        "normalization": norm_stats.to_dict(),
        "distance": distance_converter.to_dict(),
    }
    if extra is not None:
        payload["extra"] = dict(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: Optional[TimerNet] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    # Temporary compatibility fix for PyTorch>=2.6 default change:
    # keep loading full checkpoint payload (weights + numpy metadata).
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# DEBUG: utility to fail fast when numpy arrays contain NaN/Inf during stage2 diagnostics.
def _assert_finite_np(name: str, arr: np.ndarray) -> None:
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return
    bad_count = int((~finite_mask).sum())
    total_count = int(arr.size)
    raise ValueError(
        f"[DEBUG] Non-finite numpy array detected: {name}, "
        f"bad={bad_count}/{total_count}, shape={arr.shape}, dtype={arr.dtype}, "
        f"min={np.nanmin(arr)}, max={np.nanmax(arr)}"
    )


# DEBUG: utility to fail fast when torch tensors contain NaN/Inf during stage2 diagnostics.
def _assert_finite_torch(name: str, tensor: torch.Tensor) -> None:
    finite_mask = torch.isfinite(tensor)
    if bool(torch.all(finite_mask)):
        return
    bad_count = int((~finite_mask).sum().item())
    total_count = int(tensor.numel())
    t_cpu = tensor.detach().float().cpu()
    raise ValueError(
        f"[DEBUG] Non-finite tensor detected: {name}, "
        f"bad={bad_count}/{total_count}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={torch.nan_to_num(t_cpu, nan=0.0, posinf=0.0, neginf=0.0).min().item()}, "
        f"max={torch.nan_to_num(t_cpu, nan=0.0, posinf=0.0, neginf=0.0).max().item()}"
    )


# DEBUG: utility to fail fast when model parameters or gradients become NaN/Inf.
def _assert_model_finite(model: TimerNet, tag: str) -> None:
    for name, param in model.named_parameters():
        _assert_finite_torch(f"{tag}:param:{name}", param.data)
        if param.grad is not None:
            _assert_finite_torch(f"{tag}:grad:{name}", param.grad)


def run_sft_training(
    cfg: Mapping[str, Any],
    dataset_path: Path,
    output_ckpt_path: Path,
    resume_ckpt_path: Optional[Path] = None,
) -> Path:
    ensure_dirs(cfg["paths"])
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    payload = load_dataset(dataset_path)
    tuples = payload["tuples"]

    norm_stats = NormalizationStats.from_tuples(tuples)
    converter = DiscreteDistanceConverter(
        min_distance=float(cfg["distance"]["min_distance"]),
        max_distance=float(cfg["distance"]["max_distance"]),
        num_bins=int(cfg["distance"]["num_bins"]),
    )

    obs_vec, act_vec, tts = _build_sft_arrays(tuples, norm_stats)
    train_loader, val_loader = _make_train_val_loaders(
        obs_vec=obs_vec,
        act_vec=act_vec,
        tts=tts,
        batch_size=int(cfg["stage1"]["batch_size"]),
        train_ratio=float(cfg["stage1"]["train_ratio"]),
        seed=int(cfg["seed"]),
    )

    model = TimerNet(
        obs_dim=6,
        act_dim=2,
        hidden_sizes=tuple(cfg["model"]["hidden_sizes"]),
        num_distance_bins=int(cfg["distance"]["num_bins"]),
        min_act_scale=float(cfg["model"]["min_act_scale"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["stage1"]["learning_rate"]),
        weight_decay=float(cfg["stage1"]["weight_decay"]),
        eps=1e-7,
    )

    if resume_ckpt_path is not None and resume_ckpt_path.exists():
        load_checkpoint(resume_ckpt_path, model=model, optimizer=optimizer, map_location=str(device))

    num_updates = int(cfg["stage1"]["num_updates"])
    eval_interval = int(cfg["stage1"]["eval_interval"])
    train_iter = _infinite(train_loader)

    print(f"[Stage1] device={device}, updates={num_updates}, batch_size={cfg['stage1']['batch_size']}")
    for step in range(1, num_updates + 1):
        model.train()
        obs, action, tts_batch = next(train_iter)
        obs = obs.to(device)
        action = action.to(device)
        tts_batch = tts_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _compute_sft_metrics(model, obs, action, tts_batch, converter)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == num_updates:
            val_metrics = _evaluate_sft(model, val_loader, converter, device)
            print(
                f"[Stage1][{step:05d}] "
                f"train_loss={metrics['pretrain_loss'].item():.4f} "
                f"train_act_lp={metrics['act_log_prob'].item():.4f} "
                f"train_dist_lp={metrics['dist_log_prob'].item():.4f} "
                f"val_loss={val_metrics['pretrain_loss']:.4f}"
            )

    save_checkpoint(
        output_ckpt_path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        norm_stats=norm_stats,
        distance_converter=converter,
        extra={"stage": "sft", "num_updates": num_updates},
    )
    print(f"[Stage1] checkpoint saved to: {output_ckpt_path}")
    return output_ckpt_path


def _predict_action_and_distance(
    model: TimerNet,
    obs_vec_norm: np.ndarray,
    converter: DiscreteDistanceConverter,
    device: torch.device,
    deterministic: bool = False,
) -> Tuple[np.ndarray, float]:
    # DEBUG: assert normalized observation is finite before model forward.
    _assert_finite_np("rollout_obs_vec_norm", obs_vec_norm)
    obs_t = torch.from_numpy(obs_vec_norm[None, :]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        loc, scale, logits = model(obs_t)
        # DEBUG: assert forward outputs are finite to localize NaN source in rollout.
        _assert_finite_torch("rollout_loc", loc)
        _assert_finite_torch("rollout_scale", scale)
        _assert_finite_torch("rollout_logits", logits)
        act_dist = Independent(Normal(loc, scale), 1)
        norm_action = loc if deterministic else act_dist.sample()
        # DEBUG: assert sampled normalized action is finite before environment step.
        _assert_finite_torch("rollout_norm_action", norm_action)
        pred_dist = converter.logits_to_expected_distance(logits).item()
    # DEBUG: assert predicted distance is finite before reward-shaping computations.
    if not np.isfinite(pred_dist):
        raise ValueError(f"[DEBUG] Non-finite predicted distance detected: {pred_dist}")
    action = norm_action[0].detach().cpu().numpy().astype(np.float32)
    # DEBUG: assert output action vector is finite.
    _assert_finite_np("rollout_action_np", action)
    return action, float(pred_dist)


def generate_reinforce_dataset(
    env: Point2D,
    model: TimerNet,
    norm_stats: NormalizationStats,
    converter: DiscreteDistanceConverter,
    num_steps: int,
    gamma: float,
    max_steps: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    total_steps = 0
    invalid_episode_count = 0
    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    episode_stats: List[Dict[str, float]] = []

    while total_steps < num_steps:
        traj_obs: List[np.ndarray] = []
        traj_act: List[np.ndarray] = []
        traj_dist_pred: List[float] = []

        obs = env.reset()
        obs_vec = norm_stats.normalize_obs_single_to_vector(obs)
        # DEBUG: assert normalized initial observation is finite for each rollout.
        _assert_finite_np("episode_obs_vec_init", obs_vec)
        traj_obs.append(obs_vec)

        episode_return = 0.0
        steps = 0
        done = env.success()
        episode_valid = True

        while (not done) and steps < max_steps:
            norm_action, pred_dist = _predict_action_and_distance(
                model=model,
                obs_vec_norm=obs_vec,
                converter=converter,
                device=device,
                deterministic=False,
            )
            action = norm_stats.unnormalize_action_np(norm_action[None, :])[0]
            # DEBUG: assert unnormalized action is finite before stepping environment.
            _assert_finite_np("episode_action_unnorm", action)
            traj_act.append(norm_action)
            traj_dist_pred.append(pred_dist)

            obs, reward, done = env.step(action)
            # DEBUG: discard rollout episode when environment transition becomes non-finite.
            if (
                (not np.isfinite(reward))
                or (not np.isfinite(obs["cur_pos"]).all())
                or (not np.isfinite(obs["cur_vel"]).all())
                or (not np.isfinite(obs["goal_pos"]).all())
            ):
                invalid_episode_count += 1
                print(
                    "[DEBUG] Discarding episode due to non-finite env transition: "
                    f"reward={reward}, step={steps}, invalid_episodes={invalid_episode_count}"
                )
                episode_valid = False
                break
            episode_return += float(reward)
            steps += 1
            obs_vec = norm_stats.normalize_obs_single_to_vector(obs)
            # DEBUG: assert normalized observation is finite at each step.
            _assert_finite_np("episode_obs_vec_step", obs_vec)
            traj_obs.append(obs_vec)

        # DEBUG: skip invalid episodes to prevent non-finite rollout data entering training.
        if not episode_valid:
            if invalid_episode_count >= 1000:
                raise RuntimeError(
                    "[DEBUG] Too many invalid rollout episodes (>=1000). "
                    "Stage2 policy likely diverged; reduce stage2 LR or restart from earlier checkpoint."
                )
            continue

        if steps < 1:
            continue

        _, final_dist = _predict_action_and_distance(
            model=model,
            obs_vec_norm=obs_vec,
            converter=converter,
            device=device,
            deterministic=False,
        )
        # DEBUG: skip episode when final distance prediction is non-finite.
        if not np.isfinite(final_dist):
            invalid_episode_count += 1
            print(
                "[DEBUG] Discarding episode due to non-finite final distance prediction: "
                f"final_dist={final_dist}, invalid_episodes={invalid_episode_count}"
            )
            if invalid_episode_count >= 1000:
                raise RuntimeError(
                    "[DEBUG] Too many invalid rollout episodes (>=1000). "
                    "Stage2 policy likely diverged; reduce stage2 LR or restart from earlier checkpoint."
                )
            continue
        traj_dist_pred.append(final_dist)

        traj_obs_arr = np.stack(traj_obs, axis=0).astype(np.float32)
        traj_act_arr = np.stack(traj_act, axis=0).astype(np.float32)
        traj_dist = np.asarray(traj_dist_pred, dtype=np.float32)
        # DEBUG: assert trajectory arrays are finite before computing weights.
        _assert_finite_np("traj_obs_arr", traj_obs_arr)
        _assert_finite_np("traj_act_arr", traj_act_arr)
        _assert_finite_np("traj_dist", traj_dist)

        rews = -1.0 * (traj_dist[1:] - traj_dist[:-1])
        discounted = np.zeros_like(rews, dtype=np.float32)
        running = 0.0
        for i in range(rews.shape[0] - 1, -1, -1):
            running = rews[i] + float(gamma) * running
            discounted[i] = running
        # DEBUG: assert REINFORCE weights are finite before batching.
        _assert_finite_np("traj_discounted_weights", discounted)

        all_obs.append(traj_obs_arr[:-1])
        all_actions.append(traj_act_arr)
        all_weights.append(discounted)

        total_steps += traj_act_arr.shape[0]
        episode_stats.append(
            {
                "success": float(done),
                "return": float(episode_return),
                "len": float(steps),
            }
        )
        invalid_episode_count = 0

    obs_arr = np.concatenate(all_obs, axis=0)[:num_steps]
    act_arr = np.concatenate(all_actions, axis=0)[:num_steps]
    weight_arr = np.concatenate(all_weights, axis=0)[:num_steps]
    # DEBUG: filter non-finite rows in reinforce dataset to keep optimizer inputs valid.
    finite_mask = (
        np.isfinite(obs_arr).all(axis=1)
        & np.isfinite(act_arr).all(axis=1)
        & np.isfinite(weight_arr)
    )
    dropped = int((~finite_mask).sum())
    if dropped > 0:
        print(f"[DEBUG] Dropping non-finite reinforce samples: dropped={dropped}, total={finite_mask.shape[0]}")
    obs_arr = obs_arr[finite_mask]
    act_arr = act_arr[finite_mask]
    weight_arr = weight_arr[finite_mask]
    # DEBUG: assert there are valid reinforce samples left after filtering.
    if obs_arr.shape[0] == 0:
        raise ValueError("[DEBUG] All REINFORCE samples are non-finite after filtering.")

    if len(episode_stats) == 0:
        raise RuntimeError("No REINFORCE trajectories generated. Check environment and policy.")

    stat_keys = episode_stats[0].keys()
    stats = {k: np.asarray([x[k] for x in episode_stats], dtype=np.float32) for k in stat_keys}
    data = {"obs": obs_arr.astype(np.float32), "action": act_arr.astype(np.float32), "weight": weight_arr.astype(np.float32)}
    return data, stats


def _compute_reinforce_loss(
    model: TimerNet,
    obs: torch.Tensor,
    action: torch.Tensor,
    weight: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    # DEBUG: assert reinforce minibatch inputs are finite before forward.
    _assert_finite_torch("reinforce_obs_batch", obs)
    _assert_finite_torch("reinforce_action_batch", action)
    _assert_finite_torch("reinforce_weight_batch", weight)
    loc, scale, _ = model(obs)
    # DEBUG: assert reinforce forward outputs are finite before constructing distributions.
    _assert_finite_torch("reinforce_loc", loc)
    _assert_finite_torch("reinforce_scale", scale)
    act_dist = Independent(Normal(loc, scale), 1)
    log_prob = act_dist.log_prob(action)
    # DEBUG: assert log-probabilities are finite before loss reduction.
    _assert_finite_torch("reinforce_log_prob", log_prob)
    scaled_weight = weight / float(max_steps)
    # DEBUG: assert scaled reinforce weights are finite.
    _assert_finite_torch("reinforce_scaled_weight", scaled_weight)
    loss = -1.0 * torch.mean(scaled_weight * log_prob)
    # DEBUG: assert scalar loss is finite before backward.
    _assert_finite_torch("reinforce_loss", loss)
    return loss


def run_self_improve_training(
    cfg: Mapping[str, Any],
    init_ckpt_path: Path,
    output_ckpt_path: Path,
) -> Path:
    ensure_dirs(cfg["paths"])
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    env = build_env_from_cfg(cfg)

    converter = DiscreteDistanceConverter(
        min_distance=float(cfg["distance"]["min_distance"]),
        max_distance=float(cfg["distance"]["max_distance"]),
        num_bins=int(cfg["distance"]["num_bins"]),
    )
    model = TimerNet(
        obs_dim=6,
        act_dim=2,
        hidden_sizes=tuple(cfg["model"]["hidden_sizes"]),
        num_distance_bins=int(cfg["distance"]["num_bins"]),
        min_act_scale=float(cfg["model"]["min_act_scale"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["stage2"]["learning_rate"]),
        weight_decay=float(cfg["stage2"]["weight_decay"]),
        eps=1e-7,
    )

    init_ckpt = load_checkpoint(init_ckpt_path, model=model, optimizer=optimizer, map_location=str(device))
    norm_stats = NormalizationStats.from_dict(init_ckpt["normalization"])

    num_iterations = int(cfg["stage2"]["num_iterations"])
    env_steps_per_batch = int(cfg["stage2"]["env_steps_per_batch"])
    minibatch_size = int(cfg["stage2"]["minibatch_size"])
    gamma = float(cfg["stage2"]["gamma"])
    max_steps = int(cfg["eval"]["max_steps"])

    print(f"[Stage2] device={device}, iterations={num_iterations}, env_steps_per_batch={env_steps_per_batch}")
    for it in range(1, num_iterations + 1):
        # DEBUG: assert model parameters are finite before each rollout-data generation.
        _assert_model_finite(model, f"iter{it:04d}:before_rollout")
        reinforce_data, ep_stats = generate_reinforce_dataset(
            env=env,
            model=model,
            norm_stats=norm_stats,
            converter=converter,
            num_steps=env_steps_per_batch,
            gamma=gamma,
            max_steps=max_steps,
            device=device,
        )

        ds = TensorDataset(
            torch.from_numpy(reinforce_data["obs"]),
            torch.from_numpy(reinforce_data["action"]),
            torch.from_numpy(reinforce_data["weight"]),
        )
        loader = DataLoader(ds, batch_size=minibatch_size, shuffle=True, drop_last=True)

        model.train()
        losses = []
        for obs, action, weight in loader:
            obs = obs.to(device)
            action = action.to(device)
            weight = weight.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = _compute_reinforce_loss(model, obs, action, weight, max_steps=max_steps)
            loss.backward()
            # DEBUG: assert gradients are finite immediately after backward.
            _assert_model_finite(model, f"iter{it:04d}:after_backward")
            optimizer.step()
            # DEBUG: assert parameters are finite immediately after optimizer step.
            _assert_model_finite(model, f"iter{it:04d}:after_step")
            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        success_rate = float(np.mean(ep_stats["success"]))
        ret_mean = float(np.mean(ep_stats["return"]))
        ret_std = float(np.std(ep_stats["return"]))
        len_mean = float(np.mean(ep_stats["len"]))
        len_std = float(np.std(ep_stats["len"]))
        print(
            f"[Stage2][{it:04d}] "
            f"loss={mean_loss:.4f} success={success_rate:.3f} "
            f"return={ret_mean:.2f}+/-{ret_std:.2f} "
            f"len={len_mean:.2f}+/-{len_std:.2f}"
        )

    save_checkpoint(
        output_ckpt_path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        norm_stats=norm_stats,
        distance_converter=converter,
        extra={"stage": "self_improve", "num_iterations": num_iterations},
    )
    print(f"[Stage2] checkpoint saved to: {output_ckpt_path}")
    return output_ckpt_path


@torch.no_grad()
def evaluate_policy(
    cfg: Mapping[str, Any],
    ckpt_path: Path,
    num_episodes: int,
    deterministic_action: bool = False,
) -> Dict[str, float]:
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    env = build_env_from_cfg(cfg)

    model = TimerNet(
        obs_dim=6,
        act_dim=2,
        hidden_sizes=tuple(cfg["model"]["hidden_sizes"]),
        num_distance_bins=int(cfg["distance"]["num_bins"]),
        min_act_scale=float(cfg["model"]["min_act_scale"]),
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, model=model, optimizer=None, map_location=str(device))
    model.eval()

    norm_stats = NormalizationStats.from_dict(ckpt["normalization"])
    converter = DiscreteDistanceConverter.from_dict(ckpt["distance"])
    max_steps = int(cfg["eval"]["max_steps"])

    stats = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = env.success()
        episode_return = 0.0
        steps = 0

        while (not done) and steps < max_steps:
            obs_vec = norm_stats.normalize_obs_single_to_vector(obs)
            norm_action, _ = _predict_action_and_distance(
                model=model,
                obs_vec_norm=obs_vec,
                converter=converter,
                device=device,
                deterministic=deterministic_action,
            )
            action = norm_stats.unnormalize_action_np(norm_action[None, :])[0]
            obs, reward, done = env.step(action)
            episode_return += float(reward)
            steps += 1

        stats.append({"success": float(done), "return": episode_return, "len": float(steps)})

    success = float(np.mean([x["success"] for x in stats]))
    returns = np.asarray([x["return"] for x in stats], dtype=np.float32)
    lengths = np.asarray([x["len"] for x in stats], dtype=np.float32)
    result = {
        "success_rate": success,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "len_mean": float(np.mean(lengths)),
        "len_std": float(np.std(lengths)),
        "len_min": float(np.min(lengths)),
        "len_max": float(np.max(lengths)),
    }
    return result


def _write_video_file(
    frames: Sequence[np.ndarray],
    output_video_path: Path,
    fps: int,
) -> Path:
    import imageio.v2 as imageio

    if len(frames) == 0:
        raise ValueError("No frames to write.")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    uint8_frames = [frame.astype(np.uint8) for frame in frames]
    try:
        with imageio.get_writer(str(output_video_path), fps=fps) as writer:
            for frame in uint8_frames:
                writer.append_data(frame)
        return output_video_path
    except Exception as exc:
        # Fallback for environments where imageio_ffmpeg cannot locate a valid ffmpeg binary.
        fallback_path = output_video_path.with_suffix(".gif")
        print(
            "[Visualize] mp4 writer failed; fallback to gif. "
            f"reason={type(exc).__name__}: {exc}. output={fallback_path}"
        )
        imageio.mimsave(str(fallback_path), uint8_frames, duration=max(1.0 / float(fps), 1e-3), loop=0)
        return fallback_path


def generate_dataset_trajectory_video(
    cfg: Mapping[str, Any],
    dataset_path: Path,
    output_video_path: Path,
    num_episodes: int,
    fps: int,
) -> Path:
    seed_everything(int(cfg["seed"]))
    payload = load_dataset(dataset_path)
    episodes = payload["episodes"]
    if len(episodes) == 0:
        raise ValueError("Dataset is empty. Run generate-data first.")

    env = build_env_from_cfg(cfg)
    episode_count = min(num_episodes, len(episodes))
    frames: List[np.ndarray] = []
    for ep_idx in range(episode_count):
        episode = episodes[ep_idx]
        points = np.asarray(episode["cur_pos"], dtype=np.float32)
        goal_pos = np.asarray(episode["goal_pos"][0], dtype=np.float32)
        for t in range(points.shape[0]):
            frames.append(env.render(title="Dataset Trajectory", points=points[: t + 1], goal_pos=goal_pos))

    video_path = _write_video_file(frames=frames, output_video_path=output_video_path, fps=fps)
    print(f"[Visualize][Dataset] saved to: {video_path}")
    return video_path


def _load_policy_from_checkpoint(
    cfg: Mapping[str, Any],
    ckpt_path: Path,
):
    device = resolve_device(cfg.get("device", "auto"))
    model = TimerNet(
        obs_dim=6,
        act_dim=2,
        hidden_sizes=tuple(cfg["model"]["hidden_sizes"]),
        num_distance_bins=int(cfg["distance"]["num_bins"]),
        min_act_scale=float(cfg["model"]["min_act_scale"]),
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, model=model, optimizer=None, map_location=str(device))
    model.eval()
    norm_stats = NormalizationStats.from_dict(ckpt["normalization"])
    converter = DiscreteDistanceConverter.from_dict(ckpt["distance"])
    return model, norm_stats, converter, device


def _generate_policy_trajectory_frames(
    cfg: Mapping[str, Any],
    ckpt_path: Path,
    num_episodes: int,
    max_steps: int,
    deterministic_action: bool,
    title: str,
) -> List[np.ndarray]:
    seed_everything(int(cfg["seed"]))
    env = build_env_from_cfg(cfg)
    model, norm_stats, converter, device = _load_policy_from_checkpoint(cfg=cfg, ckpt_path=ckpt_path)

    frames: List[np.ndarray] = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = env.success()
        steps = 0
        frames.append(env.render(title=title))
        while (not done) and steps < max_steps:
            obs_vec = norm_stats.normalize_obs_single_to_vector(obs)
            norm_action, _ = _predict_action_and_distance(
                model=model,
                obs_vec_norm=obs_vec,
                converter=converter,
                device=device,
                deterministic=deterministic_action,
            )
            action = norm_stats.unnormalize_action_np(norm_action[None, :])[0]
            obs, _, done = env.step(action)
            frames.append(env.render(title=title))
            steps += 1
    return frames


def generate_stage1_trajectory_video(
    cfg: Mapping[str, Any],
    ckpt_path: Path,
    output_video_path: Path,
    num_episodes: int,
    fps: int,
    max_steps: int,
    deterministic_action: bool = False,
) -> Path:
    frames = _generate_policy_trajectory_frames(
        cfg=cfg,
        ckpt_path=ckpt_path,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic_action=deterministic_action,
        title="Stage1 SFT Policy",
    )
    video_path = _write_video_file(frames=frames, output_video_path=output_video_path, fps=fps)
    print(f"[Visualize][Stage1] saved to: {video_path}")
    return video_path


def generate_stage2_trajectory_video(
    cfg: Mapping[str, Any],
    ckpt_path: Path,
    output_video_path: Path,
    num_episodes: int,
    fps: int,
    max_steps: int,
    deterministic_action: bool = False,
) -> Path:
    frames = _generate_policy_trajectory_frames(
        cfg=cfg,
        ckpt_path=ckpt_path,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic_action=deterministic_action,
        title="Stage2 Self-Improvement Policy",
    )
    video_path = _write_video_file(frames=frames, output_video_path=output_video_path, fps=fps)
    print(f"[Visualize][Stage2] saved to: {video_path}")
    return video_path
