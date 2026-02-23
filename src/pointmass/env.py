from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


Observation = Dict[str, np.ndarray]


@dataclass
class Point2DConfig:
    bounds_x: Tuple[float, float] = (-1.0, 1.0)
    bounds_y: Tuple[float, float] = (-1.0, 1.0)
    physics_substeps: int = 10
    success_radius: float = 0.15
    goal_border_ratio: float = 0.05
    render_height_inches: float = 5.0
    dpi: int = 200


class Point2D:
    def __init__(self, cfg: Point2DConfig):
        self.cfg = cfg
        self._cur_pos = np.zeros(2, dtype=np.float32)
        self._goal_pos = np.zeros(2, dtype=np.float32)
        self._cur_vel = np.zeros(2, dtype=np.float32)
        self._cur_episode_traj = []

    def sample_goal(self) -> np.ndarray:
        bx = self.cfg.bounds_x
        by = self.cfg.bounds_y
        border_x = (bx[1] - bx[0]) * self.cfg.goal_border_ratio
        border_y = (by[1] - by[0]) * self.cfg.goal_border_ratio
        goal_x = np.random.uniform(bx[0] + border_x, bx[1] - border_x)
        goal_y = np.random.uniform(by[0] + border_y, by[1] - border_y)
        return np.array([goal_x, goal_y], dtype=np.float32)

    def set_goal(self, goal_pos: np.ndarray) -> None:
        self._goal_pos = np.asarray(goal_pos, dtype=np.float32)

    def _make_obs(self) -> Observation:
        return {
            "cur_pos": self._cur_pos.copy(),
            "cur_vel": self._cur_vel.copy(),
            "goal_pos": self._goal_pos.copy(),
        }

    def reset(self) -> Observation:
        self._goal_pos = self.sample_goal()
        bx = self.cfg.bounds_x
        by = self.cfg.bounds_y
        cur_x = np.random.uniform(bx[0], bx[1])
        cur_y = np.random.uniform(by[0], by[1])
        self._cur_pos = np.array([cur_x, cur_y], dtype=np.float32)
        self._cur_vel = np.zeros(2, dtype=np.float32)
        self._cur_episode_traj = [self._cur_pos.copy()]
        return self._make_obs()

    def step(self, action: np.ndarray) -> Tuple[Observation, float, bool]:
        action = np.asarray(action, dtype=np.float32)
        for _ in range(self.cfg.physics_substeps):
            self._cur_vel += action
            self._cur_pos += self._cur_vel
        obs = self._make_obs()
        reward = -1.0 * float(np.linalg.norm(self._cur_pos - self._goal_pos))
        done = self.success()
        self._cur_episode_traj.append(self._cur_pos.copy())
        return obs, reward, done

    def success(self, waypoint: Optional[np.ndarray] = None) -> bool:
        goal = self._goal_pos if waypoint is None else np.asarray(waypoint, dtype=np.float32)
        return float(np.linalg.norm(self._cur_pos - goal)) < self.cfg.success_radius

    @property
    def cur_episode_traj(self) -> np.ndarray:
        if not self._cur_episode_traj:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(self._cur_episode_traj, dtype=np.float32)

    def render(
        self,
        title: str = "",
        points: Optional[np.ndarray] = None,
        goal_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig, ax = plt.subplots(
            figsize=(self.cfg.render_height_inches, self.cfg.render_height_inches),
            dpi=self.cfg.dpi,
        )
        ax.set_xlim(self.cfg.bounds_x[0], self.cfg.bounds_x[1])
        ax.set_ylim(self.cfg.bounds_y[0], self.cfg.bounds_y[1])
        ax.set_aspect("equal")

        if points is None:
            points = self.cur_episode_traj
            cur_pos = self._cur_pos
        else:
            cur_pos = points[-1]

        if goal_pos is None:
            goal_pos = self._goal_pos

        if points.shape[0] > 0:
            ax.plot(points[:, 0], points[:, 1], marker=".", color="blue", markersize=8, linewidth=2)
        ax.scatter(goal_pos[0], goal_pos[1], marker="*", s=180, color="orange")
        ax.scatter(cur_pos[0], cur_pos[1], marker="o", s=90, color="red")

        circle = patches.Circle(
            (goal_pos[0], goal_pos[1]),
            self.cfg.success_radius,
            edgecolor="green",
            linestyle="--",
            linewidth=2,
            fill=False,
        )
        ax.add_patch(circle)

        if title:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(int(height), int(width), 3)
        plt.close(fig)
        return image


def pd_controller(cur_pos: np.ndarray, cur_vel: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    kp = 0.0002
    kd = 0.0125
    action = kp * (goal_pos - cur_pos) + kd * (-1.0 * cur_vel)
    return action.astype(np.float32)
