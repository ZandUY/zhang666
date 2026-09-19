"""实验四移动障碍物轨迹、统计图和代表性 GIF。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from experiment4_simulator import Experiment4Result
from visualization import (
    UAV_COLORS,
    _configure_3d_axes,
    _draw_quadcopter,
)


METHODS = ("current_position", "predicted_position")
METHOD_LABELS = {
    "current_position": "Current position APF",
    "predicted_position": "Predicted position APF",
}
METHOD_COLORS = {
    "current_position": "tab:blue",
    "predicted_position": "tab:green",
}


def _draw_moving_sphere(
    axis: plt.Axes,
    center: np.ndarray,
    radius: float,
    alpha: float = 0.34,
) -> None:
    """绘制橙色半透明移动球体。"""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 24)
    polar = np.linspace(0.0, np.pi, 16)
    unit_x = np.outer(np.cos(azimuth), np.sin(polar))
    unit_y = np.outer(np.sin(azimuth), np.sin(polar))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))
    axis.plot_surface(
        center[0] + radius * unit_x,
        center[1] + radius * unit_y,
        center[2] + radius * unit_z,
        color="orange",
        edgecolor="darkorange",
        linewidth=0.2,
        alpha=alpha,
        shade=True,
    )


def plot_dynamic_trajectory(
    result: Experiment4Result, output_path: Path
) -> None:
    """绘制 UAV 轨迹、移动障碍物轨迹及最近接近时的球体。"""
    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection="3d")
    for uav_index in range(result.positions.shape[1]):
        trajectory = result.positions[:, uav_index]
        label = "Leader UAV 0" if uav_index == 0 else f"Follower UAV {uav_index}"
        axis.plot(
            *trajectory.T,
            color=UAV_COLORS[uav_index],
            linewidth=1.7,
            label=label,
        )
        axis.scatter(*trajectory[0], color=UAV_COLORS[uav_index], marker="o", s=24)
        axis.scatter(*trajectory[-1], color=UAV_COLORS[uav_index], marker="^", s=32)

    for obstacle_index, obstacle in enumerate(result.config.moving_obstacles):
        history = result.moving_obstacle_positions[:, obstacle_index]
        axis.plot(
            *history.T,
            color="darkorange",
            linestyle="--",
            linewidth=2.0,
            label="Moving obstacle path" if obstacle_index == 0 else None,
        )
        closest_frame = int(np.argmin(result.obstacle_min_clearances))
        _draw_moving_sphere(
            axis, history[closest_frame], obstacle.radius, alpha=0.38
        )
    axis.scatter(*result.config.goal, color="black", marker="*", s=120, label="Goal")
    _configure_3d_axes(axis, result)
    axis.set_title(
        f"{METHOD_LABELS[result.method_name]} | obstacle speed={result.obstacle_speed:g} m/s"
    )
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _grouped_bar(
    summaries: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    annotate_unrecovered: bool = False,
) -> None:
    """按移动速度分组绘制当前位置/预测位置柱状图。"""
    speeds = sorted({float(row["obstacle_speed_m_per_s"]) for row in summaries})
    x_locations = np.arange(len(speeds), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 5.7))
    for method_index, method in enumerate(METHODS):
        method_rows = {
            float(row["obstacle_speed_m_per_s"]): row
            for row in summaries
            if row["method"] == method
        }
        values = [float(method_rows[speed][metric]) for speed in speeds]
        positions = x_locations + (method_index - 0.5) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for bar, value, speed in zip(bars, values, speeds):
            text = f"{value:.2f}"
            if annotate_unrecovered:
                count = int(method_rows[speed]["unrecovered_count"])
                text += f"\nNR={count}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x_locations, [f"{speed:g}" for speed in speeds])
    axis.set_xlabel("Moving obstacle speed (m/s)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_success_rates(summaries: list[dict[str, Any]], output_path: Path) -> None:
    """绘制不同移动速度下的成功率。"""
    _grouped_bar(
        summaries,
        "success_rate_percent",
        "Success rate (%)",
        "Dynamic Obstacle Avoidance Success Rate",
        output_path,
    )


def plot_clearance_comparison(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """绘制每组运行最小净距离的均值。"""
    _grouped_bar(
        summaries,
        "average_minimum_obstacle_clearance_m",
        "Average minimum clearance (m)",
        "Minimum Moving-Obstacle Clearance",
        output_path,
    )


def plot_max_error_comparison(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """绘制每组运行最大编队误差的均值。"""
    _grouped_bar(
        summaries,
        "average_max_formation_error_m",
        "Average run maximum error (m)",
        "Maximum Formation Error Comparison",
        output_path,
    )


def plot_recovery_comparison(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """绘制已恢复运行的平均恢复时间，并标出未恢复次数。"""
    _grouped_bar(
        summaries,
        "average_recovery_time_recovered_s",
        "Average recovery time (s)",
        "Formation Recovery Time (NR = not recovered)",
        output_path,
        annotate_unrecovered=True,
    )


def create_dynamic_animation(
    result: Experiment4Result, output_path: Path
) -> None:
    """生成移动橙色球、历史轨迹和 UAV 机体/旋翼双视图 GIF。"""
    figure = plt.figure(figsize=(14, 6.5))
    global_axis = figure.add_subplot(121, projection="3d")
    closeup_axis = figure.add_subplot(122, projection="3d")
    frame_indices = list(range(0, len(result.times), result.config.animation_stride))
    if frame_indices[-1] != len(result.times) - 1:
        frame_indices.append(len(result.times) - 1)

    def update(frame_number: int) -> tuple[()]:
        frame = frame_indices[frame_number]
        global_axis.clear()
        closeup_axis.clear()
        _configure_3d_axes(global_axis, result)
        mean_error = float(np.mean(result.formation_errors[frame]))
        clearance = float(result.obstacle_min_clearances[frame])
        global_axis.set_title(
            f"t={result.times[frame]:.2f} s | formation error={mean_error:.2f} m\n"
            f"minimum clearance={clearance:.2f} m"
        )
        global_axis.scatter(*result.config.goal, color="black", marker="*", s=100)

        for obstacle_index, obstacle in enumerate(result.config.moving_obstacles):
            center = result.moving_obstacle_positions[frame, obstacle_index]
            _draw_moving_sphere(global_axis, center, obstacle.radius)
            obstacle_trail = result.moving_obstacle_positions[: frame + 1, obstacle_index]
            global_axis.plot(
                *obstacle_trail.T,
                color="darkorange",
                linestyle="--",
                linewidth=1.8,
            )

        for uav_index in range(result.positions.shape[1]):
            trail = result.positions[: frame + 1, uav_index]
            global_axis.plot(
                *trail.T,
                color=UAV_COLORS[uav_index],
                linewidth=1.0,
                alpha=0.7,
            )
            _draw_quadcopter(
                global_axis,
                result.positions[frame, uav_index],
                result.attitudes[frame, uav_index],
                result.rotor_phases[frame, uav_index],
                UAV_COLORS[uav_index],
                result.config.arm_length,
                result.config.rotor_radius,
            )

        representative_index = 1
        closeup_axis.set_xlim(-1.25, 1.25)
        closeup_axis.set_ylim(-1.25, 1.25)
        closeup_axis.set_zlim(-0.9, 0.9)
        closeup_axis.set_box_aspect((2.5, 2.5, 1.8))
        closeup_axis.set_xlabel("body-centered x (m)")
        closeup_axis.set_ylabel("body-centered y (m)")
        closeup_axis.set_zlabel("relative z (m)")
        closeup_axis.view_init(elev=28.0, azim=-52.0)
        closeup_axis.grid(True, alpha=0.3)
        angles = np.rad2deg(result.attitudes[frame, representative_index])
        closeup_axis.set_title(
            "UAV 1 close-up\n"
            f"roll={angles[0]:.1f}, pitch={angles[1]:.1f}, yaw={angles[2]:.1f} deg"
        )
        _draw_quadcopter(
            closeup_axis,
            np.zeros(3, dtype=float),
            result.attitudes[frame, representative_index],
            result.rotor_phases[frame, representative_index],
            UAV_COLORS[representative_index],
            result.config.arm_length,
            result.config.rotor_radius,
        )
        return ()

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / result.config.animation_fps,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=result.config.animation_fps))
    plt.close(figure)

