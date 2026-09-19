"""实验三多障碍物轨迹与十次重复实验统计图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiment3_simulator import Experiment3Result
from visualization import UAV_COLORS, draw_sphere_obstacles


METHOD_LABELS = {
    "standard_apf": "Standard APF",
    "improved_apf": "Improved APF + tangential escape",
}
METHOD_COLORS = {"standard_apf": "tab:orange", "improved_apf": "tab:green"}


def _configure_axis(axis: plt.Axes, result: Experiment3Result) -> None:
    """设置多障碍物场景的固定三维坐标范围。"""
    config = result.config
    axis.set_xlim(*config.x_limits)
    axis.set_ylim(*config.y_limits)
    axis.set_zlim(*config.z_limits)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_box_aspect(
        (
            config.x_limits[1] - config.x_limits[0],
            config.y_limits[1] - config.y_limits[0],
            config.z_limits[1] - config.z_limits[0],
        )
    )
    axis.view_init(elev=25.0, azim=-62.0)
    axis.grid(True, alpha=0.3)


def plot_trajectory(result: Experiment3Result, output_path: Path) -> None:
    """绘制一次标准或改进 APF 的五机三维轨迹。"""
    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection="3d")
    draw_sphere_obstacles(axis, result.config.obstacles)
    for uav_index in range(result.positions.shape[1]):
        trajectory = result.positions[:, uav_index]
        label = "Leader UAV 0" if uav_index == 0 else f"Follower UAV {uav_index}"
        axis.plot(
            *trajectory.T,
            color=UAV_COLORS[uav_index],
            linewidth=1.6,
            label=label,
        )
        axis.scatter(*trajectory[0], color=UAV_COLORS[uav_index], marker="o", s=24)
        axis.scatter(*trajectory[-1], color=UAV_COLORS[uav_index], marker="^", s=32)
    axis.scatter(*result.config.goal, color="black", marker="*", s=120, label="Goal")
    _configure_axis(axis, result)
    outcome = "success" if result.metrics["success"] else "failure"
    axis.set_title(
        f"{METHOD_LABELS[result.method_name]} | seed={result.random_seed} | {outcome}"
    )
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _bar_chart(
    summaries: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    suffix: str = "",
) -> None:
    """绘制两种方法的简单统计柱状图。"""
    methods = [row["method"] for row in summaries]
    values = [float(row[metric]) for row in summaries]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    bars = axis.bar(labels, values, color=colors, width=0.58)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_success_rate(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """保存成功率柱状图。"""
    _bar_chart(
        summaries,
        "success_rate_percent",
        "Success rate (%)",
        "Success Rate over 10 Fixed Seeds",
        output_path,
        "%",
    )


def plot_average_completion_time(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """保存成功运行的平均任务时间柱状图。"""
    _bar_chart(
        summaries,
        "average_completion_time_successful_s",
        "Average completion time (s)",
        "Average Completion Time (Successful Runs)",
        output_path,
    )


def plot_average_flight_distance(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """保存全部有效运行的平均集群飞行距离柱状图。"""
    _bar_chart(
        summaries,
        "average_total_flight_distance_m",
        "Average total flight distance (m)",
        "Average Swarm Flight Distance",
        output_path,
    )


def plot_formation_error_comparison(
    standard: Experiment3Result,
    improved: Experiment3Result,
    output_path: Path,
) -> None:
    """比较代表性相同种子下两种方法的平均编队误差。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for result in (standard, improved):
        axis.plot(
            result.times,
            np.mean(result.formation_errors, axis=1),
            color=METHOD_COLORS[result.method_name],
            linewidth=1.8,
            label=METHOD_LABELS[result.method_name],
        )
    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="1 m")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mean formation error (m)")
    axis.set_title(f"Formation Error Comparison (seed={standard.random_seed})")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

