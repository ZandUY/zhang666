"""实验二的障碍物轨迹、安全距离、误差与结果表绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiment2_simulator import Experiment2Result
from visualization import UAV_COLORS, draw_sphere_obstacles


METHOD_COLORS = {
    "A_no_avoidance": "tab:red",
    "B_obstacle_apf": "tab:orange",
    "C_apf_and_separation": "tab:green",
}

METHOD_LABELS = {
    "A_no_avoidance": "方法 A：无避障",
    "B_obstacle_apf": "方法 B：障碍物人工势场（APF）",
    "C_apf_and_separation": "方法 C：人工势场（APF）+ 机间避碰",
}


def _configure_axis(axis: plt.Axes, result: Experiment2Result) -> None:
    """应用实验二固定三维视野。"""
    config = result.config
    axis.set_xlim(*config.x_limits)
    axis.set_ylim(*config.y_limits)
    axis.set_zlim(*config.z_limits)
    axis.set_xlabel("X 坐标（m）")
    axis.set_ylabel("Y 坐标（m）")
    axis.set_zlabel("Z 坐标（m）")
    axis.set_box_aspect(
        (
            config.x_limits[1] - config.x_limits[0],
            config.y_limits[1] - config.y_limits[0],
            config.z_limits[1] - config.z_limits[0],
        )
    )
    axis.view_init(elev=25.0, azim=-62.0)
    axis.grid(True, alpha=0.3)


def plot_method_trajectory(result: Experiment2Result, output_path: Path) -> None:
    """绘制一种方法的五架无人机轨迹和球形障碍物。"""
    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection="3d")
    draw_sphere_obstacles(axis, result.config.obstacles)
    for index in range(result.positions.shape[1]):
        trajectory = result.positions[:, index]
        label = "领航无人机（UAV 0）" if index == 0 else f"跟随无人机（UAV {index}）"
        axis.plot(*trajectory.T, color=UAV_COLORS[index], linewidth=1.8, label=label)
        axis.scatter(*trajectory[0], color=UAV_COLORS[index], marker="o", s=25)
        axis.scatter(*trajectory[-1], color=UAV_COLORS[index], marker="^", s=35)
    axis.scatter(*result.config.goal, color="black", marker="*", s=130, label="目标点")
    _configure_axis(axis, result)
    axis.set_title(METHOD_LABELS[result.method_name])
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_trajectories(
    results: list[Experiment2Result], output_path: Path
) -> None:
    """将 A/B/C 三种方法的三维轨迹并排比较。"""
    figure = plt.figure(figsize=(18, 6))
    for plot_index, result in enumerate(results, start=1):
        axis = figure.add_subplot(1, 3, plot_index, projection="3d")
        draw_sphere_obstacles(axis, result.config.obstacles)
        for uav_index in range(result.positions.shape[1]):
            axis.plot(
                *result.positions[:, uav_index].T,
                color=UAV_COLORS[uav_index],
                linewidth=1.4,
            )
        axis.scatter(*result.config.goal, color="black", marker="*", s=90)
        _configure_axis(axis, result)
        axis.set_title(METHOD_LABELS[result.method_name])
    figure.suptitle("实验二：三维飞行轨迹对比")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_formation_error_comparison(
    results: list[Experiment2Result], output_path: Path
) -> None:
    """比较三种方法的平均编队误差。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for result in results:
        axis.plot(
            result.times,
            np.mean(result.formation_errors, axis=1),
            color=METHOD_COLORS[result.method_name],
            linewidth=1.8,
            label=METHOD_LABELS[result.method_name],
        )
    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="1 m 阈值")
    axis.set_xlabel("时间（s）")
    axis.set_ylabel("平均编队误差（m）")
    axis.set_title("平均编队误差对比")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_obstacle_clearance_comparison(
    results: list[Experiment2Result], output_path: Path
) -> None:
    """比较集群对障碍物的最小表面净距离。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for result in results:
        axis.plot(
            result.times,
            result.obstacle_min_clearances,
            color=METHOD_COLORS[result.method_name],
            linewidth=1.8,
            label=METHOD_LABELS[result.method_name],
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="碰撞边界")
    axis.set_xlabel("时间（s）")
    axis.set_ylabel("最小障碍物净距离（m）")
    axis.set_title("无人机与障碍物表面净距离")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_inter_uav_distance_comparison(
    results: list[Experiment2Result], output_path: Path
) -> None:
    """比较集群的最小机间中心距离。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for result in results:
        axis.plot(
            result.times,
            result.min_inter_uav_distances,
            color=METHOD_COLORS[result.method_name],
            linewidth=1.8,
            label=METHOD_LABELS[result.method_name],
        )
    config = results[0].config
    axis.axhline(
        2.0 * config.uav_radius,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="碰撞边界",
    )
    axis.axhline(
        config.d_safe,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label="安全距离（d_safe）",
    )
    axis.set_xlabel("时间（s）")
    axis.set_ylabel("最小无人机间距（m）")
    axis.set_title("无人机间安全距离")
    axis.grid(True, alpha=0.3)
    axis.legend()

    # B/C 差异量级为厘米，主图还需要保留碰撞边界，因此增加局部放大窗。
    inset = axis.inset_axes([0.12, 0.13, 0.34, 0.28])
    for result in results[1:]:
        inset.plot(
            result.times,
            result.min_inter_uav_distances,
            color=METHOD_COLORS[result.method_name],
            linewidth=1.4,
        )
    inset.set_xlim(8.0, 13.0)
    inset.set_ylim(2.90, 3.08)
    inset.set_title("方法 B/C 最小值局部放大", fontsize=8)
    inset.grid(True, alpha=0.25)
    inset.tick_params(labelsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_result_table(results: list[Experiment2Result], output_path: Path) -> None:
    """把实验二主要指标保存为可直接查看的图片表格。"""
    columns = [
        "方法",
        "成功",
        "时间（s）",
        "平均误差（m）",
        "最大误差（m）",
        "最小障碍物净距（m）",
        "最小机间距（m）",
        "障碍物碰撞",
        "机间碰撞",
        "总航程（m）",
        "最大横滚角（°）",
        "最大俯仰角（°）",
        "限幅次数",
    ]
    rows: list[list[str]] = []
    for result in results:
        metric = result.metrics
        rows.append(
            [
                result.method_name.split("_")[0],
                "是" if metric["success"] else "否",
                "-" if metric["completion_time_s"] is None else f"{metric['completion_time_s']:.2f}",
                f"{metric['mean_formation_error_m']:.3f}",
                f"{metric['max_formation_error_m']:.3f}",
                f"{metric['minimum_obstacle_clearance_m']:.3f}",
                f"{metric['minimum_inter_uav_distance_m']:.3f}",
                str(metric["obstacle_collision_count"]),
                str(metric["uav_collision_count"]),
                f"{metric['total_flight_distance_m']:.2f}",
                f"{metric['avoidance_max_abs_roll_deg']:.2f}",
                f"{metric['avoidance_max_abs_pitch_deg']:.2f}",
                str(metric["attitude_limit_sample_count"]),
            ]
        )
    figure, axis = plt.subplots(figsize=(18, 2.8))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    axis.set_title("实验二结果对比", pad=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
