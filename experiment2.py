"""实验二入口：球形障碍物人工势场与无人机间防碰撞对比。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from config import Experiment2Config
from controller import obstacle_repulsion_force, separation_forces
from experiment2_simulator import (
    Experiment2Result,
    run_experiment2_simulation,
    verify_experiment2_visual_independence,
)
from experiment2_visualization import (
    METHOD_LABELS,
    plot_all_trajectories,
    plot_formation_error_comparison,
    plot_inter_uav_distance_comparison,
    plot_method_trajectory,
    plot_obstacle_clearance_comparison,
    plot_result_table,
)
from models import UAV
from visualization import create_formation_animation


METHODS: tuple[tuple[str, bool, bool], ...] = (
    ("A_no_avoidance", False, False),
    ("B_obstacle_apf", True, False),
    ("C_apf_and_separation", True, True),
)


def _json_default(value: Any) -> Any:
    """转换 JSON 不直接支持的常用对象。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def save_method_result(result: Experiment2Result, output_dir: Path) -> None:
    """保存一种方法的历史、指标、验收信息和三维轨迹图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": result.method_name,
        "method_label": METHOD_LABELS[result.method_name],
        "obstacle_avoidance_enabled": result.config.enable_obstacle_avoidance,
        "separation_enabled": result.config.enable_separation,
        "model_note": (
            "三维一阶运动学模型；人工势场只生成期望速度，姿态和旋翼为视觉状态，"
            "不包含四旋翼飞行动力学。"
        ),
        "metrics": result.metrics,
        "validation": result.validation,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=_json_default)
    np.savez_compressed(
        output_dir / "histories.npz",
        times=result.times,
        positions=result.positions,
        velocities=result.velocities,
        attitudes=result.attitudes,
        desired_attitudes=result.desired_attitudes,
        rotor_speeds=result.rotor_speeds,
        rotor_phases=result.rotor_phases,
        desired_positions=result.desired_positions,
        formation_errors=result.formation_errors,
        obstacle_forces=result.obstacle_forces,
        separation_forces=result.separation_forces,
        obstacle_min_clearances=result.obstacle_min_clearances,
        min_inter_uav_distances=result.min_inter_uav_distances,
    )
    plot_method_trajectory(result, output_dir / "trajectory_3d.png")


def save_summary_csv(results: list[Experiment2Result], output_path: Path) -> None:
    """保存实验二三种方法的完整指标对比表。"""
    fieldnames = [
        "method",
        "success",
        "goal_and_formation_completed",
        "completion_time_s",
        "mean_formation_error_m",
        "max_formation_error_m",
        "minimum_obstacle_clearance_m",
        "minimum_inter_uav_distance_m",
        "obstacle_collision_count",
        "uav_collision_count",
        "total_flight_distance_m",
        "avoidance_max_abs_roll_deg",
        "avoidance_max_abs_pitch_deg",
        "attitude_limit_sample_count",
        "max_actual_speed_m_per_s",
        "final_formation_error_m",
        "final_leader_goal_distance_m",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({"method": result.method_name, **result.metrics})


def controller_unit_checks(config: Experiment2Config) -> dict[str, Any]:
    """对排斥方向、范围外零向量和重合机间方向做确定性检查。"""
    obstacle = config.obstacles[0]
    outside_position = obstacle.center + np.array(
        [obstacle.radius + config.uav_radius + obstacle.influence_distance + 1.0, 0.0, 0.0]
    )
    outside_force = obstacle_repulsion_force(
        outside_position,
        config.obstacles,
        config.uav_radius,
        config.k_obs,
        config.max_obstacle_force,
        config.distance_epsilon,
    )
    sample_position = obstacle.center + np.array(
        [obstacle.radius + config.uav_radius + 1.0, 1.0, 0.5]
    )
    sample_force = obstacle_repulsion_force(
        sample_position,
        config.obstacles,
        config.uav_radius,
        config.k_obs,
        config.max_obstacle_force,
        config.distance_epsilon,
    )
    away_dot = float(np.dot(sample_force, sample_position - obstacle.center))

    overlapping = [
        UAV(id=0, position=np.zeros(3, dtype=float)),
        UAV(id=1, position=np.zeros(3, dtype=float)),
    ]
    overlap_force_first = separation_forces(
        overlapping,
        config.d_safe,
        config.k_sep,
        config.max_separation_force,
        config.distance_epsilon,
    )
    overlap_force_second = separation_forces(
        overlapping,
        config.d_safe,
        config.k_sep,
        config.max_separation_force,
        config.distance_epsilon,
    )
    return {
        "outside_influence_returns_shape_3": outside_force.shape == (3,),
        "outside_influence_returns_exact_zero": bool(np.array_equal(outside_force, np.zeros(3))),
        "obstacle_repulsion_points_away": away_dot > 0.0,
        "obstacle_force_within_limit": bool(
            np.linalg.norm(sample_force) <= config.max_obstacle_force + 1.0e-12
        ),
        "overlap_separation_is_finite": bool(np.all(np.isfinite(overlap_force_first))),
        "overlap_separation_is_deterministic": bool(
            np.array_equal(overlap_force_first, overlap_force_second)
        ),
        "overlap_pair_forces_are_opposite": bool(
            np.allclose(overlap_force_first[0], -overlap_force_first[1])
        ),
        "overlap_pair_force_is_nonzero": bool(
            np.linalg.norm(overlap_force_first[0]) > 0.0
        ),
        "sample_away_dot_product": away_dot,
    }


def cross_method_checks(results: list[Experiment2Result]) -> dict[str, Any]:
    """检查 A/B/C 的公平初值和实验二关键对比结论。"""
    by_name = {result.method_name: result for result in results}
    method_a = by_name["A_no_avoidance"]
    method_b = by_name["B_obstacle_apf"]
    method_c = by_name["C_apf_and_separation"]
    initial_positions_equal = all(
        np.array_equal(result.positions[0], method_a.positions[0])
        for result in results[1:]
    )
    initial_velocities_equal = all(
        np.array_equal(result.velocities[0], method_a.velocities[0])
        for result in results[1:]
    )
    return {
        "all_methods_same_initial_positions": initial_positions_equal,
        "all_methods_same_initial_velocities": initial_velocities_equal,
        "no_avoidance_detects_obstacle_collision": (
            method_a.obstacle_collision_count > 0
        ),
        "obstacle_apf_avoids_obstacle": (
            method_b.obstacle_collision_count == 0
            and method_b.metrics["minimum_obstacle_clearance_m"] >= 0.0
        ),
        "combined_method_avoids_obstacle": (
            method_c.obstacle_collision_count == 0
            and method_c.metrics["minimum_obstacle_clearance_m"] >= 0.0
        ),
        "separation_improves_minimum_distance": (
            method_c.metrics["minimum_inter_uav_distance_m"]
            > method_b.metrics["minimum_inter_uav_distance_m"]
        ),
        "method_b_minimum_inter_uav_distance_m": method_b.metrics[
            "minimum_inter_uav_distance_m"
        ],
        "method_c_minimum_inter_uav_distance_m": method_c.metrics[
            "minimum_inter_uav_distance_m"
        ],
    }


def print_result(result: Experiment2Result) -> None:
    """打印一种方法的实验二指标。"""
    metric = result.metrics
    print(f"\n=== {METHOD_LABELS[result.method_name]} ===")
    print(f"是否成功: {metric['success']}")
    print(f"任务完成时间: {metric['completion_time_s']} s")
    print(f"平均/最大编队误差: {metric['mean_formation_error_m']:.4f} / {metric['max_formation_error_m']:.4f} m")
    print(f"障碍物最小净距离: {metric['minimum_obstacle_clearance_m']:.4f} m")
    print(f"最小机间距离: {metric['minimum_inter_uav_distance_m']:.4f} m")
    print(f"障碍物/无人机碰撞次数: {metric['obstacle_collision_count']} / {metric['uav_collision_count']}")
    print(f"集群总飞行距离: {metric['total_flight_distance_m']:.3f} m")
    print(
        "避障过程最大横滚/俯仰: "
        f"{metric['avoidance_max_abs_roll_deg']:.3f} / "
        f"{metric['avoidance_max_abs_pitch_deg']:.3f} deg"
    )
    print(f"姿态达到显示上限次数: {metric['attitude_limit_sample_count']}")


def parse_arguments() -> argparse.Namespace:
    """解析实验二命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-animation", action="store_true", help="跳过代表性 GIF 生成"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="覆盖默认输出目录 results/experiment_2",
    )
    return parser.parse_args()


def main() -> int:
    """运行 A/B/C 三种方法、作图、保存指标并执行自动验收。"""
    if sys.version_info < (3, 10):
        raise RuntimeError("本项目要求 Python 3.10 或更高版本")
    args = parse_arguments()
    base_config = Experiment2Config()
    if args.output_dir is not None:
        base_config = replace(base_config, results_dir=args.output_dir)
    base_config.results_dir.mkdir(parents=True, exist_ok=True)

    results: list[Experiment2Result] = []
    for method_name, obstacle_enabled, separation_enabled in METHODS:
        config = replace(
            base_config,
            method_name=method_name,
            enable_obstacle_avoidance=obstacle_enabled,
            enable_separation=separation_enabled,
        )
        print(f"正在计算 {METHOD_LABELS[method_name]} ...")
        result = run_experiment2_simulation(config)
        independent, maximum_difference = verify_experiment2_visual_independence(result)
        result.validation["trajectory_independent_of_visual_parameters"] = independent
        result.validation[
            "max_position_difference_after_visual_parameter_change_m"
        ] = maximum_difference
        save_method_result(result, base_config.results_dir / method_name)
        print_result(result)
        results.append(result)

    plot_all_trajectories(results, base_config.results_dir / "trajectory_comparison_3d.png")
    plot_formation_error_comparison(
        results, base_config.results_dir / "formation_error_comparison.png"
    )
    plot_obstacle_clearance_comparison(
        results, base_config.results_dir / "obstacle_clearance_comparison.png"
    )
    plot_inter_uav_distance_comparison(
        results, base_config.results_dir / "inter_uav_distance_comparison.png"
    )
    plot_result_table(results, base_config.results_dir / "result_comparison_table.png")
    save_summary_csv(results, base_config.results_dir / "metrics_summary.csv")

    checks = {
        "controller_unit_checks": controller_unit_checks(base_config),
        "cross_method_checks": cross_method_checks(results),
    }
    with (base_config.results_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(checks, file, ensure_ascii=False, indent=2, default=_json_default)

    if not args.skip_animation:
        representative = next(
            result for result in results if result.method_name == "C_apf_and_separation"
        )
        gif_path = base_config.results_dir / "avoidance_animation_method_C.gif"
        print(f"\n正在生成代表性 GIF: {gif_path}")
        create_formation_animation(representative, gif_path)

    boolean_checks = [
        value
        for group in checks.values()
        for value in group.values()
        if isinstance(value, bool)
    ]
    print("\n实验二跨方法验收: " + ("全部通过" if all(boolean_checks) else "存在未通过项"))
    print(f"实验二结果已保存到: {base_config.results_dir.resolve()}")
    return 0 if all(boolean_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
