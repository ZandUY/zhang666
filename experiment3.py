"""实验三入口：标准 APF 局部极小与切向逃逸的单次/十次重复实验。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    EXPERIMENT3_RANDOM_SEEDS,
    EXPERIMENT3_REPRESENTATIVE_SEED,
    Experiment3Config,
)
from controller import tangential_escape_force
from experiment3_simulator import (
    Experiment3Result,
    run_experiment3_simulation,
    verify_experiment3_visual_independence,
)
from experiment3_visualization import (
    METHOD_LABELS,
    plot_average_completion_time,
    plot_average_flight_distance,
    plot_formation_error_comparison,
    plot_success_rate,
    plot_trajectory,
)
from visualization import create_formation_animation


METHODS: tuple[str, ...] = ("standard_apf", "improved_apf")
ALL_RUN_FIELDS = [
    "method",
    "random_seed",
    "success",
    "goal_and_formation_completed",
    "completion_time_s",
    "total_flight_distance_m",
    "mean_formation_error_m",
    "max_formation_error_m",
    "final_formation_error_m",
    "minimum_obstacle_clearance_m",
    "minimum_inter_uav_distance_m",
    "local_minimum_detection_count",
    "escape_activation_count",
    "obstacle_collision_count",
    "uav_collision_count",
    "total_collision_count",
    "escape_max_abs_roll_deg",
    "escape_max_abs_pitch_deg",
    "attitude_change_rms_deg_per_s",
    "max_actual_speed_m_per_s",
    "simulation_exception",
    "error_message",
]


def _json_default(value: Any) -> Any:
    """把 numpy 和 Path 对象转换为 JSON 兼容类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _method_config(
    base: Experiment3Config, method: str, seed: int
) -> Experiment3Config:
    """为方法和固定随机种子构造独立配置。"""
    return replace(
        base,
        method_name=method,
        random_seed=int(seed),
        improved_apf_enabled=(method == "improved_apf"),
    )


def save_config_snapshot(config: Experiment3Config, output_path: Path) -> None:
    """记录实验三最终场景和调参说明。"""
    snapshot = {
        "model": "3D first-order kinematics; attitude and rotors are visual only",
        "goal": config.goal,
        "initial_velocity_m_per_s": config.initial_velocity_bias,
        "obstacles": [
            {
                "obstacle_id": obstacle.obstacle_id,
                "center": obstacle.center,
                "radius_m": obstacle.radius,
                "influence_distance_m": obstacle.influence_distance,
            }
            for obstacle in config.obstacles
        ],
        "uav_radius_m": config.uav_radius,
        "k_obs": config.k_obs,
        "max_obstacle_force": config.max_obstacle_force,
        "obstacle_parameter_note": (
            "Compared with experiment 2, k_obs was increased from 8 to 20 and "
            "max_obstacle_force from 5 to 8 so that the symmetric four-obstacle "
            "field reproducibly exhibits a standard-APF local minimum."
        ),
        "d_safe_m": config.d_safe,
        "k_sep": config.k_sep,
        "local_minimum": {
            "history_steps": config.local_min_history_steps,
            "distance_threshold_m": config.local_min_distance_threshold,
            "speed_threshold_m_per_s": config.local_min_speed_threshold,
            "improvement_threshold_m": config.local_min_improvement_threshold,
            "consecutive_steps": config.local_min_consecutive_steps,
        },
        "escape": {
            "k_escape": config.k_escape,
            "maximum_duration_s": config.escape_max_duration,
            "exit_improvement_m": config.escape_exit_improvement,
            "cooldown_s": config.escape_cooldown,
        },
        "random_seeds": EXPERIMENT3_RANDOM_SEEDS,
        "representative_seed": EXPERIMENT3_REPRESENTATIVE_SEED,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2, default=_json_default)


def save_run_artifacts(result: Experiment3Result, output_dir: Path) -> None:
    """保存单次或代表性运行的完整历史、指标与轨迹图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": result.method_name,
        "random_seed": result.random_seed,
        "metrics": result.metrics,
        "validation": result.validation,
        "detection_steps": result.detection_steps,
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
        escape_forces=result.escape_forces,
        escape_active=result.escape_active,
        obstacle_min_clearances=result.obstacle_min_clearances,
        min_inter_uav_distances=result.min_inter_uav_distances,
    )
    plot_trajectory(result, output_dir / "trajectory_3d.png")


def result_to_row(result: Experiment3Result) -> dict[str, Any]:
    """把有效仿真结果转换为 all_runs.csv 行。"""
    return {
        **result.metrics,
        "simulation_exception": False,
        "error_message": "",
    }


def exception_row(method: str, seed: int, error: Exception) -> dict[str, Any]:
    """在单次仿真异常时生成仍可写入 CSV 的失败行。"""
    row = {field: "" for field in ALL_RUN_FIELDS}
    row.update(
        {
            "method": method,
            "random_seed": int(seed),
            "success": False,
            "goal_and_formation_completed": False,
            "simulation_exception": True,
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return row


def save_all_runs(rows: list[dict[str, Any]], output_path: Path) -> None:
    """保存所有种子，包括失败和异常运行。"""
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=ALL_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按方法汇总成功率、时间、距离、误差、碰撞和姿态指标。"""
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        valid_rows = [row for row in method_rows if not row["simulation_exception"]]
        successful = [row for row in valid_rows if bool(row["success"])]

        def values(name: str, source: list[dict[str, Any]] = valid_rows) -> list[float]:
            return [float(row[name]) for row in source if row[name] not in ("", None)]

        completion_values = values("completion_time_s", successful)
        summary = {
            "method": method,
            "run_count": len(method_rows),
            "simulation_exception_count": sum(
                bool(row["simulation_exception"]) for row in method_rows
            ),
            "success_count": len(successful),
            "success_rate_percent": 100.0 * len(successful) / len(method_rows),
            "average_completion_time_successful_s": (
                float(np.mean(completion_values)) if completion_values else 0.0
            ),
            "average_total_flight_distance_m": float(
                np.mean(values("total_flight_distance_m"))
            ),
            "average_mean_formation_error_m": float(
                np.mean(values("mean_formation_error_m"))
            ),
            "maximum_formation_error_m": float(
                np.max(values("max_formation_error_m"))
            ),
            "minimum_obstacle_clearance_m": float(
                np.min(values("minimum_obstacle_clearance_m"))
            ),
            "total_local_minimum_detections": int(
                sum(values("local_minimum_detection_count"))
            ),
            "total_escape_activations": int(sum(values("escape_activation_count"))),
            "total_collisions": int(sum(values("total_collision_count"))),
            "maximum_escape_roll_deg": float(
                np.max(values("escape_max_abs_roll_deg"))
            ),
            "maximum_escape_pitch_deg": float(
                np.max(values("escape_max_abs_pitch_deg"))
            ),
            "average_attitude_change_rms_deg_per_s": float(
                np.mean(values("attitude_change_rms_deg_per_s"))
            ),
        }
        summaries.append(summary)
    return summaries


def save_summaries(summaries: list[dict[str, Any]], output_path: Path) -> None:
    """保存实验三方法级汇总 CSV。"""
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def tangent_unit_checks(config: Experiment3Config) -> dict[str, Any]:
    """检查普通和退化几何下的切向归一化与正交性。"""
    obstacle = config.obstacles[0]
    sample_position = obstacle.center + np.array([4.0, 2.0, 1.0])
    force = tangential_escape_force(
        sample_position,
        obstacle,
        config.k_escape,
        config.distance_epsilon,
        0,
        preferred_sign=1.0,
    )
    radial = sample_position - obstacle.center
    radial /= np.linalg.norm(radial)
    vertical_position = obstacle.center + np.array([0.0, 0.0, 4.0])
    fallback_force = tangential_escape_force(
        vertical_position,
        obstacle,
        config.k_escape,
        config.distance_epsilon,
        1,
        preferred_sign=1.0,
    )
    return {
        "tangent_force_has_k_escape_norm": bool(
            np.isclose(np.linalg.norm(force), config.k_escape)
        ),
        "tangent_is_perpendicular_to_radial": bool(
            abs(float(np.dot(force, radial))) < 1.0e-12
        ),
        "fallback_tangent_is_finite": bool(np.all(np.isfinite(fallback_force))),
        "fallback_tangent_has_k_escape_norm": bool(
            np.isclose(np.linalg.norm(fallback_force), config.k_escape)
        ),
    }


def run_batch(config: Experiment3Config) -> int:
    """运行两种方法各 10 个固定种子；单次异常不会中断批次。"""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, config.results_dir / "experiment_config.json")
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, int], Experiment3Result] = {}

    for method in METHODS:
        for seed in EXPERIMENT3_RANDOM_SEEDS:
            print(f"正在计算 {method}, seed={seed} ...")
            try:
                result = run_experiment3_simulation(
                    _method_config(config, method, seed)
                )
                results[(method, seed)] = result
                rows.append(result_to_row(result))
            except Exception as error:  # 批量实验必须保存失败并继续后续种子
                print(f"  本次运行异常，已记录并继续: {type(error).__name__}: {error}")
                rows.append(exception_row(method, seed, error))

    save_all_runs(rows, config.results_dir / "all_runs.csv")
    summaries = build_summaries(rows)
    save_summaries(summaries, config.results_dir / "summary.csv")

    representative_standard = results.get(
        ("standard_apf", EXPERIMENT3_REPRESENTATIVE_SEED)
    )
    representative_improved = results.get(
        ("improved_apf", EXPERIMENT3_REPRESENTATIVE_SEED)
    )
    if representative_standard is None or representative_improved is None:
        raise RuntimeError("代表性随机种子运行异常，无法生成对比轨迹")

    standard_dir = config.results_dir / "representative_standard_apf"
    improved_dir = config.results_dir / "representative_improved_apf"
    save_run_artifacts(representative_standard, standard_dir)
    save_run_artifacts(representative_improved, improved_dir)
    # 同时按题目要求在实验根目录给出明确命名的两幅轨迹图。
    plot_trajectory(
        representative_standard,
        config.results_dir / "trajectory_standard_apf.png",
    )
    plot_trajectory(
        representative_improved,
        config.results_dir / "trajectory_improved_apf.png",
    )
    plot_success_rate(summaries, config.results_dir / "success_rate.png")
    plot_average_completion_time(
        summaries, config.results_dir / "average_completion_time.png"
    )
    plot_average_flight_distance(
        summaries, config.results_dir / "average_flight_distance.png"
    )
    plot_formation_error_comparison(
        representative_standard,
        representative_improved,
        config.results_dir / "formation_error_comparison.png",
    )

    initial_pairs_equal = all(
        np.array_equal(
            results[("standard_apf", seed)].positions[0],
            results[("improved_apf", seed)].positions[0],
        )
        and np.array_equal(
            results[("standard_apf", seed)].velocities[0],
            results[("improved_apf", seed)].velocities[0],
        )
        for seed in EXPERIMENT3_RANDOM_SEEDS
        if ("standard_apf", seed) in results and ("improved_apf", seed) in results
    )
    standard_zero_escape = all(
        result.escape_activation_count == 0
        and np.all(result.escape_forces == 0.0)
        for (method, _), result in results.items()
        if method == "standard_apf"
    )
    multistep_detection = all(
        bool(result.validation["local_minimum_requires_full_window_and_streak"])
        for result in results.values()
    )
    independent, max_difference = verify_experiment3_visual_independence(
        representative_improved
    )
    validation = {
        "tangent_checks": tangent_unit_checks(config),
        "batch_checks": {
            "same_initial_conditions_for_each_seed": initial_pairs_equal,
            "standard_method_never_uses_escape_force": standard_zero_escape,
            "all_detections_use_window_and_consecutive_streak": multistep_detection,
            "all_20_runs_written_to_csv": len(rows) == 2 * len(EXPERIMENT3_RANDOM_SEEDS),
            "standard_failures_are_present_in_csv": any(
                row["method"] == "standard_apf" and not bool(row["success"])
                for row in rows
            ),
            "batch_continued_after_failures": len(rows)
            == 2 * len(EXPERIMENT3_RANDOM_SEEDS),
            "visual_parameters_do_not_change_trajectory": independent,
            "max_visual_independence_position_difference_m": max_difference,
            "simulation_exception_count": sum(
                bool(row["simulation_exception"]) for row in rows
            ),
        },
    }
    with (config.results_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(validation, file, ensure_ascii=False, indent=2, default=_json_default)

    print("\n批量实验汇总:")
    for summary in summaries:
        print(
            f"{summary['method']}: 成功率={summary['success_rate_percent']:.1f}%, "
            f"成功平均时间={summary['average_completion_time_successful_s']:.3f}s, "
            f"局部极小检测={summary['total_local_minimum_detections']}, "
            f"逃逸启用={summary['total_escape_activations']}"
        )
    print(f"结果已保存到: {config.results_dir.resolve()}")
    return 0


def run_single(config: Experiment3Config, method: str, seed: int, animate: bool) -> int:
    """运行一次指定方法和种子，并可选生成动画。"""
    single_config = _method_config(config, method, seed)
    output_dir = config.results_dir / f"single_{method}_seed_{seed}"
    result = run_experiment3_simulation(single_config)
    save_run_artifacts(result, output_dir)
    independent, maximum_difference = verify_experiment3_visual_independence(result)
    result.validation["trajectory_independent_of_visual_parameters"] = independent
    result.validation[
        "max_position_difference_after_visual_parameter_change_m"
    ] = maximum_difference
    # 视觉独立性字段加入后重新写入指标文件。
    save_run_artifacts(result, output_dir)
    if animate:
        create_formation_animation(result, output_dir / "animation.gif")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, default=_json_default))
    print(f"单次实验结果已保存到: {output_dir.resolve()}")
    return 0


def parse_arguments() -> argparse.Namespace:
    """解析单次与批量运行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "batch"), default="batch")
    parser.add_argument("--method", choices=METHODS, default="improved_apf")
    parser.add_argument("--seed", type=int, default=EXPERIMENT3_REPRESENTATIVE_SEED)
    parser.add_argument(
        "--animate",
        action="store_true",
        help="仅单次模式可选生成 GIF；批量模式始终不生成动画",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """实验三命令行入口。"""
    if sys.version_info < (3, 10):
        raise RuntimeError("本项目要求 Python 3.10 或更高版本")
    args = parse_arguments()
    config = Experiment3Config()
    if args.output_dir is not None:
        config = replace(config, results_dir=args.output_dir)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "single":
        return run_single(config, args.method, args.seed, args.animate)
    if args.animate:
        print("批量模式按要求不生成动画，已忽略 --animate。")
    return run_batch(config)


if __name__ == "__main__":
    raise SystemExit(main())
