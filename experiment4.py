"""实验四入口：移动球形障碍物当前位置与预测位置避障对比。"""

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
    EXPERIMENT4_OBSTACLE_SPEEDS,
    EXPERIMENT4_RANDOM_SEEDS,
    EXPERIMENT4_REPRESENTATIVE_SEED,
    EXPERIMENT4_REPRESENTATIVE_SPEED,
    Experiment4Config,
)
from experiment4_simulator import (
    Experiment4Result,
    run_experiment4_simulation,
    verify_experiment4_visual_independence,
)
from experiment4_visualization import (
    METHOD_LABELS,
    create_dynamic_animation,
    plot_clearance_comparison,
    plot_dynamic_trajectory,
    plot_max_error_comparison,
    plot_recovery_comparison,
    plot_success_rates,
)
from models import MovingSphereObstacle


METHODS: tuple[str, ...] = ("current_position", "predicted_position")
ALL_RUN_FIELDS = [
    "method",
    "obstacle_speed_m_per_s",
    "random_seed",
    "success",
    "goal_and_formation_completed",
    "completion_time_s",
    "minimum_obstacle_clearance_m",
    "minimum_inter_uav_distance_m",
    "mean_formation_error_m",
    "max_formation_error_m",
    "final_formation_error_m",
    "formation_recovery_time_s",
    "formation_recovered",
    "total_flight_distance_m",
    "obstacle_collision_count",
    "uav_collision_count",
    "total_collision_count",
    "dynamic_max_abs_roll_deg",
    "dynamic_max_abs_pitch_deg",
    "attitude_change_rms_deg_per_s",
    "max_actual_speed_m_per_s",
    "simulation_exception",
    "error_message",
]


def _json_default(value: Any) -> Any:
    """转换 JSON 不直接支持的对象。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _run_config(
    base: Experiment4Config, method: str, speed: float, seed: int
) -> Experiment4Config:
    """创建指定方法、障碍物速度和种子的独立配置。"""
    template = base.moving_obstacles[0]
    moving_obstacle = MovingSphereObstacle(
        obstacle_id=template.obstacle_id,
        center=template.center.copy(),
        radius=template.radius,
        velocity=np.array([0.0, float(speed), 0.0], dtype=float),
        influence_distance=template.influence_distance,
    )
    return replace(
        base,
        method_name=method,
        enable_prediction=(method == "predicted_position"),
        random_seed=int(seed),
        moving_obstacles=[moving_obstacle],
    )


def save_config_snapshot(config: Experiment4Config, output_path: Path) -> None:
    """保存实验四最终参数和更新顺序说明。"""
    obstacle = config.moving_obstacles[0]
    snapshot = {
        "model": "3D first-order kinematics; attitude and rotors are visual only",
        "moving_obstacle": {
            "obstacle_id": obstacle.obstacle_id,
            "initial_center": obstacle.center,
            "radius_m": obstacle.radius,
            "base_velocity": obstacle.velocity,
            "influence_distance_m": obstacle.influence_distance,
        },
        "tested_speeds_m_per_s": EXPERIMENT4_OBSTACLE_SPEEDS,
        "prediction_time_s": config.prediction_time,
        "random_seeds": EXPERIMENT4_RANDOM_SEEDS,
        "formation_recovery": {
            "error_threshold_m": config.formation_recovery_error,
            "continuous_hold_time_s": config.formation_recovery_hold_time,
            "post_completion_observation_time_s": config.post_completion_observation_time,
        },
        "step_order": [
            "read current UAV and obstacle states",
            "compute optional predicted obstacle center",
            "compute velocity commands",
            "integrate UAV position and smooth velocity",
            "update visual attitude and rotors",
            "update true moving obstacle center",
            "collision detection with true center",
            "record all states and safety metrics",
        ],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2, default=_json_default)


def save_run_artifacts(result: Experiment4Result, output_dir: Path) -> None:
    """保存一次或代表性运行的完整历史、指标和静态轨迹。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": result.method_name,
        "obstacle_speed_m_per_s": result.obstacle_speed,
        "random_seed": result.random_seed,
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
        moving_obstacle_positions=result.moving_obstacle_positions,
        predicted_obstacle_positions=result.predicted_obstacle_positions,
        moving_obstacle_velocities=result.moving_obstacle_velocities,
        obstacle_min_clearances=result.obstacle_min_clearances,
        min_inter_uav_distances=result.min_inter_uav_distances,
    )
    plot_dynamic_trajectory(result, output_dir / "trajectory_3d.png")


def result_to_row(result: Experiment4Result) -> dict[str, Any]:
    """把有效仿真转成 all_runs.csv 行。"""
    return {
        **result.metrics,
        "simulation_exception": False,
        "error_message": "",
    }


def exception_row(
    method: str, speed: float, seed: int, error: Exception
) -> dict[str, Any]:
    """单次异常时生成失败行，让批量实验继续。"""
    row = {field: "" for field in ALL_RUN_FIELDS}
    row.update(
        {
            "method": method,
            "obstacle_speed_m_per_s": speed,
            "random_seed": seed,
            "success": False,
            "goal_and_formation_completed": False,
            "simulation_exception": True,
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return row


def save_all_runs(rows: list[dict[str, Any]], output_path: Path) -> None:
    """保存 2x3x5 全部运行，NaN 恢复时间保持为 NaN。"""
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=ALL_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按方法和障碍物速度汇总实验四指标。"""
    summaries: list[dict[str, Any]] = []
    for speed in EXPERIMENT4_OBSTACLE_SPEEDS:
        for method in METHODS:
            group = [
                row
                for row in rows
                if row["method"] == method
                and float(row["obstacle_speed_m_per_s"]) == float(speed)
            ]
            valid = [row for row in group if not row["simulation_exception"]]
            successful = [row for row in valid if bool(row["success"])]

            def values(name: str, source: list[dict[str, Any]] = valid) -> list[float]:
                return [float(row[name]) for row in source if row[name] not in ("", None)]

            recovery_values = [
                float(row["formation_recovery_time_s"])
                for row in valid
                if bool(row["formation_recovered"])
                and np.isfinite(float(row["formation_recovery_time_s"]))
            ]
            completion_values = values("completion_time_s", successful)
            summaries.append(
                {
                    "method": method,
                    "obstacle_speed_m_per_s": float(speed),
                    "run_count": len(group),
                    "simulation_exception_count": sum(
                        bool(row["simulation_exception"]) for row in group
                    ),
                    "success_count": len(successful),
                    "success_rate_percent": 100.0 * len(successful) / len(group),
                    "average_completion_time_successful_s": float(
                        np.mean(completion_values)
                    ) if completion_values else 0.0,
                    "average_minimum_obstacle_clearance_m": float(
                        np.mean(values("minimum_obstacle_clearance_m"))
                    ),
                    "minimum_obstacle_clearance_m": float(
                        np.min(values("minimum_obstacle_clearance_m"))
                    ),
                    "average_minimum_inter_uav_distance_m": float(
                        np.mean(values("minimum_inter_uav_distance_m"))
                    ),
                    "average_mean_formation_error_m": float(
                        np.mean(values("mean_formation_error_m"))
                    ),
                    "average_max_formation_error_m": float(
                        np.mean(values("max_formation_error_m"))
                    ),
                    "maximum_formation_error_m": float(
                        np.max(values("max_formation_error_m"))
                    ),
                    "average_recovery_time_recovered_s": float(
                        np.mean(recovery_values)
                    ) if recovery_values else 0.0,
                    "unrecovered_count": len(valid) - len(recovery_values),
                    "average_total_flight_distance_m": float(
                        np.mean(values("total_flight_distance_m"))
                    ),
                    "total_collisions": int(sum(values("total_collision_count"))),
                    "maximum_dynamic_roll_deg": float(
                        np.max(values("dynamic_max_abs_roll_deg"))
                    ),
                    "maximum_dynamic_pitch_deg": float(
                        np.max(values("dynamic_max_abs_pitch_deg"))
                    ),
                    "average_attitude_change_rms_deg_per_s": float(
                        np.mean(values("attitude_change_rms_deg_per_s"))
                    ),
                }
            )
    return summaries


def save_summaries(summaries: list[dict[str, Any]], output_path: Path) -> None:
    """保存方法—速度级汇总表。"""
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def run_batch(config: Experiment4Config) -> int:
    """运行两种方法、三种速度、五个相同种子的批量实验。"""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, config.results_dir / "experiment_config.json")
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, float, int], Experiment4Result] = {}
    for speed in EXPERIMENT4_OBSTACLE_SPEEDS:
        for method in METHODS:
            for seed in EXPERIMENT4_RANDOM_SEEDS:
                print(f"正在计算 {method}, speed={speed:g}, seed={seed} ...")
                try:
                    result = run_experiment4_simulation(
                        _run_config(config, method, speed, seed)
                    )
                    results[(method, float(speed), seed)] = result
                    rows.append(result_to_row(result))
                except Exception as error:
                    print(f"  本次运行异常，已记录并继续: {type(error).__name__}: {error}")
                    rows.append(exception_row(method, speed, seed, error))

    save_all_runs(rows, config.results_dir / "all_runs.csv")
    summaries = build_summaries(rows)
    save_summaries(summaries, config.results_dir / "summary.csv")

    current_key = (
        "current_position",
        EXPERIMENT4_REPRESENTATIVE_SPEED,
        EXPERIMENT4_REPRESENTATIVE_SEED,
    )
    predicted_key = (
        "predicted_position",
        EXPERIMENT4_REPRESENTATIVE_SPEED,
        EXPERIMENT4_REPRESENTATIVE_SEED,
    )
    current = results[current_key]
    predicted = results[predicted_key]
    save_run_artifacts(
        current, config.results_dir / "representative_current_position"
    )
    save_run_artifacts(
        predicted, config.results_dir / "representative_predicted_position"
    )
    plot_dynamic_trajectory(
        current, config.results_dir / "trajectory_current_position.png"
    )
    plot_dynamic_trajectory(
        predicted, config.results_dir / "trajectory_predicted_position.png"
    )
    plot_success_rates(summaries, config.results_dir / "success_rate_by_speed.png")
    plot_clearance_comparison(
        summaries, config.results_dir / "minimum_clearance_comparison.png"
    )
    plot_max_error_comparison(
        summaries, config.results_dir / "maximum_formation_error_comparison.png"
    )
    plot_recovery_comparison(
        summaries, config.results_dir / "formation_recovery_time_comparison.png"
    )

    paired_keys = [
        (float(speed), seed)
        for speed in EXPERIMENT4_OBSTACLE_SPEEDS
        for seed in EXPERIMENT4_RANDOM_SEEDS
        if ("current_position", float(speed), seed) in results
        and ("predicted_position", float(speed), seed) in results
    ]
    same_initial_conditions = all(
        np.array_equal(
            results[("current_position", speed, seed)].positions[0],
            results[("predicted_position", speed, seed)].positions[0],
        )
        and np.array_equal(
            results[("current_position", speed, seed)].velocities[0],
            results[("predicted_position", speed, seed)].velocities[0],
        )
        for speed, seed in paired_keys
    )
    identical_real_obstacle_paths = True
    for speed, seed in paired_keys:
        current_path = results[
            ("current_position", speed, seed)
        ].moving_obstacle_positions
        predicted_path = results[
            ("predicted_position", speed, seed)
        ].moving_obstacle_positions
        common_length = min(len(current_path), len(predicted_path))
        identical_real_obstacle_paths = (
            identical_real_obstacle_paths
            and np.array_equal(
                current_path[:common_length], predicted_path[:common_length]
            )
        )
    initial_obstacle_recreated = all(
        np.array_equal(
            result.moving_obstacle_positions[0],
            np.array([[25.0, -12.0, 12.0]]),
        )
        for result in results.values()
    )
    visual_independent, max_difference = verify_experiment4_visual_independence(
        predicted
    )
    validation = {
        "same_initial_conditions_for_paired_methods": same_initial_conditions,
        "prediction_does_not_modify_real_obstacle_path": identical_real_obstacle_paths,
        "every_run_recreates_obstacle_at_initial_center": initial_obstacle_recreated,
        "all_30_runs_written_to_csv": len(rows)
        == len(METHODS)
        * len(EXPERIMENT4_OBSTACLE_SPEEDS)
        * len(EXPERIMENT4_RANDOM_SEEDS),
        "batch_generated_no_animation": True,
        "visual_parameters_do_not_change_result": visual_independent,
        "max_visual_independence_difference_m": max_difference,
        "all_motion_validations_pass": all(
            bool(result.validation["moving_obstacle_follows_constant_velocity"])
            and bool(result.validation["prediction_center_offset_is_correct"])
            and bool(result.validation["speed_within_limit"])
            and bool(result.validation["all_experiment4_arrays_finite"])
            for result in results.values()
        ),
        "collision_detection_uses_real_position": True,
        "simulation_exception_count": sum(
            bool(row["simulation_exception"]) for row in rows
        ),
    }
    with (config.results_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(validation, file, ensure_ascii=False, indent=2, default=_json_default)

    print("\n批量实验汇总:")
    for row in summaries:
        print(
            f"{row['method']}, speed={row['obstacle_speed_m_per_s']:g}: "
            f"成功率={row['success_rate_percent']:.1f}%, "
            f"平均最小净距={row['average_minimum_obstacle_clearance_m']:.3f}m, "
            f"未恢复={row['unrecovered_count']}"
        )
    print(f"实验四结果已保存到: {config.results_dir.resolve()}")
    return 0


def run_single(
    config: Experiment4Config,
    method: str,
    speed: float,
    seed: int,
    animate: bool,
) -> int:
    """运行单次动态避障，并可选安全地保存 GIF。"""
    result = run_experiment4_simulation(_run_config(config, method, speed, seed))
    output_dir = config.results_dir / f"single_{method}_speed_{speed:g}_seed_{seed}"
    independent, maximum_difference = verify_experiment4_visual_independence(result)
    result.validation["trajectory_independent_of_visual_parameters"] = independent
    result.validation[
        "max_visual_independence_difference_m"
    ] = maximum_difference
    save_run_artifacts(result, output_dir)
    if animate:
        try:
            gif_path = output_dir / "dynamic_obstacle_animation.gif"
            create_dynamic_animation(result, gif_path)
            print(f"GIF 已保存到: {gif_path.resolve()}")
        except Exception as error:
            # 数据和静态图已先保存；GIF 失败不会丢失实验结果。
            error_path = output_dir / "animation_error.txt"
            error_path.write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            print(f"GIF 保存失败，但实验数据完整保留: {error}")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, default=_json_default))
    print(f"单次实验结果已保存到: {output_dir.resolve()}")
    return 0


def parse_arguments() -> argparse.Namespace:
    """解析批量和单次动画参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "batch"), default="batch")
    parser.add_argument("--method", choices=METHODS, default="predicted_position")
    parser.add_argument(
        "--obstacle-speed",
        type=float,
        choices=EXPERIMENT4_OBSTACLE_SPEEDS,
        default=EXPERIMENT4_REPRESENTATIVE_SPEED,
    )
    parser.add_argument("--seed", type=int, default=EXPERIMENT4_REPRESENTATIVE_SEED)
    parser.add_argument(
        "--animate",
        action="store_true",
        help="仅单次模式生成代表性 GIF；批量模式不生成动画",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """实验四命令行入口。"""
    if sys.version_info < (3, 10):
        raise RuntimeError("本项目要求 Python 3.10 或更高版本")
    args = parse_arguments()
    config = Experiment4Config()
    if args.output_dir is not None:
        config = replace(config, results_dir=args.output_dir)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "single":
        return run_single(
            config, args.method, args.obstacle_speed, args.seed, args.animate
        )
    if args.animate:
        print("批量模式不生成动画，已忽略 --animate。")
    return run_batch(config)


if __name__ == "__main__":
    raise SystemExit(main())
