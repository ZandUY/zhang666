"""实验一入口：三维编队形成、保持、姿态跟随与旋翼视觉动画。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from config import EXPERIMENT_K_FORM_VALUES, SimulationConfig
from simulator import (
    SimulationResult,
    run_simulation,
    verify_visual_state_independence,
)
from visualization import (
    create_formation_animation,
    plot_attitudes,
    plot_error_comparison,
    plot_formation_errors,
    plot_rotor_speeds,
    plot_trajectories,
)


def _gain_directory_name(k_form: float) -> str:
    """把 0.6 转成适合作为目录名的 k_form_0_6。"""
    return f"k_form_{k_form:g}".replace(".", "_")


def _json_default(value: Any) -> Any:
    """把少量 numpy/Path 对象转成 JSON 可序列化对象。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def save_result_files(result: SimulationResult, output_dir: Path) -> None:
    """保存单组增益的指标、历史数组和四类静态图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "k_form": result.config.k_form,
        "model_note": (
            "三维一阶运动学模型；姿态和视觉旋翼不构成四旋翼飞行动力学模型，"
            "rotor_speeds 不参与位置、速度或姿态更新。"
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
    )

    plot_trajectories(result, output_dir / "trajectory_3d.png")
    plot_formation_errors(result, output_dir / "formation_errors.png")
    plot_attitudes(result, output_dir / "attitudes_leader_and_uav1.png")
    plot_rotor_speeds(result, output_dir / "rotor_speeds_uav1.png", uav_index=1)


def save_summary(results: list[SimulationResult], output_path: Path) -> None:
    """以 CSV 汇总三组编队增益的主要指标。"""
    fieldnames = [
        "k_form",
        "reached_goal_and_formation",
        "completion_time_s",
        "mean_formation_error_m",
        "max_formation_error_m",
        "final_formation_error_m",
        "uav_0_distance_m",
        "uav_1_distance_m",
        "uav_2_distance_m",
        "uav_3_distance_m",
        "uav_4_distance_m",
        "total_flight_distance_m",
        "max_abs_roll_deg",
        "max_abs_pitch_deg",
        "attitude_change_rms_deg_per_s",
        "visual_rotor_speed_min_rad_per_s",
        "visual_rotor_speed_max_rad_per_s",
        "max_actual_speed_m_per_s",
        "all_validations_passed",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result.metrics
            distances = metrics["flight_distance_per_uav_m"]
            validation_bools = [
                value
                for key, value in result.validation.items()
                if isinstance(value, bool) and key != "position_update_uses_rotor_thrust"
                and key != "model_is_quadcopter_flight_dynamics"
            ]
            writer.writerow(
                {
                    "k_form": result.config.k_form,
                    "reached_goal_and_formation": metrics[
                        "reached_goal_and_formation"
                    ],
                    "completion_time_s": metrics["completion_time_s"],
                    "mean_formation_error_m": metrics["mean_formation_error_m"],
                    "max_formation_error_m": metrics["max_formation_error_m"],
                    "final_formation_error_m": metrics["final_formation_error_m"],
                    **{
                        f"uav_{index}_distance_m": distances[f"uav_{index}"]
                        for index in range(5)
                    },
                    "total_flight_distance_m": metrics["total_flight_distance_m"],
                    "max_abs_roll_deg": metrics["max_abs_roll_deg"],
                    "max_abs_pitch_deg": metrics["max_abs_pitch_deg"],
                    "attitude_change_rms_deg_per_s": metrics[
                        "attitude_change_rms_deg_per_s"
                    ],
                    "visual_rotor_speed_min_rad_per_s": metrics[
                        "visual_rotor_speed_range_rad_per_s"
                    ][0],
                    "visual_rotor_speed_max_rad_per_s": metrics[
                        "visual_rotor_speed_range_rad_per_s"
                    ][1],
                    "max_actual_speed_m_per_s": metrics[
                        "max_actual_speed_m_per_s"
                    ],
                    "all_validations_passed": all(validation_bools),
                }
            )


def print_result(result: SimulationResult) -> None:
    """向终端打印一组实验的简明指标。"""
    metrics = result.metrics
    rotor_range = metrics["visual_rotor_speed_range_rad_per_s"]
    print(f"\n=== k_form={result.config.k_form:g} ===")
    print(f"到达目标并形成编队: {metrics['reached_goal_and_formation']}")
    print(f"任务完成时间: {metrics['completion_time_s']} s")
    print(f"全过程平均编队误差: {metrics['mean_formation_error_m']:.4f} m")
    print(f"最大编队误差: {metrics['max_formation_error_m']:.4f} m")
    print(f"最终平均编队误差: {metrics['final_formation_error_m']:.4f} m")
    for name, distance in metrics["flight_distance_per_uav_m"].items():
        print(f"{name} 飞行距离: {distance:.3f} m")
    print(f"集群总飞行距离: {metrics['total_flight_distance_m']:.3f} m")
    print(f"最大横滚角: {metrics['max_abs_roll_deg']:.3f} deg")
    print(f"最大俯仰角: {metrics['max_abs_pitch_deg']:.3f} deg")
    print(
        "姿态变化均方根: "
        f"{metrics['attitude_change_rms_deg_per_s']:.3f} deg/s"
    )
    print(f"视觉旋翼转速范围: [{rotor_range[0]:.3f}, {rotor_range[1]:.3f}] rad/s")
    failed = [
        name
        for name, passed in result.validation.items()
        if isinstance(passed, bool)
        and not passed
        and name not in {
            "position_update_uses_rotor_thrust",
            "model_is_quadcopter_flight_dynamics",
        }
    ]
    print("自动验收: " + ("全部通过" if not failed else f"未通过 {failed}"))


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="跳过 GIF 生成（静态图和全部数值实验仍会运行）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="覆盖默认输出目录 results/experiment_1",
    )
    return parser.parse_args()


def main() -> int:
    """运行三组实验、自动验收、绘图并生成代表性 GIF。"""
    if sys.version_info < (3, 10):
        raise RuntimeError("本项目要求 Python 3.10 或更高版本")

    args = parse_arguments()
    base_config = SimulationConfig()
    if args.output_dir is not None:
        base_config = replace(base_config, results_dir=args.output_dir)
    base_config.results_dir.mkdir(parents=True, exist_ok=True)

    results: list[SimulationResult] = []
    for k_form in EXPERIMENT_K_FORM_VALUES:
        config = replace(base_config, k_form=k_form)
        print(f"正在计算 k_form={k_form:g} ...")
        result = run_simulation(config)

        independent, maximum_difference = verify_visual_state_independence(result)
        result.validation["trajectory_independent_of_visual_parameters"] = independent
        result.validation[
            "max_position_difference_after_visual_parameter_change_m"
        ] = maximum_difference

        result_dir = base_config.results_dir / _gain_directory_name(k_form)
        save_result_files(result, result_dir)
        print_result(result)
        results.append(result)

    plot_error_comparison(
        results, base_config.results_dir / "formation_error_comparison.png"
    )
    save_summary(results, base_config.results_dir / "metrics_summary.csv")

    representative = min(results, key=lambda item: abs(item.config.k_form - 0.6))
    if not args.skip_animation:
        gif_path = base_config.results_dir / "formation_animation_k_form_0_6.gif"
        print(f"\n正在生成 GIF: {gif_path}")
        create_formation_animation(representative, gif_path)

    print(f"\n实验一结果已保存到: {base_config.results_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

