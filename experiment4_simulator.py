"""实验四：移动球形障碍物当前位置与预测位置避障仿真。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import Experiment4Config
from controller import compute_experiment4_velocity_commands, limit_vector_norm
from experiment2_simulator import detect_uav_contacts, minimum_inter_uav_distance
from models import MovingSphereObstacle, UAV
from simulator import (
    _formation_state,
    _update_visual_attitude_and_rotors,
    _validate_histories,
    create_uavs,
)


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class Experiment4Result:
    """一次移动障碍物实验的完整历史和指标。"""

    config: Experiment4Config
    method_name: str
    random_seed: int
    obstacle_speed: float
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    attitudes: FloatArray
    desired_attitudes: FloatArray
    rotor_speeds: FloatArray
    rotor_phases: FloatArray
    desired_positions: FloatArray
    formation_errors: FloatArray
    obstacle_forces: FloatArray
    separation_forces: FloatArray
    moving_obstacle_positions: FloatArray
    predicted_obstacle_positions: FloatArray
    moving_obstacle_velocities: FloatArray
    obstacle_min_clearances: FloatArray
    min_inter_uav_distances: FloatArray
    completed: bool
    completion_time: float | None
    obstacle_collision_count: int
    uav_collision_count: int
    metrics: dict[str, Any]
    validation: dict[str, Any]


def create_moving_obstacles(config: Experiment4Config) -> list[MovingSphereObstacle]:
    """为每次仿真重新创建无共享历史的移动障碍物。"""
    return [
        MovingSphereObstacle(
            obstacle_id=obstacle.obstacle_id,
            center=obstacle.center.copy(),
            radius=obstacle.radius,
            velocity=obstacle.velocity.copy(),
            influence_distance=obstacle.influence_distance,
        )
        for obstacle in config.moving_obstacles
    ]


def moving_obstacle_clearances(
    positions: FloatArray,
    obstacles: list[MovingSphereObstacle],
    uav_radius: float,
) -> FloatArray:
    """计算无人机到移动障碍物真实球面的净距离矩阵。"""
    return np.column_stack(
        [
            np.linalg.norm(positions - obstacle.center[None, :], axis=1)
            - obstacle.radius
            - uav_radius
            for obstacle in obstacles
        ]
    )


def detect_moving_obstacle_contacts(
    positions: FloatArray,
    obstacles: list[MovingSphereObstacle],
    uav_radius: float,
) -> set[tuple[int, int]]:
    """仅使用移动障碍物真实当前位置执行碰撞检测。"""
    contacts: set[tuple[int, int]] = set()
    for uav_id, position in enumerate(positions):
        for obstacle in obstacles:
            if (
                float(np.linalg.norm(position - obstacle.center))
                < obstacle.radius + uav_radius
            ):
                contacts.add((uav_id, obstacle.obstacle_id))
    return contacts


def _predicted_centers(
    obstacles: list[MovingSphereObstacle], config: Experiment4Config
) -> FloatArray:
    """返回当前控制时刻使用的当前/预测中心，不修改真实状态。"""
    centers = []
    for obstacle in obstacles:
        center = obstacle.center.copy()
        if config.enable_prediction:
            center += config.prediction_time * obstacle.velocity
        centers.append(center)
    return np.stack(centers)


def _record_state(
    uavs: list[UAV],
    obstacles: list[MovingSphereObstacle],
    config: Experiment4Config,
    time: float,
    obstacle_forces: FloatArray,
    separation_forces: FloatArray,
    storage: dict[str, list[Any]],
) -> None:
    """记录 UAV、移动障碍物和真实安全距离。"""
    desired, errors = _formation_state(uavs, config)
    positions = np.stack([uav.position.copy() for uav in uavs])
    clearances = moving_obstacle_clearances(
        positions, obstacles, config.uav_radius
    )
    storage["times"].append(float(time))
    storage["positions"].append(positions)
    storage["velocities"].append(np.stack([uav.velocity.copy() for uav in uavs]))
    storage["attitudes"].append(np.stack([uav.attitude.copy() for uav in uavs]))
    storage["desired_attitudes"].append(
        np.stack([uav.desired_attitude.copy() for uav in uavs])
    )
    storage["rotor_speeds"].append(
        np.stack([uav.rotor_speeds.copy() for uav in uavs])
    )
    storage["rotor_phases"].append(
        np.stack([uav.rotor_phases.copy() for uav in uavs])
    )
    storage["desired_positions"].append(desired)
    storage["formation_errors"].append(errors)
    storage["obstacle_forces"].append(obstacle_forces.copy())
    storage["separation_forces"].append(separation_forces.copy())
    storage["moving_obstacle_positions"].append(
        np.stack([obstacle.center.copy() for obstacle in obstacles])
    )
    storage["predicted_obstacle_positions"].append(
        _predicted_centers(obstacles, config)
    )
    storage["moving_obstacle_velocities"].append(
        np.stack([obstacle.velocity.copy() for obstacle in obstacles])
    )
    storage["obstacle_min_clearances"].append(float(np.min(clearances)))
    storage["min_inter_uav_distances"].append(
        minimum_inter_uav_distance(positions)
    )


def run_experiment4_simulation(config: Experiment4Config) -> Experiment4Result:
    """运行一次当前位置或预测位置动态避障。

    本实验采用以下一致顺序：读取当前状态和预测中心；计算速度命令；用旧速度
    积分位置并平滑速度；基于新速度与命令更新显示姿态/旋翼；移动真实障碍物；
    使用更新后的真实中心碰撞检测；最后记录全部状态。该顺序与实验一略有不同，
    但位置仍严格满足一阶运动学，视觉状态不反馈到运动。
    """
    uavs = create_uavs(config)
    for uav in uavs:
        uav.velocity = config.initial_velocity_bias.copy()
    obstacles = create_moving_obstacles(config)

    storage: dict[str, list[Any]] = {
        "times": [],
        "positions": [],
        "velocities": [],
        "attitudes": [],
        "desired_attitudes": [],
        "rotor_speeds": [],
        "rotor_phases": [],
        "desired_positions": [],
        "formation_errors": [],
        "obstacle_forces": [],
        "separation_forces": [],
        "moving_obstacle_positions": [],
        "predicted_obstacle_positions": [],
        "moving_obstacle_velocities": [],
        "obstacle_min_clearances": [],
        "min_inter_uav_distances": [],
    }
    zero_forces = np.zeros((len(uavs), 3), dtype=float)
    _record_state(
        uavs, obstacles, config, 0.0, zero_forces, zero_forces, storage
    )

    positions = np.stack([uav.position for uav in uavs])
    active_obstacle_contacts = detect_moving_obstacle_contacts(
        positions, obstacles, config.uav_radius
    )
    active_uav_contacts = detect_uav_contacts(positions, config.uav_radius)
    obstacle_collision_count = len(active_obstacle_contacts)
    uav_collision_count = len(active_uav_contacts)
    completed = False
    completion_time: float | None = None

    maximum_steps = int(np.ceil(config.max_time / config.dt))
    for step in range(1, maximum_steps + 1):
        # 1-3：读取当前状态，控制器内部构造预测中心并计算速度命令。
        commands, _, components = compute_experiment4_velocity_commands(
            uavs, obstacles, config
        )
        old_positions = [uav.position.copy() for uav in uavs]
        old_velocities = [uav.velocity.copy() for uav in uavs]

        # 4：严格使用旧速度更新位置，再平滑并按模长限制新速度。
        for index, uav in enumerate(uavs):
            uav.position = old_positions[index] + old_velocities[index] * config.dt
            smoothed_velocity = (
                (1.0 - config.alpha) * old_velocities[index]
                + config.alpha * commands[index]
            )
            uav.velocity = limit_vector_norm(smoothed_velocity, config.v_max)

        # 5：姿态和旋翼为显示状态，使用速度命令与新平滑速度更新。
        for index, uav in enumerate(uavs):
            _update_visual_attitude_and_rotors(uav, commands[index], config)
            uav.append_history()

        # 6：真实移动障碍物按恒定速度更新一次。
        for obstacle in obstacles:
            obstacle.update(config.dt)

        # 7：碰撞检测只读取更新后的真实中心，绝不读取预测中心。
        positions = np.stack([uav.position for uav in uavs])
        obstacle_contacts = detect_moving_obstacle_contacts(
            positions, obstacles, config.uav_radius
        )
        uav_contacts = detect_uav_contacts(positions, config.uav_radius)
        obstacle_collision_count += len(
            obstacle_contacts - active_obstacle_contacts
        )
        uav_collision_count += len(uav_contacts - active_uav_contacts)
        active_obstacle_contacts = obstacle_contacts
        active_uav_contacts = uav_contacts

        # 8：记录更新后的真实障碍物和全部 UAV/视觉/安全状态。
        current_time = step * config.dt
        _record_state(
            uavs,
            obstacles,
            config,
            current_time,
            components["obstacle"],
            components["separation"],
            storage,
        )

        leader_goal_distance = float(np.linalg.norm(uavs[0].position - config.goal))
        _, current_errors = _formation_state(uavs, config)
        if (
            not completed
            and leader_goal_distance <= config.goal_tolerance
            and float(np.mean(current_errors)) < config.formation_tolerance
        ):
            completed = True
            completion_time = current_time
        # 首次完成时间保持不变；额外记录窗口只用于确认连续 2 秒编队恢复。
        if (
            completed
            and completion_time is not None
            and current_time
            >= completion_time + config.post_completion_observation_time
        ):
            break

    arrays = {key: np.asarray(value, dtype=float) for key, value in storage.items()}
    recovery_time = _formation_recovery_time(arrays, config)
    metrics = _compute_metrics(
        arrays,
        completed,
        completion_time,
        recovery_time,
        obstacle_collision_count,
        uav_collision_count,
        config,
    )
    validation = _validate_result(arrays, completed, config)
    obstacle_speed = float(np.linalg.norm(obstacles[0].velocity))
    return Experiment4Result(
        config=config,
        method_name=config.method_name,
        random_seed=config.random_seed,
        obstacle_speed=obstacle_speed,
        times=arrays["times"],
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        attitudes=arrays["attitudes"],
        desired_attitudes=arrays["desired_attitudes"],
        rotor_speeds=arrays["rotor_speeds"],
        rotor_phases=arrays["rotor_phases"],
        desired_positions=arrays["desired_positions"],
        formation_errors=arrays["formation_errors"],
        obstacle_forces=arrays["obstacle_forces"],
        separation_forces=arrays["separation_forces"],
        moving_obstacle_positions=arrays["moving_obstacle_positions"],
        predicted_obstacle_positions=arrays["predicted_obstacle_positions"],
        moving_obstacle_velocities=arrays["moving_obstacle_velocities"],
        obstacle_min_clearances=arrays["obstacle_min_clearances"],
        min_inter_uav_distances=arrays["min_inter_uav_distances"],
        completed=completed,
        completion_time=completion_time,
        obstacle_collision_count=obstacle_collision_count,
        uav_collision_count=uav_collision_count,
        metrics=metrics,
        validation=validation,
    )


def _formation_recovery_time(
    arrays: dict[str, FloatArray], config: Experiment4Config
) -> float:
    """计算离开最后一个影响区后连续 2 秒满足误差阈值的恢复时间。"""
    influence_distance = max(
        obstacle.influence_distance for obstacle in config.moving_obstacles
    )
    encounter = arrays["obstacle_min_clearances"] < influence_distance
    encounter_indices = np.flatnonzero(encounter)
    if encounter_indices.size == 0:
        return float("nan")
    encounter_end = int(encounter_indices[-1])
    required_steps = max(
        1, int(np.ceil(config.formation_recovery_hold_time / config.dt))
    )
    mean_errors = np.mean(arrays["formation_errors"], axis=1)
    consecutive = 0
    for index in range(encounter_end + 1, len(mean_errors)):
        consecutive = (
            consecutive + 1
            if mean_errors[index] < config.formation_recovery_error
            else 0
        )
        if consecutive >= required_steps:
            # 恢复时刻包含“连续保持 2 秒”的确认时间。
            return float(arrays["times"][index] - arrays["times"][encounter_end])
    return float("nan")


def _compute_metrics(
    arrays: dict[str, FloatArray],
    completed: bool,
    completion_time: float | None,
    recovery_time: float,
    obstacle_collision_count: int,
    uav_collision_count: int,
    config: Experiment4Config,
) -> dict[str, Any]:
    """计算实验四单次运行指标。"""
    positions = arrays["positions"]
    attitudes = arrays["attitudes"]
    errors = arrays["formation_errors"]
    task_end_index = (
        int(np.searchsorted(arrays["times"], completion_time, side="right") - 1)
        if completion_time is not None
        else len(arrays["times"]) - 1
    )
    task_slice = slice(0, task_end_index + 1)
    task_positions = positions[task_slice]
    task_errors = errors[task_slice]
    final_error = float(np.mean(errors[task_end_index]))
    influence_distance = max(
        obstacle.influence_distance for obstacle in config.moving_obstacles
    )
    dynamic_mask = arrays["obstacle_min_clearances"] < influence_distance
    dynamic_attitudes = attitudes[dynamic_mask] if np.any(dynamic_mask) else attitudes
    attitude_rate = np.diff(attitudes, axis=0) / config.dt
    attitude_rms = (
        float(np.rad2deg(np.sqrt(np.mean(attitude_rate**2))))
        if attitude_rate.size
        else 0.0
    )
    success = bool(
        completed
        and obstacle_collision_count == 0
        and uav_collision_count == 0
        and final_error < 1.0
    )
    return {
        "method": config.method_name,
        "obstacle_speed_m_per_s": float(
            np.linalg.norm(config.moving_obstacles[0].velocity)
        ),
        "random_seed": int(config.random_seed),
        "success": success,
        "goal_and_formation_completed": bool(completed),
        "completion_time_s": None if completion_time is None else float(completion_time),
        "minimum_obstacle_clearance_m": float(
            np.min(arrays["obstacle_min_clearances"])
        ),
        "minimum_inter_uav_distance_m": float(
            np.min(arrays["min_inter_uav_distances"])
        ),
        "mean_formation_error_m": float(np.mean(task_errors)),
        "max_formation_error_m": float(np.max(task_errors)),
        "final_formation_error_m": final_error,
        "formation_recovery_time_s": float(recovery_time),
        "formation_recovered": bool(np.isfinite(recovery_time)),
        "total_flight_distance_m": float(
            np.sum(np.linalg.norm(np.diff(task_positions, axis=0), axis=2))
        ),
        "obstacle_collision_count": int(obstacle_collision_count),
        "uav_collision_count": int(uav_collision_count),
        "total_collision_count": int(
            obstacle_collision_count + uav_collision_count
        ),
        "dynamic_max_abs_roll_deg": float(
            np.rad2deg(np.max(np.abs(dynamic_attitudes[:, :, 0])))
        ),
        "dynamic_max_abs_pitch_deg": float(
            np.rad2deg(np.max(np.abs(dynamic_attitudes[:, :, 1])))
        ),
        "attitude_change_rms_deg_per_s": attitude_rms,
        "max_actual_speed_m_per_s": float(
            np.max(np.linalg.norm(arrays["velocities"], axis=2))
        ),
    }


def _validate_result(
    arrays: dict[str, FloatArray], completed: bool, config: Experiment4Config
) -> dict[str, Any]:
    """检查移动轨迹、预测偏移、积分、姿态和平滑约束。"""
    common = _validate_histories(arrays, completed, config)
    obstacle_step = np.diff(arrays["moving_obstacle_positions"], axis=0)
    expected_step = arrays["moving_obstacle_velocities"][:-1] * config.dt
    obstacle_motion_error = float(np.max(np.abs(obstacle_step - expected_step)))
    expected_prediction_offset = (
        arrays["moving_obstacle_velocities"] * config.prediction_time
        if config.enable_prediction
        else np.zeros_like(arrays["moving_obstacle_velocities"])
    )
    prediction_error = float(
        np.max(
            np.abs(
                arrays["predicted_obstacle_positions"]
                - arrays["moving_obstacle_positions"]
                - expected_prediction_offset
            )
        )
    )
    integration_error = float(
        np.max(
            np.abs(
                np.diff(arrays["positions"], axis=0)
                - arrays["velocities"][:-1] * config.dt
            )
        )
    )
    rp_steps = np.diff(arrays["attitudes"][:, :, :2], axis=0)
    maximum_rp_step = float(np.max(np.abs(rp_steps))) if rp_steps.size else 0.0
    common.update(
        {
            "moving_obstacle_follows_constant_velocity": obstacle_motion_error <= 1.0e-12,
            "maximum_obstacle_motion_error_m": obstacle_motion_error,
            "prediction_center_offset_is_correct": prediction_error <= 1.0e-12,
            "maximum_prediction_center_error_m": prediction_error,
            "collision_detection_uses_real_center": True,
            "position_matches_first_order_kinematics": integration_error <= 1.0e-12,
            "maximum_position_integration_error_m": integration_error,
            "attitude_changes_smoothly": maximum_rp_step <= np.deg2rad(10.0),
            "maximum_roll_pitch_step_deg": float(np.rad2deg(maximum_rp_step)),
            "all_experiment4_arrays_finite": bool(
                all(np.all(np.isfinite(value)) for value in arrays.values())
            ),
        }
    )
    return common


def verify_experiment4_visual_independence(
    baseline: Experiment4Result,
) -> tuple[bool, float]:
    """改变视觉参数复算，确认预测避障轨迹不受动画状态影响。"""
    altered_config = replace(
        baseline.config,
        beta_rp=0.72,
        beta_yaw=0.64,
        omega_hover_visual=33.0,
        omega_min_visual=5.0,
        omega_max_visual=120.0,
        k_rotor_z=6.0,
        k_rotor_attitude_visual=20.0,
    )
    altered = run_experiment4_simulation(altered_config)
    if altered.positions.shape != baseline.positions.shape:
        return False, float("inf")
    maximum_difference = float(np.max(np.abs(altered.positions - baseline.positions)))
    obstacle_difference = float(
        np.max(
            np.abs(
                altered.moving_obstacle_positions
                - baseline.moving_obstacle_positions
            )
        )
    )
    maximum_difference = max(maximum_difference, obstacle_difference)
    return maximum_difference <= 1.0e-12, maximum_difference
