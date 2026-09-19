"""实验二：球形障碍物人工势场与无人机间防碰撞仿真。

位置仍采用实验一的三维一阶运动学更新。人工势场只生成期望速度，姿态和
旋翼仍是视觉状态，不包含推力、力矩或飞行动力学方程。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import Experiment2Config
from controller import (
    compute_experiment2_velocity_commands,
    desired_formation_positions,
    limit_vector_norm,
    wrap_to_pi,
    yaw_from_velocity,
)
from models import SphereObstacle, UAV
from simulator import (
    _formation_state,
    _update_visual_attitude_and_rotors,
    _validate_histories,
    create_uavs,
)


FloatArray = NDArray[np.float64]
ObstacleContact = tuple[int, int]
UAVContact = tuple[int, int]


@dataclass(slots=True)
class Experiment2Result:
    """实验二单种方法的完整历史和评价结果。"""

    config: Experiment2Config
    method_name: str
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
    obstacle_min_clearances: FloatArray
    min_inter_uav_distances: FloatArray
    completed: bool
    completion_time: float | None
    obstacle_collision_count: int
    uav_collision_count: int
    metrics: dict[str, Any]
    validation: dict[str, Any]


def obstacle_surface_clearances(
    positions: FloatArray,
    obstacles: list[SphereObstacle],
    uav_radius: float,
) -> FloatArray:
    """返回 shape=(无人机数, 障碍物数) 的表面净距离。"""
    if not obstacles:
        return np.empty((len(positions), 0), dtype=float)
    return np.column_stack(
        [
            np.linalg.norm(positions - obstacle.center[None, :], axis=1)
            - obstacle.radius
            - uav_radius
            for obstacle in obstacles
        ]
    )


def detect_obstacle_contacts(
    positions: FloatArray,
    obstacles: list[SphereObstacle],
    uav_radius: float,
) -> set[ObstacleContact]:
    """检测当前无人机—障碍物接触集合。"""
    contacts: set[ObstacleContact] = set()
    for uav_id, position in enumerate(positions):
        for obstacle in obstacles:
            if (
                float(np.linalg.norm(position - obstacle.center))
                < obstacle.radius + uav_radius
            ):
                contacts.add((uav_id, obstacle.obstacle_id))
    return contacts


def detect_uav_contacts(
    positions: FloatArray, uav_radius: float
) -> set[UAVContact]:
    """检测当前无序无人机碰撞对；(i,j) 与 (j,i) 只保留一个。"""
    contacts: set[UAVContact] = set()
    for first in range(len(positions) - 1):
        for second in range(first + 1, len(positions)):
            if float(np.linalg.norm(positions[first] - positions[second])) < 2.0 * uav_radius:
                contacts.add((first, second))
    return contacts


def minimum_inter_uav_distance(positions: FloatArray) -> float:
    """计算当前所有无序无人机对的最小中心距离。"""
    minimum = np.inf
    for first in range(len(positions) - 1):
        for second in range(first + 1, len(positions)):
            minimum = min(
                minimum,
                float(np.linalg.norm(positions[first] - positions[second])),
            )
    return float(minimum)


def _record_state(
    uavs: list[UAV],
    config: Experiment2Config,
    time: float,
    obstacle_force: FloatArray,
    separation_force: FloatArray,
    storage: dict[str, list[Any]],
) -> None:
    """记录实验二同一时刻的运动、视觉和安全距离状态。"""
    desired, errors = _formation_state(uavs, config)
    positions = np.stack([uav.position.copy() for uav in uavs])
    clearances = obstacle_surface_clearances(
        positions, config.obstacles, config.uav_radius
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
    storage["obstacle_forces"].append(obstacle_force.copy())
    storage["separation_forces"].append(separation_force.copy())
    storage["obstacle_min_clearances"].append(
        float(np.min(clearances)) if clearances.size else float("inf")
    )
    storage["min_inter_uav_distances"].append(
        minimum_inter_uav_distance(positions)
    )


def run_experiment2_simulation(config: Experiment2Config) -> Experiment2Result:
    """运行实验二的一种控制方法。"""
    uavs = create_uavs(config)
    # 三种方法共享同一初始位置和同一确定性初速度。
    for uav in uavs:
        uav.velocity = config.initial_velocity_bias.copy()

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
        "obstacle_min_clearances": [],
        "min_inter_uav_distances": [],
    }
    zero_forces = np.zeros((len(uavs), 3), dtype=float)
    _record_state(uavs, config, 0.0, zero_forces, zero_forces, storage)

    positions = np.stack([uav.position for uav in uavs])
    active_obstacle_contacts = detect_obstacle_contacts(
        positions, config.obstacles, config.uav_radius
    )
    active_uav_contacts = detect_uav_contacts(positions, config.uav_radius)
    obstacle_collision_count = len(active_obstacle_contacts)
    uav_collision_count = len(active_uav_contacts)

    completed = False
    completion_time: float | None = None
    maximum_steps = int(np.ceil(config.max_time / config.dt))

    for step in range(1, maximum_steps + 1):
        commands, _, components = compute_experiment2_velocity_commands(uavs, config)
        old_positions = [uav.position.copy() for uav in uavs]
        old_velocities = [uav.velocity.copy() for uav in uavs]

        for index, uav in enumerate(uavs):
            # 势场只影响期望速度；沿用实验一的显示姿态和视觉旋翼更新。
            _update_visual_attitude_and_rotors(uav, commands[index], config)
            # 不直接改位置避障，严格执行一阶运动学积分。
            uav.position = old_positions[index] + old_velocities[index] * config.dt
            smoothed_velocity = (
                (1.0 - config.alpha) * old_velocities[index]
                + config.alpha * commands[index]
            )
            uav.velocity = limit_vector_norm(smoothed_velocity, config.v_max)
            uav.append_history()

        current_time = step * config.dt
        _record_state(
            uavs,
            config,
            current_time,
            components["obstacle"],
            components["separation"],
            storage,
        )

        positions = np.stack([uav.position for uav in uavs])
        current_obstacle_contacts = detect_obstacle_contacts(
            positions, config.obstacles, config.uav_radius
        )
        current_uav_contacts = detect_uav_contacts(positions, config.uav_radius)
        obstacle_collision_count += len(
            current_obstacle_contacts - active_obstacle_contacts
        )
        uav_collision_count += len(current_uav_contacts - active_uav_contacts)
        active_obstacle_contacts = current_obstacle_contacts
        active_uav_contacts = current_uav_contacts

        leader_goal_distance = float(np.linalg.norm(uavs[0].position - config.goal))
        _, current_errors = _formation_state(uavs, config)
        if (
            leader_goal_distance <= config.goal_tolerance
            and float(np.mean(current_errors)) < config.formation_tolerance
        ):
            completed = True
            completion_time = current_time
            break

    arrays = {key: np.asarray(value, dtype=float) for key, value in storage.items()}
    metrics = _compute_metrics(
        arrays,
        completed,
        completion_time,
        obstacle_collision_count,
        uav_collision_count,
        config,
    )
    validation = _validate_experiment2(
        arrays, completed, obstacle_collision_count, uav_collision_count, config
    )
    return Experiment2Result(
        config=config,
        method_name=config.method_name,
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
        obstacle_min_clearances=arrays["obstacle_min_clearances"],
        min_inter_uav_distances=arrays["min_inter_uav_distances"],
        completed=completed,
        completion_time=completion_time,
        obstacle_collision_count=obstacle_collision_count,
        uav_collision_count=uav_collision_count,
        metrics=metrics,
        validation=validation,
    )


def _compute_metrics(
    arrays: dict[str, FloatArray],
    completed: bool,
    completion_time: float | None,
    obstacle_collision_count: int,
    uav_collision_count: int,
    config: Experiment2Config,
) -> dict[str, Any]:
    """计算实验二要求的全部评价指标。"""
    positions = arrays["positions"]
    attitudes = arrays["attitudes"]
    errors = arrays["formation_errors"]
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    total_distance = float(np.sum(segment_lengths))

    influence_limit = max(
        (obstacle.influence_distance for obstacle in config.obstacles), default=0.0
    )
    avoidance_mask = arrays["obstacle_min_clearances"] < influence_limit
    avoidance_attitudes = attitudes[avoidance_mask] if np.any(avoidance_mask) else attitudes
    roll_abs = np.abs(avoidance_attitudes[:, :, 0])
    pitch_abs = np.abs(avoidance_attitudes[:, :, 1])
    at_limit = (
        (np.abs(attitudes[:, :, 0]) >= config.max_roll - 1.0e-6)
        | (np.abs(attitudes[:, :, 1]) >= config.max_pitch - 1.0e-6)
    )
    success = bool(
        completed and obstacle_collision_count == 0 and uav_collision_count == 0
    )
    return {
        "success": success,
        "goal_and_formation_completed": bool(completed),
        "completion_time_s": None if completion_time is None else float(completion_time),
        "mean_formation_error_m": float(np.mean(errors)),
        "max_formation_error_m": float(np.max(errors)),
        "final_formation_error_m": float(np.mean(errors[-1])),
        "minimum_obstacle_clearance_m": float(
            np.min(arrays["obstacle_min_clearances"])
        ),
        "minimum_inter_uav_distance_m": float(
            np.min(arrays["min_inter_uav_distances"])
        ),
        "obstacle_collision_count": int(obstacle_collision_count),
        "uav_collision_count": int(uav_collision_count),
        "total_flight_distance_m": total_distance,
        "avoidance_max_abs_roll_deg": float(np.rad2deg(np.max(roll_abs))),
        "avoidance_max_abs_pitch_deg": float(np.rad2deg(np.max(pitch_abs))),
        "attitude_limit_sample_count": int(np.count_nonzero(at_limit)),
        "max_actual_speed_m_per_s": float(
            np.max(np.linalg.norm(arrays["velocities"], axis=2))
        ),
        "final_leader_goal_distance_m": float(
            np.linalg.norm(positions[-1, 0] - config.goal)
        ),
    }


def _validate_experiment2(
    arrays: dict[str, FloatArray],
    completed: bool,
    obstacle_collision_count: int,
    uav_collision_count: int,
    config: Experiment2Config,
) -> dict[str, Any]:
    """执行实验二自身与实验一公共约束的自动检查。"""
    common = _validate_histories(arrays, completed, config)
    finite_extra = bool(
        np.all(np.isfinite(arrays["obstacle_forces"]))
        and np.all(np.isfinite(arrays["separation_forces"]))
        and np.all(np.isfinite(arrays["obstacle_min_clearances"]))
        and np.all(np.isfinite(arrays["min_inter_uav_distances"]))
    )
    expected_steps = arrays["velocities"][:-1] * config.dt
    actual_steps = np.diff(arrays["positions"], axis=0)
    integration_error = float(np.max(np.abs(actual_steps - expected_steps)))
    attitude_steps = np.diff(arrays["attitudes"][:, :, :2], axis=0)
    maximum_rp_step = (
        float(np.max(np.abs(attitude_steps))) if attitude_steps.size else 0.0
    )
    common.update(
        {
            "avoidance_and_separation_states_finite": finite_extra,
            "position_matches_first_order_kinematics": integration_error <= 1.0e-12,
            "maximum_position_integration_error_m": integration_error,
            "attitude_tilt_changes_smoothly": maximum_rp_step <= np.deg2rad(10.0),
            "maximum_roll_pitch_step_deg": float(np.rad2deg(maximum_rp_step)),
            "collision_detection_active": True,
            "obstacle_collision_count": int(obstacle_collision_count),
            "uav_collision_count": int(uav_collision_count),
            "artificial_potential_used_as_velocity_command_only": True,
        }
    )
    return common


def verify_experiment2_visual_independence(
    baseline: Experiment2Result,
) -> tuple[bool, float]:
    """改变全部视觉参数后复算，确认实验二轨迹不发生变化。"""
    altered_config = replace(
        baseline.config,
        beta_rp=0.73,
        beta_yaw=0.62,
        omega_hover_visual=31.0,
        omega_min_visual=5.0,
        omega_max_visual=120.0,
        k_rotor_z=7.0,
        k_rotor_attitude_visual=21.0,
    )
    altered = run_experiment2_simulation(altered_config)
    if altered.positions.shape != baseline.positions.shape:
        return False, float("inf")
    maximum_difference = float(np.max(np.abs(altered.positions - baseline.positions)))
    return maximum_difference <= 1.0e-12, maximum_difference

