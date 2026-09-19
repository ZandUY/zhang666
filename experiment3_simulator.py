"""实验三：标准人工势场局部极小与切向逃逸仿真。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import Experiment3Config
from controller import (
    compute_experiment2_velocity_commands,
    limit_vector_norm,
    nearest_obstacle,
    tangential_escape_force,
)
from experiment2_simulator import (
    detect_obstacle_contacts,
    detect_uav_contacts,
    minimum_inter_uav_distance,
    obstacle_surface_clearances,
)
from models import UAV
from simulator import (
    _formation_state,
    _update_visual_attitude_and_rotors,
    _validate_histories,
    create_uavs,
)


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class LocalMinimumState:
    """单架 UAV 的固定窗口局部极小与逃逸状态。"""

    speed_history: deque[float]
    distance_history: deque[float]
    condition_streak: int = 0
    condition_latched: bool = False
    escape_active: bool = False
    escape_steps: int = 0
    escape_start_distance: float = 0.0
    cooldown_steps: int = 0
    detection_count: int = 0
    activation_count: int = 0
    detection_steps: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Experiment3Result:
    """实验三一次方法—随机种子组合的完整结果。"""

    config: Experiment3Config
    method_name: str
    random_seed: int
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
    escape_forces: FloatArray
    escape_active: NDArray[np.bool_]
    obstacle_min_clearances: FloatArray
    min_inter_uav_distances: FloatArray
    completed: bool
    completion_time: float | None
    obstacle_collision_count: int
    uav_collision_count: int
    local_minimum_detection_count: int
    escape_activation_count: int
    detection_steps: list[int]
    metrics: dict[str, Any]
    validation: dict[str, Any]


def _near_influence_region(uav: UAV, config: Experiment3Config) -> bool:
    """判断 UAV 是否位于任一障碍物的势场影响范围。"""
    for obstacle in config.obstacles:
        rho = (
            float(np.linalg.norm(uav.position - obstacle.center))
            - obstacle.radius
            - config.uav_radius
        )
        if rho < obstacle.influence_distance:
            return True
    return False


def _target_distances(
    uavs: list[UAV], desired_positions: FloatArray, config: Experiment3Config
) -> FloatArray:
    """领航者使用目标距离，跟随者使用编队期望位置距离。"""
    distances = np.zeros(len(uavs), dtype=float)
    distances[0] = np.linalg.norm(config.goal - uavs[0].position)
    for index in range(1, len(uavs)):
        distances[index] = np.linalg.norm(
            desired_positions[index] - uavs[index].position
        )
    return distances


def _leader_reference_direction(
    uavs: list[UAV], config: Experiment3Config
) -> FloatArray:
    """生成全队共享的、由随机种子确定符号的领航者绕行参考方向。"""
    leader = uavs[0]
    obstacle = nearest_obstacle(leader.position, config.obstacles)
    if obstacle is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    preferred_sign = 1.0 if config.random_seed % 2 == 0 else -1.0
    force = tangential_escape_force(
        leader.position,
        obstacle,
        1.0,
        config.distance_epsilon,
        leader.id,
        preferred_sign=preferred_sign,
    )
    norm = float(np.linalg.norm(force))
    return force / norm if norm > config.distance_epsilon else np.array([1.0, 0.0, 0.0])


def _update_local_minimum_monitors(
    uavs: list[UAV],
    desired_positions: FloatArray,
    states: list[LocalMinimumState],
    config: Experiment3Config,
    step: int,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """更新 20 步窗口和连续判据，并返回本步切向逃逸速度分量。"""
    distances = _target_distances(uavs, desired_positions, config)
    leader_direction = _leader_reference_direction(uavs, config)
    escape_forces = np.zeros((len(uavs), 3), dtype=float)
    active_mask = np.zeros(len(uavs), dtype=bool)
    maximum_escape_steps = max(1, int(round(config.escape_max_duration / config.dt)))
    cooldown_steps = max(0, int(round(config.escape_cooldown / config.dt)))

    for index, (uav, state) in enumerate(zip(uavs, states)):
        speed = float(np.linalg.norm(uav.velocity))
        current_distance = float(distances[index])
        state.speed_history.append(speed)
        state.distance_history.append(current_distance)
        if state.cooldown_steps > 0:
            state.cooldown_steps -= 1

        window_ready = len(state.distance_history) == config.local_min_history_steps
        improvement = (
            state.distance_history[0] - state.distance_history[-1]
            if window_ready
            else np.inf
        )
        nearby = _near_influence_region(uav, config)
        candidate = bool(
            window_ready
            and current_distance > config.local_min_distance_threshold
            and speed < config.local_min_speed_threshold
            and nearby
            and improvement < config.local_min_improvement_threshold
        )

        if state.escape_active:
            improvement_since_escape = state.escape_start_distance - current_distance
            may_exit_for_improvement = (
                state.escape_steps >= config.escape_min_active_steps
                and improvement_since_escape >= config.escape_exit_improvement
            )
            if (
                may_exit_for_improvement
                or state.escape_steps >= maximum_escape_steps
                or not nearby
                or current_distance <= config.local_min_distance_threshold
            ):
                state.escape_active = False
                state.escape_steps = 0
                state.cooldown_steps = cooldown_steps
                state.condition_streak = 0
                state.condition_latched = False

        if not state.escape_active:
            if candidate and state.cooldown_steps == 0:
                state.condition_streak += 1
            else:
                state.condition_streak = 0
                if not candidate:
                    state.condition_latched = False

            if (
                state.condition_streak >= config.local_min_consecutive_steps
                and not state.condition_latched
            ):
                state.detection_count += 1
                state.detection_steps.append(step)
                state.condition_latched = True
                if config.improved_apf_enabled:
                    state.escape_active = True
                    state.escape_steps = 0
                    state.escape_start_distance = current_distance
                    state.activation_count += 1

        if state.escape_active:
            obstacle = nearest_obstacle(uav.position, config.obstacles)
            if obstacle is not None:
                escape_forces[index] = tangential_escape_force(
                    uav.position,
                    obstacle,
                    config.k_escape,
                    config.distance_epsilon,
                    uav.id,
                    leader_direction=leader_direction if index > 0 else None,
                    preferred_sign=(1.0 if config.random_seed % 2 == 0 else -1.0),
                )
                active_mask[index] = True
                state.escape_steps += 1

    return escape_forces, active_mask


def _record_state(
    uavs: list[UAV],
    config: Experiment3Config,
    time: float,
    obstacle_forces: FloatArray,
    separation_forces: FloatArray,
    escape_forces: FloatArray,
    escape_active: NDArray[np.bool_],
    storage: dict[str, list[Any]],
) -> None:
    """记录实验三的运动、控制、安全距离和逃逸状态。"""
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
    storage["obstacle_forces"].append(obstacle_forces.copy())
    storage["separation_forces"].append(separation_forces.copy())
    storage["escape_forces"].append(escape_forces.copy())
    storage["escape_active"].append(escape_active.copy())
    storage["obstacle_min_clearances"].append(float(np.min(clearances)))
    storage["min_inter_uav_distances"].append(
        minimum_inter_uav_distance(positions)
    )


def run_experiment3_simulation(config: Experiment3Config) -> Experiment3Result:
    """运行实验三的一次标准或改进 APF 仿真。"""
    uavs = create_uavs(config)
    for uav in uavs:
        uav.velocity = config.initial_velocity_bias.copy()

    desired_initial, _ = _formation_state(uavs, config)
    initial_distances = _target_distances(uavs, desired_initial, config)
    states = [
        LocalMinimumState(
            speed_history=deque(
                [float(np.linalg.norm(uav.velocity))],
                maxlen=config.local_min_history_steps,
            ),
            distance_history=deque(
                [float(initial_distances[index])],
                maxlen=config.local_min_history_steps,
            ),
        )
        for index, uav in enumerate(uavs)
    ]

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
        "escape_forces": [],
        "escape_active": [],
        "obstacle_min_clearances": [],
        "min_inter_uav_distances": [],
    }
    zero_forces = np.zeros((len(uavs), 3), dtype=float)
    zero_active = np.zeros(len(uavs), dtype=bool)
    _record_state(
        uavs,
        config,
        0.0,
        zero_forces,
        zero_forces,
        zero_forces,
        zero_active,
        storage,
    )

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
        base_commands, desired_positions, components = (
            compute_experiment2_velocity_commands(uavs, config)
        )
        escape_forces, escape_active = _update_local_minimum_monitors(
            uavs, desired_positions, states, config, step
        )
        commands = base_commands + escape_forces
        for index in range(len(uavs)):
            commands[index] = limit_vector_norm(commands[index], config.v_max)

        old_positions = [uav.position.copy() for uav in uavs]
        old_velocities = [uav.velocity.copy() for uav in uavs]
        for index, uav in enumerate(uavs):
            # 逃逸分量仍先形成期望速度，再经原有姿态平滑和视觉旋翼更新。
            _update_visual_attitude_and_rotors(uav, commands[index], config)
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
            escape_forces,
            escape_active,
            storage,
        )

        positions = np.stack([uav.position for uav in uavs])
        obstacle_contacts = detect_obstacle_contacts(
            positions, config.obstacles, config.uav_radius
        )
        uav_contacts = detect_uav_contacts(positions, config.uav_radius)
        obstacle_collision_count += len(
            obstacle_contacts - active_obstacle_contacts
        )
        uav_collision_count += len(uav_contacts - active_uav_contacts)
        active_obstacle_contacts = obstacle_contacts
        active_uav_contacts = uav_contacts

        leader_goal_distance = float(np.linalg.norm(uavs[0].position - config.goal))
        _, current_errors = _formation_state(uavs, config)
        if (
            leader_goal_distance <= config.goal_tolerance
            and float(np.mean(current_errors)) < config.formation_tolerance
        ):
            completed = True
            completion_time = current_time
            break

    arrays = {
        key: np.asarray(value, dtype=(bool if key == "escape_active" else float))
        for key, value in storage.items()
    }
    detection_count = sum(state.detection_count for state in states)
    activation_count = sum(state.activation_count for state in states)
    detection_steps = sorted(
        step_index for state in states for step_index in state.detection_steps
    )
    metrics = _compute_metrics(
        arrays,
        completed,
        completion_time,
        obstacle_collision_count,
        uav_collision_count,
        detection_count,
        activation_count,
        config,
    )
    validation = _validate_result(
        arrays, completed, detection_steps, activation_count, config
    )
    return Experiment3Result(
        config=config,
        method_name=config.method_name,
        random_seed=config.random_seed,
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
        escape_forces=arrays["escape_forces"],
        escape_active=arrays["escape_active"],
        obstacle_min_clearances=arrays["obstacle_min_clearances"],
        min_inter_uav_distances=arrays["min_inter_uav_distances"],
        completed=completed,
        completion_time=completion_time,
        obstacle_collision_count=obstacle_collision_count,
        uav_collision_count=uav_collision_count,
        local_minimum_detection_count=detection_count,
        escape_activation_count=activation_count,
        detection_steps=detection_steps,
        metrics=metrics,
        validation=validation,
    )


def _compute_metrics(
    arrays: dict[str, FloatArray],
    completed: bool,
    completion_time: float | None,
    obstacle_collision_count: int,
    uav_collision_count: int,
    detection_count: int,
    activation_count: int,
    config: Experiment3Config,
) -> dict[str, Any]:
    """计算一次实验三运行的全部指标。"""
    positions = arrays["positions"]
    attitudes = arrays["attitudes"]
    errors = arrays["formation_errors"]
    total_distance = float(
        np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2))
    )
    escape_mask = arrays["escape_active"]
    if np.any(escape_mask):
        escape_roll = np.abs(attitudes[:, :, 0][escape_mask])
        escape_pitch = np.abs(attitudes[:, :, 1][escape_mask])
        escape_max_roll = float(np.rad2deg(np.max(escape_roll)))
        escape_max_pitch = float(np.rad2deg(np.max(escape_pitch)))
    else:
        escape_max_roll = 0.0
        escape_max_pitch = 0.0

    if len(attitudes) > 1:
        attitude_rate = np.diff(attitudes, axis=0) / config.dt
        attitude_rms = float(np.rad2deg(np.sqrt(np.mean(attitude_rate**2))))
    else:
        attitude_rms = 0.0
    final_error = float(np.mean(errors[-1]))
    success = bool(
        completed
        and obstacle_collision_count == 0
        and uav_collision_count == 0
        and final_error < 1.0
    )
    return {
        "method": config.method_name,
        "random_seed": int(config.random_seed),
        "success": success,
        "goal_and_formation_completed": bool(completed),
        "completion_time_s": None if completion_time is None else float(completion_time),
        "total_flight_distance_m": total_distance,
        "mean_formation_error_m": float(np.mean(errors)),
        "max_formation_error_m": float(np.max(errors)),
        "final_formation_error_m": final_error,
        "minimum_obstacle_clearance_m": float(
            np.min(arrays["obstacle_min_clearances"])
        ),
        "minimum_inter_uav_distance_m": float(
            np.min(arrays["min_inter_uav_distances"])
        ),
        "local_minimum_detection_count": int(detection_count),
        "escape_activation_count": int(activation_count),
        "obstacle_collision_count": int(obstacle_collision_count),
        "uav_collision_count": int(uav_collision_count),
        "total_collision_count": int(
            obstacle_collision_count + uav_collision_count
        ),
        "escape_max_abs_roll_deg": escape_max_roll,
        "escape_max_abs_pitch_deg": escape_max_pitch,
        "attitude_change_rms_deg_per_s": attitude_rms,
        "max_actual_speed_m_per_s": float(
            np.max(np.linalg.norm(arrays["velocities"], axis=2))
        ),
    }


def _maximum_true_run(mask: NDArray[np.bool_]) -> int:
    """返回二维布尔历史中任一列的最长连续 True 步数。"""
    maximum = 0
    for column in range(mask.shape[1]):
        current = 0
        for value in mask[:, column]:
            current = current + 1 if value else 0
            maximum = max(maximum, current)
    return maximum


def _validate_result(
    arrays: dict[str, FloatArray],
    completed: bool,
    detection_steps: list[int],
    activation_count: int,
    config: Experiment3Config,
) -> dict[str, Any]:
    """检查多步检测、切向力、积分、姿态和公共约束。"""
    common = _validate_histories(arrays, completed, config)
    nonzero_escape = np.linalg.norm(arrays["escape_forces"], axis=2)
    active_norms = nonzero_escape[nonzero_escape > 1.0e-12]
    expected_earliest = (
        config.local_min_history_steps - 1 + config.local_min_consecutive_steps
    )
    integration_error = float(
        np.max(
            np.abs(
                np.diff(arrays["positions"], axis=0)
                - arrays["velocities"][:-1] * config.dt
            )
        )
    )
    maximum_escape_steps = int(round(config.escape_max_duration / config.dt))
    attitude_steps = np.diff(arrays["attitudes"][:, :, :2], axis=0)
    maximum_attitude_step = (
        float(np.max(np.abs(attitude_steps))) if attitude_steps.size else 0.0
    )
    common.update(
        {
            "standard_apf_has_zero_escape_force": bool(
                config.improved_apf_enabled or np.all(nonzero_escape == 0.0)
            ),
            "standard_apf_has_zero_escape_activations": bool(
                config.improved_apf_enabled or activation_count == 0
            ),
            "local_minimum_requires_full_window_and_streak": bool(
                not detection_steps or min(detection_steps) >= expected_earliest
            ),
            "escape_force_has_k_escape_norm": bool(
                active_norms.size == 0
                or np.allclose(active_norms, config.k_escape, atol=1.0e-10)
            ),
            "escape_duration_within_limit": bool(
                _maximum_true_run(arrays["escape_active"])
                <= maximum_escape_steps
            ),
            "position_matches_first_order_kinematics": integration_error <= 1.0e-12,
            "maximum_position_integration_error_m": integration_error,
            "escape_attitude_changes_smoothly": maximum_attitude_step <= np.deg2rad(10.0),
            "maximum_roll_pitch_step_deg": float(np.rad2deg(maximum_attitude_step)),
            "escape_only_modifies_velocity_command": True,
            "rotor_speeds_not_used_by_escape_control": True,
            "all_experiment3_arrays_finite": bool(
                np.all(np.isfinite(arrays["obstacle_forces"]))
                and np.all(np.isfinite(arrays["separation_forces"]))
                and np.all(np.isfinite(arrays["escape_forces"]))
            ),
        }
    )
    return common


def verify_experiment3_visual_independence(
    baseline: Experiment3Result,
) -> tuple[bool, float]:
    """改变视觉参数复算，确认姿态/旋翼动画不改变逃逸轨迹。"""
    altered_config = replace(
        baseline.config,
        beta_rp=0.71,
        beta_yaw=0.63,
        omega_hover_visual=32.0,
        omega_min_visual=5.0,
        omega_max_visual=120.0,
        k_rotor_z=6.0,
        k_rotor_attitude_visual=20.0,
    )
    altered = run_experiment3_simulation(altered_config)
    if altered.positions.shape != baseline.positions.shape:
        return False, float("inf")
    maximum_difference = float(np.max(np.abs(altered.positions - baseline.positions)))
    return maximum_difference <= 1.0e-12, maximum_difference

