"""实验一仿真计算。

该模块不导入 matplotlib，也不执行绘图。位置采用三维一阶运动学模型；
姿态与旋翼状态仅用于显示，不是四旋翼飞行动力学模型。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import SimulationConfig
from controller import (
    compute_velocity_commands,
    desired_formation_positions,
    limit_vector_norm,
    rotation_z,
    wrap_to_pi,
    yaw_from_velocity,
)
from models import UAV


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class SimulationResult:
    """一次仿真的完整时间历史、指标和验收检查结果。"""

    config: SimulationConfig
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    attitudes: FloatArray
    desired_attitudes: FloatArray
    rotor_speeds: FloatArray
    rotor_phases: FloatArray
    desired_positions: FloatArray
    formation_errors: FloatArray
    completed: bool
    completion_time: float | None
    metrics: dict[str, Any]
    validation: dict[str, Any]


def create_uavs(config: SimulationConfig) -> list[UAV]:
    """按固定随机种子创建领航者和四架扰动后的跟随者。"""
    rng = np.random.default_rng(config.random_seed)
    uavs: list[UAV] = []
    for index, offset in enumerate(config.formation_offsets):
        if index == 0:
            position = config.leader_initial_position.copy()
        else:
            perturbation = rng.uniform(
                -config.follower_initial_perturbation,
                config.follower_initial_perturbation,
                size=3,
            )
            position = config.leader_initial_position + offset + perturbation

        uav = UAV(
            id=index,
            position=position,
            velocity=np.zeros(3, dtype=float),
            attitude=np.zeros(3, dtype=float),
            desired_attitude=np.zeros(3, dtype=float),
            last_valid_yaw=0.0,
            rotor_speeds=np.full(4, config.omega_hover_visual, dtype=float),
            rotor_phases=np.zeros(4, dtype=float),
            is_leader=(index == 0),
        )
        uav.append_history()
        uavs.append(uav)
    return uavs


def _update_visual_attitude_and_rotors(
    uav: UAV,
    velocity_command: FloatArray,
    config: SimulationConfig,
) -> None:
    """更新显示姿态和旋翼动画状态。

    这里没有质量、惯量、推力、力矩、电机响应或空气动力学方程。
    旋翼混控只是为了让四个旋翼的视觉转速随姿态误差产生轻微差异。
    """
    desired_acceleration = (velocity_command - uav.velocity) / config.tau_v

    yaw_desired, yaw_is_valid = yaw_from_velocity(
        uav.velocity, uav.last_valid_yaw, config.horizontal_speed_epsilon
    )
    if yaw_is_valid:
        uav.last_valid_yaw = yaw_desired

    acceleration_body = rotation_z(-yaw_desired) @ desired_acceleration
    pitch_desired = float(
        np.clip(
            acceleration_body[0] / config.g,
            -config.max_pitch,
            config.max_pitch,
        )
    )
    roll_desired = float(
        np.clip(
            -acceleration_body[1] / config.g,
            -config.max_roll,
            config.max_roll,
        )
    )
    uav.desired_attitude[:] = (roll_desired, pitch_desired, yaw_desired)

    uav.attitude[0] += config.beta_rp * (roll_desired - uav.attitude[0])
    uav.attitude[1] += config.beta_rp * (pitch_desired - uav.attitude[1])
    uav.attitude[2] += config.beta_yaw * wrap_to_pi(
        yaw_desired - uav.attitude[2]
    )
    # 数值安全限幅；理论上凸组合已经不会超过期望角上限。
    uav.attitude[0] = np.clip(
        uav.attitude[0], -config.max_roll, config.max_roll
    )
    uav.attitude[1] = np.clip(
        uav.attitude[1], -config.max_pitch, config.max_pitch
    )

    omega_base = config.omega_hover_visual + config.k_rotor_z * velocity_command[2]
    roll_error = roll_desired - uav.attitude[0]
    pitch_error = pitch_desired - uav.attitude[1]
    # 顺序对应四个 X 形臂端点：[前左、后左、后右、前右]。
    # 该简化差动混控仅服务视觉效果，不产生推力，也不反作用于任何状态。
    roll_pattern = np.array([-1.0, -1.0, 1.0, 1.0])
    pitch_pattern = np.array([-1.0, 1.0, 1.0, -1.0])
    visual_mixing = config.k_rotor_attitude_visual * (
        roll_error * roll_pattern + pitch_error * pitch_pattern
    )
    uav.rotor_speeds = np.clip(
        omega_base + visual_mixing,
        config.omega_min_visual,
        config.omega_max_visual,
    )
    uav.rotor_phases = np.mod(
        uav.rotor_phases
        + config.rotor_directions * uav.rotor_speeds * config.dt,
        2.0 * np.pi,
    )


def _formation_state(
    uavs: list[UAV], config: SimulationConfig
) -> tuple[FloatArray, FloatArray]:
    """按当前领航者状态计算期望位置和四架跟随者误差。"""
    leader = uavs[0]
    leader_yaw, _ = yaw_from_velocity(
        leader.velocity, leader.last_valid_yaw, config.horizontal_speed_epsilon
    )
    desired = desired_formation_positions(
        leader.position, leader_yaw, config.formation_offsets
    )
    positions = np.stack([uav.position for uav in uavs])
    errors = np.linalg.norm(positions[1:] - desired[1:], axis=1)
    return desired, errors


def _record_state(
    uavs: list[UAV],
    config: SimulationConfig,
    time: float,
    storage: dict[str, list[FloatArray] | list[float]],
) -> None:
    """把同一时刻的所有状态压入列表。"""
    desired, errors = _formation_state(uavs, config)
    storage["times"].append(float(time))
    storage["positions"].append(np.stack([uav.position.copy() for uav in uavs]))
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


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """运行一次无障碍三维编队形成与保持实验。"""
    uavs = create_uavs(config)
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
    }
    _record_state(uavs, config, 0.0, storage)

    completed = False
    completion_time: float | None = None
    maximum_steps = int(np.ceil(config.max_time / config.dt))

    for step in range(1, maximum_steps + 1):
        commands, _, _ = compute_velocity_commands(uavs, config)

        # 严格使用 p(t+dt)=p(t)+v(t)*dt；旋翼转速从未出现在此处。
        old_positions = [uav.position.copy() for uav in uavs]
        old_velocities = [uav.velocity.copy() for uav in uavs]
        for index, uav in enumerate(uavs):
            _update_visual_attitude_and_rotors(uav, commands[index], config)
            uav.position = old_positions[index] + old_velocities[index] * config.dt
            smoothed_velocity = (
                (1.0 - config.alpha) * old_velocities[index]
                + config.alpha * commands[index]
            )
            uav.velocity = limit_vector_norm(smoothed_velocity, config.v_max)
            uav.append_history()

        current_time = step * config.dt
        _record_state(uavs, config, current_time, storage)

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
        key: np.asarray(value, dtype=float) for key, value in storage.items()
    }
    metrics = _compute_metrics(arrays, completed, completion_time, config)
    validation = _validate_histories(arrays, completed, config)

    return SimulationResult(
        config=config,
        times=arrays["times"],
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        attitudes=arrays["attitudes"],
        desired_attitudes=arrays["desired_attitudes"],
        rotor_speeds=arrays["rotor_speeds"],
        rotor_phases=arrays["rotor_phases"],
        desired_positions=arrays["desired_positions"],
        formation_errors=arrays["formation_errors"],
        completed=completed,
        completion_time=completion_time,
        metrics=metrics,
        validation=validation,
    )


def _compute_metrics(
    arrays: dict[str, FloatArray],
    completed: bool,
    completion_time: float | None,
    config: SimulationConfig,
) -> dict[str, Any]:
    """计算实验要求的标量指标。"""
    positions = arrays["positions"]
    attitudes = arrays["attitudes"]
    errors = arrays["formation_errors"]
    speeds = arrays["rotor_speeds"]

    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    flight_distances = np.sum(segment_lengths, axis=0)

    if len(attitudes) > 1:
        attitude_rate = np.diff(attitudes, axis=0) / config.dt
        # yaw 保持为连续内部角，不需在这里对差分做额外的 2pi 展开。
        attitude_change_rms = float(np.sqrt(np.mean(attitude_rate**2)))
    else:
        attitude_change_rms = 0.0

    return {
        "reached_goal_and_formation": bool(completed),
        "completion_time_s": None if completion_time is None else float(completion_time),
        "mean_formation_error_m": float(np.mean(errors)),
        "max_formation_error_m": float(np.max(errors)),
        "final_formation_error_m": float(np.mean(errors[-1])),
        "flight_distance_per_uav_m": {
            f"uav_{index}": float(distance)
            for index, distance in enumerate(flight_distances)
        },
        "total_flight_distance_m": float(np.sum(flight_distances)),
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(attitudes[:, :, 0])))),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(attitudes[:, :, 1])))),
        "attitude_change_rms_deg_per_s": float(np.rad2deg(attitude_change_rms)),
        "visual_rotor_speed_range_rad_per_s": [
            float(np.min(speeds)),
            float(np.max(speeds)),
        ],
        "max_actual_speed_m_per_s": float(
            np.max(np.linalg.norm(arrays["velocities"], axis=2))
        ),
        "final_leader_goal_distance_m": float(
            np.linalg.norm(positions[-1, 0] - config.goal)
        ),
    }


def _validate_histories(
    arrays: dict[str, FloatArray], completed: bool, config: SimulationConfig
) -> dict[str, Any]:
    """执行数组、限幅、角度环绕和旋翼相位等自动验收检查。"""
    finite_keys = (
        "positions",
        "velocities",
        "attitudes",
        "desired_attitudes",
        "rotor_speeds",
        "rotor_phases",
    )
    all_finite = all(np.all(np.isfinite(arrays[key])) for key in finite_keys)
    velocity_norms = np.linalg.norm(arrays["velocities"], axis=2)
    attitudes = arrays["attitudes"]
    rotor_speeds = arrays["rotor_speeds"]
    rotor_phases = arrays["rotor_phases"]

    if len(rotor_phases) > 1:
        actual_phase_step = wrap_to_pi(np.diff(rotor_phases, axis=0))
        expected_phase_step = (
            config.rotor_directions[None, None, :]
            * rotor_speeds[1:]
            * config.dt
        )
        phase_error = wrap_to_pi(actual_phase_step - expected_phase_step)
        phase_update_ok = bool(np.max(np.abs(phase_error)) < 1.0e-9)
    else:
        phase_update_ok = True

    yaw_steps = np.diff(attitudes[:, :, 2], axis=0)
    maximum_yaw_step = float(np.max(np.abs(yaw_steps))) if yaw_steps.size else 0.0
    adjacent_opposite = bool(
        np.all(config.rotor_directions[:-1] * config.rotor_directions[1:] < 0.0)
    )

    return {
        "goal_and_formation_reached": bool(completed),
        "all_states_finite": bool(all_finite),
        "speed_within_limit": bool(np.max(velocity_norms) <= config.v_max + 1.0e-10),
        "roll_within_limit": bool(
            np.max(np.abs(attitudes[:, :, 0])) <= config.max_roll + 1.0e-10
        ),
        "pitch_within_limit": bool(
            np.max(np.abs(attitudes[:, :, 1])) <= config.max_pitch + 1.0e-10
        ),
        "yaw_has_no_large_reverse_jump": bool(maximum_yaw_step < np.pi),
        "maximum_yaw_step_deg": float(np.rad2deg(maximum_yaw_step)),
        "rotor_speeds_within_visual_limits": bool(
            np.min(rotor_speeds) >= config.omega_min_visual - 1.0e-10
            and np.max(rotor_speeds) <= config.omega_max_visual + 1.0e-10
        ),
        "rotor_phases_in_zero_two_pi": bool(
            np.min(rotor_phases) >= 0.0
            and np.max(rotor_phases) < 2.0 * np.pi + 1.0e-12
        ),
        "rotor_phase_update_matches_direction": phase_update_ok,
        "adjacent_rotors_have_opposite_directions": adjacent_opposite,
        "final_mean_formation_error_below_1m": bool(
            np.mean(arrays["formation_errors"][-1]) < 1.0
        ),
        "position_update_uses_rotor_thrust": False,
        "model_is_quadcopter_flight_dynamics": False,
    }


def verify_visual_state_independence(
    baseline: SimulationResult,
) -> tuple[bool, float]:
    """改变姿态/旋翼显示参数并复算，确认位置轨迹完全不受影响。"""
    altered_config = replace(
        baseline.config,
        beta_rp=0.72,
        beta_yaw=0.61,
        omega_hover_visual=30.0,
        omega_min_visual=5.0,
        omega_max_visual=120.0,
        k_rotor_z=7.0,
        k_rotor_attitude_visual=20.0,
    )
    altered = run_simulation(altered_config)
    if altered.positions.shape != baseline.positions.shape:
        return False, float("inf")
    maximum_difference = float(np.max(np.abs(altered.positions - baseline.positions)))
    return bool(maximum_difference <= 1.0e-12), maximum_difference

