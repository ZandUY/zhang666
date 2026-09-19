"""编队控制、角度工具和向量限幅。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import Experiment2Config, Experiment4Config, SimulationConfig
from models import MovingSphereObstacle, SphereObstacle, UAV


def wrap_to_pi(angle: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """把角度或角度数组环绕到 [-pi, pi]。

    该函数用于偏航误差，避免跨越正负 180 度时沿错误方向大角度旋转。
    """
    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def limit_vector_norm(vector: NDArray[np.float64], max_norm: float) -> NDArray[np.float64]:
    """按向量模长限幅，保持原方向不变。"""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > max_norm and norm > 0.0:
        return vector * (max_norm / norm)
    return vector.copy()


def rotation_z(yaw: float) -> NDArray[np.float64]:
    """返回绕世界 z 轴旋转的 3x3 矩阵。"""
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def yaw_from_velocity(
    velocity: NDArray[np.float64],
    last_valid_yaw: float,
    epsilon: float,
) -> tuple[float, bool]:
    """由水平速度计算航向；速度过小时返回上一个有效航向。"""
    if float(np.linalg.norm(velocity[:2])) > epsilon:
        return float(np.arctan2(velocity[1], velocity[0])), True
    return float(last_valid_yaw), False


def desired_formation_positions(
    leader_position: NDArray[np.float64],
    leader_yaw: float,
    offsets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """将局部编队偏移绕 z 轴旋转后平移到领航者处。"""
    return leader_position[None, :] + (rotation_z(leader_yaw) @ offsets.T).T


def compute_velocity_commands(
    uavs: list[UAV], config: SimulationConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """计算领航者与跟随者的速度命令。

    返回值依次为 shape=(5, 3) 的限幅速度命令、期望编队位置，以及领航者
    当前有效航向。命令和实际速度均按模长限幅，而非逐轴截断。
    """
    if len(uavs) != 5:
        raise ValueError("实验一固定使用 5 架无人机")

    leader = uavs[0]
    leader_yaw, yaw_is_valid = yaw_from_velocity(
        leader.velocity, leader.last_valid_yaw, config.horizontal_speed_epsilon
    )
    if yaw_is_valid:
        leader.last_valid_yaw = leader_yaw

    desired_positions = desired_formation_positions(
        leader.position, leader_yaw, config.formation_offsets
    )
    commands = np.zeros((len(uavs), 3), dtype=float)

    commands[0] = limit_vector_norm(
        config.k_goal * (config.goal - leader.position), config.v_max
    )
    for index, follower in enumerate(uavs[1:], start=1):
        position_term = config.k_form * (
            desired_positions[index] - follower.position
        )
        velocity_term = config.k_vel * (leader.velocity - follower.velocity)
        commands[index] = limit_vector_norm(
            position_term + velocity_term, config.v_max
        )

    return commands, desired_positions, leader_yaw


def _deterministic_unit_direction(first_id: int, second_id: int) -> NDArray[np.float64]:
    """为完全重合的两个编号生成确定、反对称的单位方向。"""
    low_id, high_id = sorted((int(first_id), int(second_id)))
    # 无随机数；不同编号对会得到不同但可复现的三维方向。
    raw = np.array(
        [
            1.0 + ((low_id + high_id) % 3),
            -1.0 if (low_id + high_id) % 2 else 1.0,
            0.5 + ((2 * low_id + high_id) % 2),
        ],
        dtype=float,
    )
    direction = raw / np.linalg.norm(raw)
    return direction if first_id == low_id else -direction


def obstacle_repulsion_force(
    position: NDArray[np.float64],
    obstacles: list[SphereObstacle],
    uav_radius: float,
    k_obs: float,
    max_force: float,
    epsilon: float,
    uav_id: int = 0,
) -> NDArray[np.float64]:
    """计算多个球形障碍物的合成排斥速度分量。

    每个障碍物使用题定公式，再对合力按模长限幅。影响范围外严格返回
    ``shape=(3,)`` 的零向量。``rho`` 和方向分母都具有 ``epsilon`` 下限。
    """
    position = np.asarray(position, dtype=float)
    total_force = np.zeros(3, dtype=float)
    for obstacle in obstacles:
        delta = position - obstacle.center
        center_distance = float(np.linalg.norm(delta))
        rho = center_distance - obstacle.radius - uav_radius
        if rho >= obstacle.influence_distance:
            continue

        if center_distance > epsilon:
            direction = delta / max(center_distance, epsilon)
        else:
            # 球心处径向方向没有定义，使用编号生成可复现方向以避免零向量/NaN。
            direction = _deterministic_unit_direction(
                uav_id, 10_000 + obstacle.obstacle_id
            )
        rho_safe = max(rho, epsilon)
        magnitude = (
            k_obs
            * (1.0 / rho_safe - 1.0 / obstacle.influence_distance)
            * (1.0 / rho_safe**2)
        )
        total_force += magnitude * direction

    return limit_vector_norm(total_force, max_force)


def separation_forces(
    uavs: list[UAV],
    d_safe: float,
    k_sep: float,
    max_force: float,
    epsilon: float,
) -> NDArray[np.float64]:
    """一次计算所有无人机的成对防碰撞排斥分量。

    对每个无序对只计算一次并施加大小相等、方向相反的分量。若位置完全
    重合，则按无人机编号选择确定性方向，不使用随机数。
    """
    forces = np.zeros((len(uavs), 3), dtype=float)
    for first in range(len(uavs) - 1):
        for second in range(first + 1, len(uavs)):
            delta = uavs[first].position - uavs[second].position
            distance = float(np.linalg.norm(delta))
            if distance >= d_safe:
                continue
            if distance > epsilon:
                direction = delta / max(distance, epsilon)
            else:
                direction = _deterministic_unit_direction(
                    uavs[first].id, uavs[second].id
                )
            distance_safe = max(distance, epsilon)
            magnitude = (
                k_sep
                * (1.0 / distance_safe - 1.0 / d_safe)
                * (1.0 / distance_safe**2)
            )
            pair_force = magnitude * direction
            forces[first] += pair_force
            forces[second] -= pair_force

    for index in range(len(uavs)):
        forces[index] = limit_vector_norm(forces[index], max_force)
    return forces


def compute_experiment2_velocity_commands(
    uavs: list[UAV], config: Experiment2Config
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """计算实验二速度命令，并保留各控制分量便于检查。

    人工势场仅生成期望速度分量。全部分量相加后统一按 ``v_max`` 模长
    限幅；随后仍由仿真器执行与实验一相同的速度平滑。
    """
    if len(uavs) != 5:
        raise ValueError("实验二固定使用 5 架无人机")

    leader = uavs[0]
    leader_yaw, yaw_is_valid = yaw_from_velocity(
        leader.velocity, leader.last_valid_yaw, config.horizontal_speed_epsilon
    )
    if yaw_is_valid:
        leader.last_valid_yaw = leader_yaw
    desired_positions = desired_formation_positions(
        leader.position, leader_yaw, config.formation_offsets
    )

    nominal = np.zeros((len(uavs), 3), dtype=float)
    nominal[0] = limit_vector_norm(
        config.k_goal * (config.goal - leader.position), config.v_max
    )
    for index, follower in enumerate(uavs[1:], start=1):
        formation = config.k_form * (desired_positions[index] - follower.position)
        velocity_match = config.k_vel * (leader.velocity - follower.velocity)
        nominal[index] = limit_vector_norm(
            formation + velocity_match, config.v_max
        )

    obstacle_forces = np.zeros_like(nominal)
    if config.enable_obstacle_avoidance:
        for index, uav in enumerate(uavs):
            obstacle_forces[index] = obstacle_repulsion_force(
                uav.position,
                config.obstacles,
                config.uav_radius,
                config.k_obs,
                config.max_obstacle_force,
                config.distance_epsilon,
                uav.id,
            )

    pair_forces = np.zeros_like(nominal)
    if config.enable_separation:
        pair_forces = separation_forces(
            uavs,
            config.d_safe,
            config.k_sep,
            config.max_separation_force,
            config.distance_epsilon,
        )

    commands = nominal + obstacle_forces + pair_forces
    for index in range(len(uavs)):
        commands[index] = limit_vector_norm(commands[index], config.v_max)
    components = {
        "nominal": nominal,
        "obstacle": obstacle_forces,
        "separation": pair_forces,
    }
    return commands, desired_positions, components


def nearest_obstacle(
    position: NDArray[np.float64], obstacles: list[SphereObstacle]
) -> SphereObstacle | None:
    """按到球面的净距离返回最近障碍物。"""
    if not obstacles:
        return None
    position = np.asarray(position, dtype=float)
    return min(
        obstacles,
        key=lambda obstacle: float(np.linalg.norm(position - obstacle.center))
        - obstacle.radius,
    )


def tangential_escape_force(
    position: NDArray[np.float64],
    obstacle: SphereObstacle,
    k_escape: float,
    epsilon: float,
    uav_id: int,
    leader_direction: NDArray[np.float64] | None = None,
    preferred_sign: float = 1.0,
) -> NDArray[np.float64]:
    """计算归一化切向逃逸速度分量。

    先按 ``cross(n, [0,0,1])`` 计算，退化时改用 ``cross(n, [1,0,0])``。
    跟随者选择与领航者参考绕行方向最一致的符号；几何关系正交时再使用
    UAV 编号确定符号，从而完全可复现且不依赖随机方向。
    """
    delta = np.asarray(position, dtype=float) - obstacle.center
    distance = float(np.linalg.norm(delta))
    if distance > epsilon:
        radial = delta / max(distance, epsilon)
    else:
        radial = _deterministic_unit_direction(
            int(uav_id), 20_000 + obstacle.obstacle_id
        )

    tangent = np.cross(radial, np.array([0.0, 0.0, 1.0]))
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= epsilon:
        tangent = np.cross(radial, np.array([1.0, 0.0, 0.0]))
        tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= epsilon:
        # 理论上两个参考轴不可能同时退化；此分支只作最后的数值保护。
        tangent = _deterministic_unit_direction(uav_id, obstacle.obstacle_id)
        tangent_norm = float(np.linalg.norm(tangent))
    tangent = tangent / tangent_norm

    if leader_direction is not None:
        reference = np.asarray(leader_direction, dtype=float)
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm > epsilon:
            alignment = float(np.dot(tangent, reference / reference_norm))
            if alignment < -epsilon:
                tangent = -tangent
            elif abs(alignment) <= epsilon and uav_id % 2 == 1:
                tangent = -tangent
    elif preferred_sign < 0.0:
        tangent = -tangent

    return float(k_escape) * tangent


def moving_obstacle_repulsion_force(
    position: NDArray[np.float64],
    obstacles: list[MovingSphereObstacle],
    uav_radius: float,
    k_obs: float,
    max_force: float,
    epsilon: float,
    enable_prediction: bool,
    prediction_time: float,
    uav_id: int = 0,
) -> NDArray[np.float64]:
    """使用当前或预测中心计算移动球障碍物的 APF 排斥分量。

    预测中心只存在于本控制计算中，不写回障碍物对象。净距离仍按预测球面
    与无人机等效半径计算，多障碍物合力最后按模长限幅。
    """
    position = np.asarray(position, dtype=float)
    total_force = np.zeros(3, dtype=float)
    for obstacle in obstacles:
        predicted_center = obstacle.center.copy()
        if enable_prediction:
            predicted_center += float(prediction_time) * obstacle.velocity
        delta = position - predicted_center
        center_distance = float(np.linalg.norm(delta))
        rho = center_distance - obstacle.radius - uav_radius
        if rho >= obstacle.influence_distance:
            continue
        if center_distance > epsilon:
            direction = delta / max(center_distance, epsilon)
        else:
            direction = _deterministic_unit_direction(
                uav_id, 30_000 + obstacle.obstacle_id
            )
        rho_safe = max(rho, epsilon)
        magnitude = (
            k_obs
            * (1.0 / rho_safe - 1.0 / obstacle.influence_distance)
            * (1.0 / rho_safe**2)
        )
        total_force += magnitude * direction
    return limit_vector_norm(total_force, max_force)


def compute_experiment4_velocity_commands(
    uavs: list[UAV],
    moving_obstacles: list[MovingSphereObstacle],
    config: Experiment4Config,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """计算实验四当前位置/预测位置 APF 速度命令。"""
    if len(uavs) != 5:
        raise ValueError("实验四固定使用 5 架无人机")
    leader = uavs[0]
    leader_yaw, yaw_is_valid = yaw_from_velocity(
        leader.velocity, leader.last_valid_yaw, config.horizontal_speed_epsilon
    )
    if yaw_is_valid:
        leader.last_valid_yaw = leader_yaw
    desired_positions = desired_formation_positions(
        leader.position, leader_yaw, config.formation_offsets
    )

    nominal = np.zeros((len(uavs), 3), dtype=float)
    nominal[0] = limit_vector_norm(
        config.k_goal * (config.goal - leader.position), config.v_max
    )
    for index, follower in enumerate(uavs[1:], start=1):
        formation = config.k_form * (desired_positions[index] - follower.position)
        velocity_match = config.k_vel * (leader.velocity - follower.velocity)
        nominal[index] = limit_vector_norm(
            formation + velocity_match, config.v_max
        )

    obstacle_forces = np.zeros_like(nominal)
    for index, uav in enumerate(uavs):
        obstacle_forces[index] = moving_obstacle_repulsion_force(
            uav.position,
            moving_obstacles,
            config.uav_radius,
            config.k_obs,
            config.max_obstacle_force,
            config.distance_epsilon,
            config.enable_prediction,
            config.prediction_time,
            uav.id,
        )
    pair_forces = np.zeros_like(nominal)
    if config.enable_separation:
        pair_forces = separation_forces(
            uavs,
            config.d_safe,
            config.k_sep,
            config.max_separation_force,
            config.distance_epsilon,
        )
    commands = nominal + obstacle_forces + pair_forces
    for index in range(len(uavs)):
        commands[index] = limit_vector_norm(commands[index], config.v_max)
    return commands, desired_positions, {
        "nominal": nominal,
        "obstacle": obstacle_forces,
        "separation": pair_forces,
    }
