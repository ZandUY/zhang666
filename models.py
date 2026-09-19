"""无人机数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]


def _zero_vector3() -> Vector:
    return np.zeros(3, dtype=float)


def _zero_vector4() -> Vector:
    return np.zeros(4, dtype=float)


@dataclass(slots=True)
class UAV:
    """单架无人机的状态。

    注意：这里的位置部分是一阶运动学模型。``rotor_speeds`` 与
    ``rotor_phases`` 只是视觉动画状态，不代表真实电机或旋翼动力学。
    """

    id: int
    position: Vector
    velocity: Vector = field(default_factory=_zero_vector3)
    attitude: Vector = field(default_factory=_zero_vector3)  # roll, pitch, yaw
    desired_attitude: Vector = field(default_factory=_zero_vector3)
    last_valid_yaw: float = 0.0
    rotor_speeds: Vector = field(default_factory=_zero_vector4)
    rotor_phases: Vector = field(default_factory=_zero_vector4)
    history: list[Vector] = field(default_factory=list)
    is_leader: bool = False

    def __post_init__(self) -> None:
        """复制输入数组，防止多个对象意外共享可变状态。"""
        self.position = self._as_vector(self.position, 3, "position")
        self.velocity = self._as_vector(self.velocity, 3, "velocity")
        self.attitude = self._as_vector(self.attitude, 3, "attitude")
        self.desired_attitude = self._as_vector(
            self.desired_attitude, 3, "desired_attitude"
        )
        self.rotor_speeds = self._as_vector(
            self.rotor_speeds, 4, "rotor_speeds"
        )
        self.rotor_phases = self._as_vector(
            self.rotor_phases, 4, "rotor_phases"
        )
        self.history = [np.asarray(point, dtype=float).copy() for point in self.history]

    @staticmethod
    def _as_vector(value: Vector, size: int, name: str) -> Vector:
        array = np.asarray(value, dtype=float).copy()
        if array.shape != (size,):
            raise ValueError(f"{name} 必须为 shape=({size},)，实际为 {array.shape}")
        return array

    def append_history(self) -> None:
        """保存当前位置的独立副本。"""
        self.history.append(self.position.copy())


@dataclass(slots=True)
class SphereObstacle:
    """球形静态障碍物。

    ``influence_distance`` 是从障碍物表面向外量取的人工势场影响距离。
    """

    obstacle_id: int
    center: Vector
    radius: float
    influence_distance: float

    def __post_init__(self) -> None:
        """规范中心坐标并检查几何参数。"""
        self.center = UAV._as_vector(self.center, 3, "center")
        if self.radius <= 0.0:
            raise ValueError("障碍物半径必须为正数")
        if self.influence_distance <= 0.0:
            raise ValueError("障碍物影响距离必须为正数")


@dataclass(slots=True)
class MovingSphereObstacle:
    """具有恒定平移速度的球形移动障碍物。"""

    obstacle_id: int
    center: Vector
    radius: float
    velocity: Vector
    influence_distance: float
    history: list[Vector] = field(default_factory=list)

    def __post_init__(self) -> None:
        """复制几何状态，并把初始中心写入历史。"""
        self.center = UAV._as_vector(self.center, 3, "center")
        self.velocity = UAV._as_vector(self.velocity, 3, "velocity")
        if self.radius <= 0.0:
            raise ValueError("移动障碍物半径必须为正数")
        if self.influence_distance <= 0.0:
            raise ValueError("移动障碍物影响距离必须为正数")
        if self.history:
            self.history = [
                UAV._as_vector(point, 3, "history point") for point in self.history
            ]
        else:
            self.history = [self.center.copy()]

    def update(self, dt: float) -> None:
        """按 ``center = center + velocity * dt`` 更新真实位置。"""
        self.center = self.center + self.velocity * float(dt)
        self.history.append(self.center.copy())
