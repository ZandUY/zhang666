"""实验一和实验二的集中参数配置。

本项目采用三维一阶运动学模型。姿态和旋翼均为便于观察的视觉状态，
不会参与位置或速度的计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from models import MovingSphereObstacle, SphereObstacle


@dataclass(slots=True)
class SimulationConfig:
    """仿真、显示和实验输出参数。"""

    # 仿真时间与一阶运动学参数
    dt: float = 0.05
    max_time: float = 100.0
    v_max: float = 3.0
    alpha: float = 0.2

    # 位置控制增益
    k_goal: float = 0.5
    k_form: float = 0.6
    k_vel: float = 0.3

    # 仅用于显示的姿态跟随参数
    tau_v: float = 0.5
    g: float = 9.81
    max_roll: float = np.deg2rad(25.0)
    max_pitch: float = np.deg2rad(25.0)
    beta_rp: float = 0.15
    beta_yaw: float = 0.10
    horizontal_speed_epsilon: float = 1.0e-3

    # 仅用于视觉动画的旋翼参数，绝不进入位置、速度或姿态更新方程
    omega_hover_visual: float = 45.0
    omega_min_visual: float = 20.0
    omega_max_visual: float = 80.0
    k_rotor_z: float = 2.0
    k_rotor_attitude_visual: float = 9.0
    rotor_directions: np.ndarray = field(
        default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    )

    # 场景与结束条件
    random_seed: int = 42
    leader_initial_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 12.0], dtype=float)
    )
    goal: np.ndarray = field(
        default_factory=lambda: np.array([50.0, 0.0, 12.0], dtype=float)
    )
    formation_offsets: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.0, 0.0],
                [-5.0, 3.0, 2.0],
                [-5.0, -3.0, 2.0],
                [-5.0, 3.0, -2.0],
                [-5.0, -3.0, -2.0],
            ],
            dtype=float,
        )
    )
    follower_initial_perturbation: float = 2.0
    goal_tolerance: float = 1.0
    formation_tolerance: float = 1.0

    # 四旋翼几何参数（十字臂呈 X 形）
    arm_length: float = 0.75
    rotor_radius: float = 0.28

    # 固定坐标范围避免 GIF 播放时视角抖动
    x_limits: tuple[float, float] = (-10.0, 56.0)
    y_limits: tuple[float, float] = (-12.0, 12.0)
    z_limits: tuple[float, float] = (5.0, 20.0)
    animation_stride: int = 4
    animation_fps: int = 15

    # 输出
    results_dir: Path = Path("results") / "experiment_1"

    def __post_init__(self) -> None:
        """统一数组类型并检查最重要的参数约束。"""
        self.leader_initial_position = np.asarray(
            self.leader_initial_position, dtype=float
        ).copy()
        self.goal = np.asarray(self.goal, dtype=float).copy()
        self.formation_offsets = np.asarray(
            self.formation_offsets, dtype=float
        ).copy()
        self.rotor_directions = np.asarray(
            self.rotor_directions, dtype=float
        ).copy()

        if self.leader_initial_position.shape != (3,):
            raise ValueError("leader_initial_position 必须为 shape=(3,)")
        if self.goal.shape != (3,):
            raise ValueError("goal 必须为 shape=(3,)")
        if self.formation_offsets.shape != (5, 3):
            raise ValueError("formation_offsets 必须为 shape=(5, 3)")
        if self.rotor_directions.shape != (4,):
            raise ValueError("rotor_directions 必须为 shape=(4,)")
        if self.dt <= 0.0 or self.max_time <= 0.0:
            raise ValueError("dt 和 max_time 必须为正数")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha 必须位于 (0, 1]")
        if self.v_max <= 0.0:
            raise ValueError("v_max 必须为正数")
        if self.animation_stride < 1 or self.animation_fps < 1:
            raise ValueError("动画步长和帧率必须为正整数")


EXPERIMENT_K_FORM_VALUES: tuple[float, ...] = (0.3, 0.6, 1.0)


@dataclass(slots=True)
class Experiment2Config(SimulationConfig):
    """实验二的球形障碍物、人工势场和防碰撞参数。"""

    obstacles: list[SphereObstacle] = field(
        default_factory=lambda: [
            SphereObstacle(
                obstacle_id=0,
                center=np.array([25.0, 0.0, 12.0], dtype=float),
                radius=4.0,
                influence_distance=6.0,
            )
        ]
    )
    uav_radius: float = 0.3
    k_obs: float = 8.0
    max_obstacle_force: float = 5.0
    distance_epsilon: float = 0.05

    # 题目建议值为 2.0 m；本实验楔形编队在绕障时自然最小间距约 2.95 m，
    # 因此采用 4.0 m 作为提前干预距离，使方法 C 的分离项能够实际参与对比。
    # 碰撞判据仍严格使用 2*uav_radius=0.6 m。
    d_safe: float = 4.0
    k_sep: float = 3.0
    max_separation_force: float = 4.0

    enable_obstacle_avoidance: bool = True
    enable_separation: bool = True
    method_name: str = "C_apf_and_separation"

    # 障碍物位于起点—终点连线的严格对称轴上。所有对比方法使用同一个很小的
    # 确定性初速度来打破纯径向人工势场的数学对称性；初始位置保持完全一致。
    initial_velocity_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.05, 0.0], dtype=float)
    )
    results_dir: Path = Path("results") / "experiment_2"

    def __post_init__(self) -> None:
        """调用公共配置检查，并复制实验二可变对象。"""
        SimulationConfig.__post_init__(self)
        self.initial_velocity_bias = np.asarray(
            self.initial_velocity_bias, dtype=float
        ).copy()
        if self.initial_velocity_bias.shape != (3,):
            raise ValueError("initial_velocity_bias 必须为 shape=(3,)")
        self.obstacles = [
            SphereObstacle(
                obstacle_id=item.obstacle_id,
                center=item.center.copy(),
                radius=item.radius,
                influence_distance=item.influence_distance,
            )
            for item in self.obstacles
        ]
        if self.uav_radius <= 0.0:
            raise ValueError("uav_radius 必须为正数")
        if self.k_obs < 0.0 or self.max_obstacle_force <= 0.0:
            raise ValueError("障碍物排斥参数无效")
        if self.d_safe <= 2.0 * self.uav_radius:
            raise ValueError("d_safe 必须大于两倍无人机等效半径")
        if self.k_sep < 0.0 or self.max_separation_force <= 0.0:
            raise ValueError("机间排斥参数无效")
        if self.distance_epsilon <= 0.0:
            raise ValueError("distance_epsilon 必须为正数")


@dataclass(slots=True)
class Experiment3Config(Experiment2Config):
    """实验三的多障碍物、局部极小检测和切向逃逸参数。"""

    obstacles: list[SphereObstacle] = field(
        default_factory=lambda: [
            SphereObstacle(1, np.array([20.0, -4.0, 12.0]), 3.5, 6.0),
            SphereObstacle(2, np.array([20.0, 4.0, 12.0]), 3.5, 6.0),
            SphereObstacle(3, np.array([27.0, 0.0, 9.0]), 3.0, 6.0),
            SphereObstacle(4, np.array([27.0, 0.0, 15.0]), 3.0, 6.0),
        ]
    )
    improved_apf_enabled: bool = False
    method_name: str = "standard_apf"

    # 默认实验二参数 k_obs=8、最大排斥=5 时，标准法最终会被浮点级微小
    # 不对称带离局部极小。实验三提高势场强度以稳定复现该问题；障碍物位置
    # 和半径仍严格采用题定四球场景，最终参数也会写入结果配置文件。
    k_obs: float = 20.0
    max_obstacle_force: float = 8.0

    # 实验三使用零初速度，不继承实验二单球中心线场景的破对称速度；这既避免
    # 初始航向被误设为 +y，也让标准 APF 的对称局部极小由仿真本身真实呈现。
    initial_velocity_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    local_min_history_steps: int = 20
    local_min_distance_threshold: float = 2.0
    local_min_speed_threshold: float = 0.15
    local_min_improvement_threshold: float = 0.05
    local_min_consecutive_steps: int = 5

    k_escape: float = 1.0
    escape_max_duration: float = 2.0
    escape_exit_improvement: float = 0.20
    escape_min_active_steps: int = 5
    escape_cooldown: float = 0.50

    results_dir: Path = Path("results") / "experiment_3"

    def __post_init__(self) -> None:
        """检查局部极小和切向逃逸参数。"""
        Experiment2Config.__post_init__(self)
        if self.local_min_history_steps < 2:
            raise ValueError("local_min_history_steps 必须至少为 2")
        if self.local_min_consecutive_steps < 2:
            raise ValueError("局部极小条件必须连续满足至少 2 步")
        if self.local_min_distance_threshold <= 0.0:
            raise ValueError("局部极小距离阈值必须为正数")
        if self.local_min_speed_threshold <= 0.0:
            raise ValueError("局部极小速度阈值必须为正数")
        if self.local_min_improvement_threshold < 0.0:
            raise ValueError("距离改善阈值不能为负数")
        if self.k_escape <= 0.0 or self.escape_max_duration <= 0.0:
            raise ValueError("切向逃逸参数必须为正数")
        if self.escape_min_active_steps < 1 or self.escape_cooldown < 0.0:
            raise ValueError("逃逸最短步数或冷却时间无效")


EXPERIMENT3_RANDOM_SEEDS: tuple[int, ...] = (
    3,
    7,
    11,
    19,
    23,
    31,
    42,
    57,
    73,
    91,
)
EXPERIMENT3_REPRESENTATIVE_SEED: int = 42


@dataclass(slots=True)
class Experiment4Config(Experiment2Config):
    """实验四的移动障碍物和预测位置避障配置。"""

    # 实验四不使用实验二的静态障碍物。
    obstacles: list[SphereObstacle] = field(default_factory=list)
    moving_obstacles: list[MovingSphereObstacle] = field(
        default_factory=lambda: [
            MovingSphereObstacle(
                obstacle_id=0,
                center=np.array([25.0, -12.0, 12.0], dtype=float),
                radius=2.0,
                velocity=np.array([0.0, 1.0, 0.0], dtype=float),
                influence_distance=6.0,
            )
        ]
    )
    enable_prediction: bool = False
    prediction_time: float = 1.0
    method_name: str = "current_position"

    # 动态横穿场景无需实验二的中心线破对称初速度。
    initial_velocity_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    formation_recovery_error: float = 0.8
    formation_recovery_hold_time: float = 2.0
    post_completion_observation_time: float = 10.0
    results_dir: Path = Path("results") / "experiment_4"

    def __post_init__(self) -> None:
        """检查预测参数，并深复制移动障碍物初始状态。"""
        Experiment2Config.__post_init__(self)
        self.moving_obstacles = [
            MovingSphereObstacle(
                obstacle_id=obstacle.obstacle_id,
                center=obstacle.center.copy(),
                radius=obstacle.radius,
                velocity=obstacle.velocity.copy(),
                influence_distance=obstacle.influence_distance,
            )
            for obstacle in self.moving_obstacles
        ]
        if self.prediction_time < 0.0:
            raise ValueError("prediction_time 不能为负数")
        if self.formation_recovery_error <= 0.0:
            raise ValueError("编队恢复误差阈值必须为正数")
        if self.formation_recovery_hold_time <= 0.0:
            raise ValueError("编队恢复保持时间必须为正数")
        if self.post_completion_observation_time < self.formation_recovery_hold_time:
            raise ValueError("完成后观察时间不能短于恢复保持时间")


EXPERIMENT4_OBSTACLE_SPEEDS: tuple[float, ...] = (0.5, 1.0, 1.5)
EXPERIMENT4_RANDOM_SEEDS: tuple[int, ...] = (7, 19, 42, 73, 91)
EXPERIMENT4_REPRESENTATIVE_SPEED: float = 1.0
EXPERIMENT4_REPRESENTATIVE_SEED: int = 42
