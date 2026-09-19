"""实验一绘图与 GIF 动画。

本模块只消费已经计算完毕的 ``SimulationResult``，不会改变或反馈仿真状态。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 注册 3D 投影

from models import SphereObstacle
from simulator import SimulationResult


UAV_COLORS = ("red", "tab:blue", "tab:green", "tab:orange", "tab:purple")


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """构造 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _configure_3d_axes(axis: plt.Axes, result: SimulationResult) -> None:
    """设置固定范围、标签和近似一致的三轴比例。"""
    config = result.config
    axis.set_xlim(*config.x_limits)
    axis.set_ylim(*config.y_limits)
    axis.set_zlim(*config.z_limits)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_box_aspect(
        (
            config.x_limits[1] - config.x_limits[0],
            config.y_limits[1] - config.y_limits[0],
            config.z_limits[1] - config.z_limits[0],
        )
    )
    axis.view_init(elev=24.0, azim=-62.0)
    axis.grid(True, alpha=0.3)


def draw_sphere_obstacles(
    axis: plt.Axes,
    obstacles: list[SphereObstacle],
    alpha: float = 0.32,
    color: str = "gray",
    edgecolor: str = "dimgray",
) -> None:
    """在三维坐标轴中绘制可指定颜色的半透明球形障碍物。"""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 28)
    polar = np.linspace(0.0, np.pi, 18)
    unit_x = np.outer(np.cos(azimuth), np.sin(polar))
    unit_y = np.outer(np.sin(azimuth), np.sin(polar))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))
    for obstacle in obstacles:
        axis.plot_surface(
            obstacle.center[0] + obstacle.radius * unit_x,
            obstacle.center[1] + obstacle.radius * unit_y,
            obstacle.center[2] + obstacle.radius * unit_z,
            color=color,
            alpha=alpha,
            linewidth=0.2,
            edgecolor=edgecolor,
            shade=True,
        )


def plot_trajectories(result: SimulationResult, output_path: Path) -> None:
    """保存五架无人机的三维轨迹与起终点。"""
    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection="3d")
    for index in range(result.positions.shape[1]):
        label = "Leader UAV 0" if index == 0 else f"Follower UAV {index}"
        trajectory = result.positions[:, index]
        axis.plot(*trajectory.T, color=UAV_COLORS[index], linewidth=1.8, label=label)
        axis.scatter(*trajectory[0], color=UAV_COLORS[index], marker="o", s=25)
        axis.scatter(*trajectory[-1], color=UAV_COLORS[index], marker="^", s=35)
    axis.scatter(*result.config.goal, color="black", marker="*", s=130, label="Goal")
    _configure_3d_axes(axis, result)
    axis.set_title(f"3D Formation Trajectories (k_form={result.config.k_form:g})")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_formation_errors(result: SimulationResult, output_path: Path) -> None:
    """保存四架跟随者误差和平均误差曲线。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for follower in range(4):
        axis.plot(
            result.times,
            result.formation_errors[:, follower],
            linewidth=1.2,
            label=f"UAV {follower + 1}",
        )
    axis.plot(
        result.times,
        np.mean(result.formation_errors, axis=1),
        color="black",
        linewidth=2.2,
        label="Follower mean",
    )
    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="1 m")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Formation error (m)")
    axis.set_title(f"Formation Errors (k_form={result.config.k_form:g})")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_error_comparison(
    results: list[SimulationResult], output_path: Path
) -> None:
    """比较三组 k_form 的平均编队误差。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for result in results:
        axis.plot(
            result.times,
            np.mean(result.formation_errors, axis=1),
            linewidth=1.8,
            label=f"k_form={result.config.k_form:g}",
        )
    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="1 m")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mean formation error (m)")
    axis.set_title("Formation Gain Comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_attitudes(result: SimulationResult, output_path: Path) -> None:
    """保存领航者和 UAV 1 的 roll、pitch、yaw 曲线。"""
    names = ("Roll", "Pitch", "Yaw")
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    attitude_degrees = np.rad2deg(result.attitudes)
    for component, axis in enumerate(axes):
        axis.plot(
            result.times,
            attitude_degrees[:, 0, component],
            color=UAV_COLORS[0],
            label="Leader UAV 0",
        )
        axis.plot(
            result.times,
            attitude_degrees[:, 1, component],
            color=UAV_COLORS[1],
            label="Follower UAV 1",
        )
        axis.set_ylabel(f"{names[component]} (deg)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"Visual Attitude Tracking (k_form={result.config.k_form:g})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_rotor_speeds(
    result: SimulationResult, output_path: Path, uav_index: int = 1
) -> None:
    """保存代表性无人机四个视觉旋翼转速。"""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for rotor in range(4):
        axis.plot(
            result.times,
            result.rotor_speeds[:, uav_index, rotor],
            linewidth=1.2,
            label=f"Rotor {rotor + 1}",
        )
    axis.axhline(
        result.config.omega_min_visual, color="gray", linestyle="--", linewidth=0.8
    )
    axis.axhline(
        result.config.omega_max_visual, color="gray", linestyle="--", linewidth=0.8
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Visual rotor speed (rad/s)")
    axis.set_title(f"UAV {uav_index} Visual Rotor Speeds")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _draw_quadcopter(
    axis: plt.Axes,
    position: np.ndarray,
    attitude: np.ndarray,
    phases: np.ndarray,
    color: str,
    arm_length: float,
    rotor_radius: float,
) -> None:
    """在给定世界位姿处绘制 X 形机臂和随相位旋转的四个旋翼。"""
    roll, pitch, yaw = attitude
    rotation = rotation_matrix(roll, pitch, yaw)
    diagonal = arm_length / np.sqrt(2.0)
    rotor_centers_local = np.array(
        [
            [diagonal, diagonal, 0.0],
            [-diagonal, diagonal, 0.0],
            [-diagonal, -diagonal, 0.0],
            [diagonal, -diagonal, 0.0],
        ]
    )
    centers_world = position[None, :] + (rotation @ rotor_centers_local.T).T

    # 两根对角机臂构成统一的 X 形布局。
    for first, second in ((0, 2), (1, 3)):
        segment = np.vstack((centers_world[first], centers_world[second]))
        axis.plot(*segment.T, color=color, linewidth=2.2)
    axis.scatter(*position, color=color, s=20, depthshade=True)

    circle_angles = np.linspace(0.0, 2.0 * np.pi, 18)
    circle_local = np.column_stack(
        (
            rotor_radius * np.cos(circle_angles),
            rotor_radius * np.sin(circle_angles),
            np.zeros_like(circle_angles),
        )
    )
    for rotor_index, center_world in enumerate(centers_world):
        circle_world = center_world[None, :] + (rotation @ circle_local.T).T
        axis.plot(*circle_world.T, color=color, alpha=0.25, linewidth=0.7)

        phase = phases[rotor_index]
        blade_vector_local = rotor_radius * np.array(
            [np.cos(phase), np.sin(phase), 0.0]
        )
        blade_vector_world = rotation @ blade_vector_local
        blade = np.vstack(
            (center_world - blade_vector_world, center_world + blade_vector_world)
        )
        axis.plot(*blade.T, color=color, linewidth=1.6)


def create_formation_animation(
    result: SimulationResult, output_path: Path
) -> None:
    """由预计算历史生成可观察机体倾斜和旋翼旋转的 GIF。

    左侧保留完整航程的固定世界坐标视野；右侧采用固定机体中心相对坐标，
    放大 UAV 1，从而能清楚辨认姿态倾斜和四个旋翼的相位变化。
    """
    figure = plt.figure(figsize=(14, 6.5))
    global_axis = figure.add_subplot(121, projection="3d")
    closeup_axis = figure.add_subplot(122, projection="3d")
    frame_indices = list(range(0, len(result.times), result.config.animation_stride))
    if frame_indices[-1] != len(result.times) - 1:
        frame_indices.append(len(result.times) - 1)

    def update(frame_number: int) -> tuple[()]:
        frame = frame_indices[frame_number]
        global_axis.clear()
        closeup_axis.clear()
        _configure_3d_axes(global_axis, result)
        global_axis.set_title(
            f"Global formation | t={result.times[frame]:.2f} s"
        )
        global_axis.scatter(
            *result.config.goal, color="black", marker="*", s=100, label="Goal"
        )
        obstacles = getattr(result.config, "obstacles", [])
        if obstacles:
            draw_sphere_obstacles(global_axis, obstacles)
        for uav_index in range(result.positions.shape[1]):
            trail = result.positions[: frame + 1, uav_index]
            global_axis.plot(
                *trail.T,
                color=UAV_COLORS[uav_index],
                linewidth=1.0,
                alpha=0.65,
            )
            _draw_quadcopter(
                global_axis,
                result.positions[frame, uav_index],
                result.attitudes[frame, uav_index],
                result.rotor_phases[frame, uav_index],
                UAV_COLORS[uav_index],
                result.config.arm_length,
                result.config.rotor_radius,
            )

        # 机体中心固定近景：坐标范围不变，因此不会随动画自动缩放或抖动。
        representative_index = 1
        closeup_axis.set_xlim(-1.25, 1.25)
        closeup_axis.set_ylim(-1.25, 1.25)
        closeup_axis.set_zlim(-0.9, 0.9)
        closeup_axis.set_box_aspect((2.5, 2.5, 1.8))
        closeup_axis.set_xlabel("body-centered x (m)")
        closeup_axis.set_ylabel("body-centered y (m)")
        closeup_axis.set_zlabel("relative z (m)")
        closeup_axis.view_init(elev=28.0, azim=-52.0)
        closeup_axis.grid(True, alpha=0.3)
        attitude_degrees = np.rad2deg(
            result.attitudes[frame, representative_index]
        )
        closeup_axis.set_title(
            "UAV 1 close-up\n"
            f"roll={attitude_degrees[0]:.1f} deg, "
            f"pitch={attitude_degrees[1]:.1f} deg, "
            f"yaw={attitude_degrees[2]:.1f} deg"
        )
        _draw_quadcopter(
            closeup_axis,
            np.zeros(3, dtype=float),
            result.attitudes[frame, representative_index],
            result.rotor_phases[frame, representative_index],
            UAV_COLORS[representative_index],
            result.config.arm_length,
            result.config.rotor_radius,
        )
        return ()

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / result.config.animation_fps,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=result.config.animation_fps))
    plt.close(figure)
