import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


def get_first_turn_index(points, window_size=200, angle_threshold=np.pi / 4, distance_threshold=1):
    # 将 points 转换为 numpy 数组以便于计算
    points = np.array(points)

    # 计算相邻点之间的向量
    vectors = points[1:] - points[:-1]

    # 计算相邻向量之间的角度
    angles = []
    distances = np.linalg.norm(vectors, axis=1)  # 计算每个向量的长度（移动距离）

    for i in range(1, len(vectors)):
        vec1 = vectors[i - 1]
        vec2 = vectors[i]
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)
        angle = np.arccos(np.clip(cos_theta, -np.pi * 2, np.pi * 2))  # 确保cos_theta在[-1, 1]范围内
        angles.append(angle)

    # 使用滑动窗口计算角度变化的平均值和移动距离的平均值
    total_angles = []
    total_distances = []
    for i in range(len(angles) - window_size + 1):
        window_angles = angles[i:i + window_size]
        total_angle = np.sum(window_angles)
        total_angles.append(total_angle)

        window_distances = distances[i:i + window_size]
        total_distance = np.sum(window_distances)
        total_distances.append(total_distance)

    # 找到角度变化超过阈值且移动距离超过阈值的第一个索引
    for i, (angle, distance) in enumerate(zip(total_angles, total_distances)):
        if angle > angle_threshold and distance > distance_threshold:
            return i + window_size - 1  # 返回第一个转弯点的索引

    return -1  # 如果没有找到转弯点，返回-1


def rotate_traj_matrix(curren_pos, target_pos):
    o_pos = np.array([0, 0])
    vec1 = curren_pos - o_pos
    vec2 = target_pos - o_pos

    # 检查 vec2 是否为零向量
    if np.allclose(vec2, [0, 0]):
        return np.eye(2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    cos_theta = np.dot(vec1, vec2) / ((norm_vec1 * norm_vec2) + 1e-9)
    sin_theta = np.cross(vec1, vec2) / ((norm_vec1 * norm_vec2) + 1e-9)

    theta = np.arctan2(sin_theta, cos_theta)

    # 计算旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)

    rot_matrix = np.array([
        [c, -s],
        [s, c]
    ])

    return rot_matrix


def calculate_distance(x1, y1, x2, y2):
    # 使用欧几里得距离公式
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def contrast_traj(traj1, traj2, labels=['Prediction', 'Target'], is_align=True, is_show=False, title=None):
    traj1, traj2 = align_traj_len(traj1, traj2)

    if is_align:
        aligned_traj1, aligned_traj2 = align_plots(traj1[:, :2], traj2[:, :2])

    # 计算终点误差
    end_error = calculate_distance(aligned_traj1[-1, 0], aligned_traj1[-1, 1], aligned_traj2[-1, 0],
                                   aligned_traj2[-1, 1])

    # 在计算平均误差和终点误差后，添加总距离计算
    total_dist1 = np.sum(np.array(
        [calculate_distance(aligned_traj1[i - 1, 0], aligned_traj1[i - 1, 1], aligned_traj1[i, 0], aligned_traj1[i, 1])
         for i in range(1, len(aligned_traj1))]))
    total_dist2 = np.sum(np.array(
        [calculate_distance(aligned_traj2[i - 1, 0], aligned_traj2[i - 1, 1], aligned_traj2[i, 0], aligned_traj2[i, 1])
         for i in range(1, len(aligned_traj2))]))

    ates = compute_ates(aligned_traj1, aligned_traj2, round(aligned_traj1.shape[0] * 0.01))
    max_ate_index = np.argmax(ates)
    max_ate = ates[max_ate_index]
    mean_ate = np.mean(ates)
    min_ate = np.min(ates)
    mean_error = mean_ate

    end_dist1 = calculate_distance(aligned_traj1[0, 0], aligned_traj1[0, 1], aligned_traj1[-1, 0], aligned_traj1[-1, 1])
    end_dist2 = calculate_distance(aligned_traj2[0, 0], aligned_traj2[0, 1], aligned_traj2[-1, 0], aligned_traj2[-1, 1])

    # 修改子图布局为 1 行 2 列
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})
    aligned_traj1 = np.concatenate([aligned_traj1, traj1[:, [2]]], axis=1)
    aligned_traj2 = np.concatenate([aligned_traj2, traj2[:, [2]]], axis=1)

    # 左边：轨迹图（放在第一个子图）
    ax_left = axes[0]
    if title is not None:
        ax_left.set_title(title)
    # 绘制轨迹
    ax_left.scatter(aligned_traj1[:, 0], aligned_traj1[:, 1], s=1, label=labels[0])
    ax_left.scatter(aligned_traj2[:, 0], aligned_traj2[:, 1], s=1, label=labels[1])
    ax_left.scatter(aligned_traj1[max_ate_index, 0], aligned_traj1[max_ate_index, 1], s=30, c='r')
    ax_left.scatter(aligned_traj2[max_ate_index, 0], aligned_traj2[max_ate_index, 1], s=30, c='r')

    # 添加文本标注
    ax_left.text(0.05, 0.95, f'Pred Total: {total_dist1:.2f}', transform=ax_left.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_left.text(0.05, 0.90, f'Target Total: {total_dist2:.2f}', transform=ax_left.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_left.text(0.05, 0.85, f'Mean Error: {mean_error:.2f}', transform=ax_left.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_left.text(0.05, 0.80, f'Max Error: {max_ate:.2f}', transform=ax_left.transAxes, fontsize=10,
                 verticalalignment='top')
    ax_left.text(0.05, 0.75, f'End Error: {end_error:.2f}', transform=ax_left.transAxes, fontsize=10,
                 verticalalignment='top')

    ax_left.legend(loc='upper right')
    ax_left.axis('equal')

    # 右边：X、Y、Z 分量图（分别放在第 2 个子图的 3 行 1 列）
    x = np.arange(len(aligned_traj1))  # 时间步或点索引

    # 创建 4 行 1 列的子图
    ax_right = axes[1]
    ax_right_x = ax_right.inset_axes([0, 0.75, 1, 0.20])
    ax_right_y = ax_right.inset_axes([0, 0.5, 1, 0.20])
    ax_right_z = ax_right.inset_axes([0, 0.25, 1, 0.20])
    ax_right_ate = ax_right.inset_axes([0, 0, 1, 0.20])

    # X 轴分量图
    ax_right_x.scatter(x, aligned_traj1[:, 0], label=f'{labels[0]} X', s=1)
    ax_right_x.scatter(x, aligned_traj2[:, 0], label=f'{labels[1]} X', s=1)
    ax_right_x.text(0.05, 0.95, f'Mean Error: {(np.mean(aligned_traj1[:, 0] - aligned_traj2[:, 0])):.2f}',
                    transform=ax_right_x.transAxes, fontsize=10,
                    verticalalignment='top')
    ax_right_x.text(0.05, 0.85, f'End Error: {(aligned_traj1[-1, 0] - aligned_traj2[-1, 0]):.2f}',
                    transform=ax_right_x.transAxes, fontsize=10,
                    verticalalignment='top')
    ax_right_x.legend(loc='upper right')
    ax_right_x.grid(True)

    # Y 轴分量图
    ax_right_y.scatter(x, aligned_traj1[:, 1], label=f'{labels[0]} Y', s=1)
    ax_right_y.scatter(x, aligned_traj2[:, 1], label=f'{labels[1]} Y', s=1)
    ax_right_y.text(0.05, 0.95, f'Mean Error: {(np.mean(aligned_traj1[:, 1] - aligned_traj2[:, 1])):.2f}',
                    transform=ax_right_y.transAxes, fontsize=10,
                    verticalalignment='top')
    ax_right_y.text(0.05, 0.85, f'End Error: {(aligned_traj1[-1, 1] - aligned_traj2[-1, 1]):.2f}',
                    transform=ax_right_y.transAxes, fontsize=10,
                    verticalalignment='top')
    ax_right_y.legend(loc='upper right')
    ax_right_y.grid(True)

    # Z 轴分量图
    ax_right_z.scatter(x, aligned_traj1[:, 2], label=f'{labels[0]} Z', s=1)
    ax_right_z.scatter(x, aligned_traj2[:, 2], label=f'{labels[1]} Z', s=1)
    ax_right_z.text(0.05, 0.95, f'Mean Error: {(np.mean(aligned_traj1[:, 2] - aligned_traj2[:, 2])):.2f}',
                    transform=ax_right_z.transAxes, fontsize=10,
                    verticalalignment='top')
    ax_right_z.text(0.05, 0.85, f'End Error: {(aligned_traj1[-1, 2] - aligned_traj2[-1, 2]):.2f}',
                    transform=ax_right_z.transAxes, fontsize=10,
                    verticalalignment='top')
    ax_right_z.legend(loc='upper right')
    ax_right_z.grid(True)

    # ate 轴分量图
    ax_right_ate.scatter(np.arange(len(ates)), ates, s=1, label='ates')
    ax_right_ate.plot([0, len(ates)], [mean_ate, mean_ate], c='g', label='mean ate')
    ax_right_ate.legend(loc='upper right')
    ax_right_ate.grid(True)

    # 隐藏右边的主轴
    ax_right.axis('off')

    # 调整布局
    plt.tight_layout()

    if is_show:
        plt.show()
    return fig # , f'{total_dist1},{total_dist2},{mean_error},{end_error},{mean_ate},{max_ate},{end_dist1},{end_dist2}'


def compute_ates(est, gt, delta):
    ates = compute_absolute_trajectory_errors(est, gt, delta)

    return ates


def compute_absolute_trajectory_error(est, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    return np.mean(
        np.array([calculate_distance(xy1[0], xy1[1], xy2[0], xy2[1]) for (xy1, xy2) in zip(est[:, :2], gt[:, :2])]))
    # return np.sqrt(np.mean(np.linalg.norm(est - gt, axis=-1) ** 2))


def compute_absolute_trajectory_errors(est, gt, delta):
    # 输入校验
    assert len(gt) == len(est), "轨迹长度必须一致"
    gt_points = np.asarray(gt)
    est_points = np.asarray(est)

    errors = []

    for i in range(0, len(gt_points)):
        ate = compute_absolute_trajectory_error(est_points[i:i + delta], gt_points[i:i + delta])
        errors.append(ate)

    return errors


def align_traj_len(traj1, traj2):
    # 获取原始轨迹长度
    len_pred = len(traj1)
    len_target = len(traj2)
    target_length = max(len_pred, len_target)  # 选择较长的长度作为目标长度

    # 定义统一的采样点
    x_pred = np.linspace(0, 1, len_pred)
    x_target = np.linspace(0, 1, len_target)
    x_common = np.linspace(0, 1, target_length)

    # 对预测轨迹进行插值
    pred_x_func = interp1d(x_pred, traj1[:, 0], kind='linear')
    pred_y_func = interp1d(x_pred, traj1[:, 1], kind='linear')
    pred_z_func = interp1d(x_pred, traj1[:, 2], kind='linear')
    traj1 = np.column_stack([
        pred_x_func(x_common),
        pred_y_func(x_common),
        pred_z_func(x_common),
    ])

    # 对目标轨迹进行插值
    target_x_func = interp1d(x_target, traj2[:, 0], kind='linear')
    target_y_func = interp1d(x_target, traj2[:, 1], kind='linear')
    target_z_func = interp1d(x_target, traj2[:, 2], kind='linear')
    traj2 = np.column_stack([
        target_x_func(x_common),
        target_y_func(x_common),
        target_z_func(x_common),
    ])
    return traj1, traj2


def optimize_rotation(traj1, traj2):
    """
    使用 scipy.optimize.least_squares 优化 traj1 的旋转，使其与 traj2 达到最小误差。

    参数:
        traj1 (np.ndarray): 预测轨迹点集 (N, 2)。
        traj2 (np.ndarray): 目标轨迹点集 (N, 2)。

    返回:
        np.ndarray: 旋转后的 traj1。
    """

    def error_function(params, traj1, traj2):
        """
        误差函数，用于计算旋转后的 traj1 和 traj2 之间的误差。

        参数:
            params (list): 包含旋转角度 theta 的参数列表。
            traj1 (np.ndarray): 预测轨迹点集 (N, 2)。
            traj2 (np.ndarray): 目标轨迹点集 (N, 2)。

        返回:
            np.ndarray: 每个点的误差向量。
        """
        theta = params[0]
        # 构造旋转矩阵
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        # 旋转 traj1
        traj1_rotated = np.dot(rotation_matrix, traj1.T).T
        # 计算误差
        return (traj1_rotated - traj2).ravel()  # 展平为一维数组

    # 初始猜测值为 0（无旋转）
    initial_guess = [0.0]

    align_index = -1
    dist = 0
    for i in range(1,traj2.shape[0]):
        dist += np.linalg.norm(traj2[i-1] - traj2[i])
        if dist >= 50:
            align_index = i
            break

    # 使用 least_squares 进行优化
    result = least_squares(error_function, initial_guess, args=(traj1[:align_index, :], traj2[:align_index, :]))

    # 提取优化后的旋转角度
    optimized_theta = result.x[0]

    # 应用优化后的旋转矩阵
    c_opt, s_opt = np.cos(optimized_theta), np.sin(optimized_theta)
    optimized_rotation_matrix = np.array([[c_opt, -s_opt], [s_opt, c_opt]])
    traj1_rotated = np.dot(optimized_rotation_matrix, traj1.T).T

    return traj1_rotated


def align_plots(traj1, traj2):
    return optimize_rotation(traj1, traj2), traj2
