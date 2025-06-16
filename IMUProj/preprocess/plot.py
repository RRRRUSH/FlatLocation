import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


GROUND_TRUTH = np.array([
    [0, 0], [0, 41], [-17, 41], [-17, 3], [-4, 3], [-4, 0], [0, 0]
]) * 0.8

def rot(x, y, theta):
    theta_rad = np.deg2rad(theta)  # 将角度转换为弧度
    x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    return x_rot, y_rot

def calculate_initial_angle(x, y, index=3000):
    """计算初始方向的角度"""
    angle = np.arctan2(np.mean(y[:index]), np.mean(x[:index]))
    y_up = np.arctan2(1, 0)
    return np.rad2deg(angle - y_up)  # 转换为角度

def calculate_error(traj_rot, gt):
    """
    计算旋转后轨迹与GROUND_TRUTH之间的最近邻距离平均值
    :param traj_rot: 旋转后的轨迹点，形状为 (N, 2)
    :param gt: GROUND_TRUTH 定点，形状为 (M, 2)
    :return: 最近邻距离的平均值
    """
    distances = cdist(traj_rot, gt, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)

def gen_gts(nums):
    pass

def gt_limit():
    global GROUND_TRUTH
    x, y = GROUND_TRUTH.T
    plt.scatter(x, y, color='red')
    plt.plot(x, y, color='black')

def traj(path) -> None:
    if path.endswith('txt'):
        # ronin
        data = pd.read_csv(path, sep=' ', header=None).values
        x, y = data[:, 0], data[:, 1]
    else:
        # tlio
        data = np.load(path)
        x, y = data[:, 12], data[:, 13]

    best_angle = 0
    best_loss = 999
    for index in range(1, 1000, 1):
        x_rot, y_rot = rot(x, y, -calculate_initial_angle(x, y, index))
        loss = abs(np.mean(x_rot[:index]))
        best_angle = -calculate_initial_angle(x, y, index)

        if loss < best_loss:
            best_loss = loss
            best_angle = -calculate_initial_angle(x, y, index)

    best_x_rot, best_y_rot = rot(x, y, best_angle)

    traj_rot = np.vstack((best_x_rot, best_y_rot)).T
    error = calculate_error(traj_rot, GROUND_TRUTH)

    print(f"旋转后轨迹与 GROUND_TRUTH 的平均最近邻距离: {error:.4f}")

    plt.plot(best_x_rot, best_y_rot, color='red')

def flat_rect():
    global GROUND_TRUTH
    data_path = r"""
E:\HaozhanLi\Project\FlatLoc\IMUProj\data\backup\output\tlio\xxb_lsf_imucali\0325_4\ch040\not_vio_state.txt.npy
    """.strip()
    plt.figure(figsize=(8, 8))

    gt_limit()
    traj(data_path)
    plt.title(os.path.split(os.path.dirname(data_path))[-1])

    plt.axis('equal')
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(data_path), "traj.png"))    


if __name__=="__main__":
    flat_rect()