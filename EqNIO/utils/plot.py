import os
import glob

import matplotlib.pyplot as plt
import pandas as pd

from contrast_traj import *

GROUND_TRUTH = np.array([
    [0, 0], [0, 41], [-17, 41], [-17, 3], [-4, 3], [-4, 0], [0, 0]
]) * 0.8

def rot(x, y, theta):
    theta_rad = np.deg2rad(theta)  # 将角度转换为弧度
    x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    return x_rot, y_rot

def gt_limit():
    global GROUND_TRUTH
    x, y = GROUND_TRUTH.T
    plt.scatter(x, y, color='red')
    plt.plot(x, y, color='black')
def calculate_initial_angle(x, y, index=3000):
    """计算初始方向的角度"""
    angle = np.arctan2(np.mean(y[:index]), np.mean(x[:index]))
    y_up = np.arctan2(1, 0)
    return np.rad2deg(angle - y_up)  # 转换为角度

def plot_traj(pos, outfile, title):

    x, y = pos[:, 0], pos[:, 1]
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

    plt.figure(figsize=(8, 6))
    gt_limit()

    plt.plot(traj_rot[:, 0], traj_rot[:, 1], 'r-')
    plt.plot(traj_rot[0, 0], traj_rot[0, 1], 'bo')
    plt.plot(traj_rot[-1, 0], traj_rot[-1, 1], 'bo')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.axis('equal')
    plt.savefig(outfile)
    plt.close()

def run():

    root_dir = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\output\0410\test_radar_body_wq"
    gt_dir = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\data\tx\test_radar_body"

    is_040 = False

    sub_dir = [item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))]

    for sub_dir_name in sub_dir:
        sub_dir_path = os.path.join(root_dir, sub_dir_name)
        if os.path.isdir(sub_dir_path):

            if is_040:
                npy_path = glob.glob(os.path.join(sub_dir_path, "not_vio_state.txt.npy"))[0]
                print(npy_path)

                pred = np.load(npy_path)

                # 无真实轨迹
                plot_traj(pred[:, 12:15], os.path.join(sub_dir_path, f"{sub_dir_name}.png"), sub_dir_name)
            else:
                npy_path = glob.glob(os.path.join(sub_dir_path, "not_vio_state.txt.npy"))[0]

                print(npy_path)
                pred = np.load(npy_path)

                # 读取真实轨迹、计算误差
                targ = np.loadtxt(
                    os.path.join(gt_dir, sub_dir_name, "target_pos.txt"),
                    delimiter=" "
                )

                fig = contrast_traj(pred[:, 12:15], targ[:, 1:4])
                fig.savefig(os.path.join(sub_dir_path, f"{sub_dir_name}.png"))
                plt.close(fig)

if __name__ == "__main__":
    run()