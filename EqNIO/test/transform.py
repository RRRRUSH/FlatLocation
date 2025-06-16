import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

def plot(lidar, imu):
    # 创建图形窗口
    fig = plt.figure()

    # 定义数据源和对应的列索引
    data_sources = [
        {"name": "lidar", "data": lidar, "columns": [7, 4, 5, 6], "color": 'r-', "x_label": "Time"},
        {"name": "imu", "data": imu, "columns": [4, 1, 2, 3], "color": 'b-', "x_label": "Time"}
    ]

    # 定义子图的标题
    titles = ["Angular Velocity (W)", "Linear Acceleration (X)", "Linear Acceleration (Y)", "Linear Acceleration (Z)"]

    for i, title in enumerate(titles):
        for j, source in enumerate(data_sources):
            subplot_index = i * 2 + j + 1

            ax = fig.add_subplot(4, 2, subplot_index)

            x = np.arange(source["data"].shape[0])
            y = source["data"][:, source["columns"][i]]

            ax.plot(x, y, source["color"], label=f"{source['name']}")
            ax.set_title(title)
            ax.set_xlabel(source["x_label"])
            ax.set_ylabel("Value")
            ax.legend()

    # 调整布局并显示图形
    plt.show()

def main():
    # 读取数据
    lidar = pd.read_csv(r"temp/37.o.csv", sep=",")
    imu = pd.read_csv(r"temp/37.i.csv", sep=",")

    # plot(lidar.values, imu.values)

    rot = R.from_quat(imu[['qx', 'qy', 'qz', 'qw']])
    euler = rot.as_euler('yxz', degrees=True)

    rot_lidar = R.from_quat(lidar[['qx', 'qy', 'qz', 'qw']])
    euler_lidar = rot_lidar.as_euler('yxz', degrees=True)

    plt.subplot(3, 2, 1)
    plt.scatter(np.arange(euler.shape[0]), euler[:, 0], s=1, c='r', label='roll')

    plt.subplot(3, 2, 2)
    plt.scatter(np.arange(euler_lidar.shape[0]), euler_lidar[:, 0], s=1, c='b', label='roll')

    plt.subplot(3, 2, 3)
    plt.scatter(np.arange(euler.shape[0]), euler[:, 1], s=1, c='r', label='pitch')

    plt.subplot(3, 2, 4)
    plt.scatter(np.arange(euler_lidar.shape[0]), euler_lidar[:, 1], s=1, c='b', label='pitch')

    plt.subplot(3, 2, 5)
    plt.scatter(np.arange(euler.shape[0]), euler[:, 2], s=1, c='r', label='yaw')

    plt.subplot(3, 2, 6)
    plt.scatter(np.arange(euler_lidar.shape[0]), euler_lidar[:, 2], s=1, c='b', label='yaw')

    plt.show()



if __name__=="__main__":
    main()