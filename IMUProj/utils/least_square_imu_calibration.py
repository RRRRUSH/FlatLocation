import time
import pandas as pd
import numpy as np
import json

from icecream import ic
from scipy.optimize import least_squares

"""
module1:
    平面向上时 Zacc 为 +
    板子 x 轴指向上时 Xacc 为 +
    板子 y 轴指向上时 Yacc 为 +
"""

def get_offdiag_index():
    """ 返回非对角位置下标 rows, cols """
    return np.where(~np.eye(3, dtype=bool))

def get_diag_index():
    """ 返回对角位置下标 rows, cols """
    return np.where(np.eye(3, dtype=bool))


def stay_error_model(theta, true_x):
    """ 多面静止 """
    bias = theta[:3]
    scale_factor = theta[3:6]
    
    cross_axis = np.zeros((3, 3))
    cross_axis[get_offdiag_index()] = theta[6:]

    return (true_x + true_x @ cross_axis) * scale_factor + bias


def multi_error_model(theta, true_x):
    """ 多点校正 """
    bias = theta[:3]

    scale_factor = np.zeros((3, 3))
    scale_factor[get_diag_index()] = theta[3:6]

    cross_axis = np.zeros((3, 3))
    cross_axis[get_offdiag_index()] = theta[6:]

    return 9.8 * (scale_factor @ cross_axis) + bias


def m1_residuals(theta, messure_y, true_x):
    """ 多面静止残差 """
    return (messure_y - stay_error_model(theta, true_x)).flatten()

def m2_residuals(theta, messure_y):
    """ 多面静止残差 """
    return 


def generate_data(paths, ts, label, is_gyro=False):
    """ 测量值与真实标签拼接 """
    fusion = []

    if is_gyro:
        index = slice(4, 7)
        to_rad = np.pi / 180
    else:
        index = slice(1, 4)

    for i in range(3):
        # clear
        ax_raw = pd.read_csv(paths[i]).dropna()

        # slice
        sensor = np.array(ax_raw.iloc[slice(ts[i]), index]) * (to_rad if is_gyro else 1)

        # add label and fusion
        fusion.append(
            np.concatenate((sensor, np.repeat([label[i]], sensor.shape[0], axis=0)), axis=1)
        )

    return np.concatenate(fusion, axis=0)


def get_mean(paths, ts):

    acc_mean_matrix = np.zeros((3, 3))
    gyro_mean_matrix = np.zeros((3, 3))
    
    for i in range(3):
        data = pd.read_csv(paths[i]).dropna()
        acc = data.iloc[slice(ts[i]), 1:4]
        gyro = data.iloc[slice(ts[i]), 4:7]

        acc_mean_matrix[i] = np.mean(acc, axis=0)
        gyro_mean_matrix[i] = np.mean(gyro, axis=0)

    return acc_mean_matrix, gyro_mean_matrix


if __name__ == "__main__":
    # 200hz 5分钟 60000 条数据
    ts = [500000, 1000000, 1000000]
    raw_data_paths = [
        r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\stay3h_xG.csv",
        r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\stay3h_yG.csv",
        r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\stay3h_zG.csv"
    ]
    acc_mean, gyro_mean = get_mean(raw_data_paths, ts)

    is_gyro = False
    factor = np.mean(np.mean(gyro_mean, axis=0)) if is_gyro else 9.8

    real_label = np.zeros((3, 3))
    real_label[get_diag_index()] = acc_mean[get_diag_index()]
    # real_label[get_offdiag_index()] = factor

    ic(real_label)

    data = generate_data(raw_data_paths, ts, real_label, is_gyro=is_gyro)

    # fit
    # init_theta = np.zeros(12)  # 3 bias + 3 scale + 9 cross_axis
    init_theta = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    beg = time.time()
    result = least_squares(m1_residuals, init_theta, args=(data[:, :3], data[:, 3:]))

    print("Spend ", time.time() - beg, "s")
    
    bias = result.x[:3]  # 零偏

    np.set_printoptions(suppress=True)

    fusion_matrix = np.zeros((3,3))
    fusion_matrix[get_diag_index()] =  result.x[3:6]  # scale factor
    fusion_matrix[get_offdiag_index()] = result.x[6:] # cross axis

    print("bias\n", bias)
    print("error matrix\n", fusion_matrix)