import os
import time
import glob
import numpy as np
import os.path as osp

import pandas as pd
import quaternion
import json


def preprocess(path):
    data = np.loadtxt(osp.join(path, "imu_processed.csv"), delimiter=',', skiprows=1)

    ts_us = data[:, 0] * 1e06
    gyr = data[:, 1:4] # rad/s
    acc = data[:, 4:7] # m/s^2
    q = data[:, 7:11]
    pos_xyz = data[:, 11:14]
    vel_xyz = data[:, 14:17]
    # 'imu0_resampled.npy'

    print(gyr.shape, acc.shape, q.shape, pos_xyz.shape, vel_xyz.shape)

    np.save(osp.join(path, 'imu0_resampled.npy'),
            np.concatenate(
                [ts_us.reshape(-1, 1), gyr, acc, q, pos_xyz, vel_xyz],
                axis=1)
            )
    description = {
        "columns_name(width)": [
            "ts_us(1)",
            "gyr_compensated_rotated_in_World(3)",
            "acc_compensated_rotated_in_World(3)",
            "qwxyz_World_Device(4)",
            "pos_World_Device(3)",
            "vel_World(3)"
        ],
        "num_rows": int(gyr.shape[0]),
        "approximate_frequency_hz": 200.0,
        "t_start_us": float(ts_us[0]),
        "t_end_us": float(ts_us[-1])
    }
    with open(osp.join(path, 'imu0_resampled_description.json'), 'w') as f:
        json.dump(description, f, indent=4)


def compare():
    path = r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\raw\golden-new-format-cc-by-nc-with-imus\110982076486017"
    nog_p = r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\raw\TLIOv2\110982076486017"
    data_resampled = np.load(osp.join(path, "imu0_resampled.npy"))
    nog = np.load(osp.join(nog_p, "imu0_resampled.npy"))
    nog_csv = np.loadtxt(osp.join(nog_p, "imu_processed.csv"), delimiter=',', skiprows=1)

    np.set_printoptions(precision=4, edgeitems=10, linewidth=300)

    print(data_resampled, data_resampled.shape, '\n')

    print(data_resampled - nog, nog.shape)

    print(data_resampled - nog_csv, nog_csv.shape)


if __name__=="__main__":
    root_path = r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\raw\TLIOv2"
    sub_dirs = [item for item in glob.glob(osp.join(root_path, "*")) if osp.isdir(item)]

    for sub_dir in sub_dirs:
        preprocess(osp.join(sub_dir))
    # compare()