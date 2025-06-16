import os
import glob
import json
import time

import numpy as np
import pandas as pd

def batch(path):
    sub_dirs = [item for item in glob.glob(os.path.join(path, "*")) if os.path.isdir(item)]

    if len(sub_dirs) == 0:
        print("No sub dir!")
        return

    for i, sub_dir in enumerate(sub_dirs):
        print(f"Processing {i+1}/{len(sub_dirs)}: {sub_dir}")

        data = pd.read_csv(os.path.join(sub_dir, "data.csv"), header=None, sep=",").dropna().values

        ts = data[:, 0] * 1e03 # us
        gyro = data[:, 1:4]
        acce = data[:, 4:7]
        quat = data[:, 7:11]
        pos = data[:, 11:14]


        np.save(
            os.path.join(sub_dir, "imu0_resampled.npy"),
            np.concatenate([ts.reshape(-1, 1), gyro, acce, quat, pos, pos], axis=1)
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
            "num_rows": int(gyro.shape[0]),
            "approximate_frequency_hz": 200.0,
            "t_start_us": float(ts[0]),
            "t_end_us": float(ts[-1])
        }
        with open(os.path.join(sub_dir, 'imu0_resampled_description.json'), 'w') as f:
            json.dump(description, f, indent=4)


def single():
    pass

def test():
    npy = np.load(r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\vr_step_20250311\yanlijun_250311141236\imu0_resampled.npy")
    csv = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\vr_step_20250311\yanlijun_250311141236\data.csv", header=None).values
    raw = np.load(r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\left_breast_4h\1740583225239\data1\imu0_resampled.npy")

    print(npy.shape, csv.shape)

    np.set_printoptions(precision=3, suppress=True, linewidth=300)

    print(npy[:10, :], "\n")
    print(csv[:10, :])

    print(time.time())

    print(raw[:10, :])

if __name__ == '__main__':
    root_path = r"E:\HaozhanLi\Project\FlatLoc\tlio\data\train\vr_step_20250311"

    # batch(root_path)
    test()