import glob

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
def task1():
    ''' 1: 非纯净数据集训练LLIO与TLIO_O2 '''

    # 指定原始文件和新文件的路径
    original_file_path = r'/val_list_all.txt'  # 替换为实际的原始文件路径
    filtered_file_path = r'/val_list.txt'  # 替换为实际的新文件路径

    # 使用with语句同时打开原始文件和新文件
    with open(original_file_path, 'r', encoding='utf-8') as f:
        with open(filtered_file_path, 'w', encoding='utf-8') as nf:
            # 逐行读取原始文件
            for line in f:
                # 检查该行是否包含‘250401’或‘250402’
                if '250401' in line or '250402' in line:
                    # 将符合条件的行写入新文件
                    nf.write(line)

def temp():
    ''' 2: 标记出丢包数据所在的位置 '''
    rtk = False

    if rtk:
        # 读取原始的 IMU 时间戳
        root_dir = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\data\tx\test_rtk_body\2025-04-10_193717"

        data_raw = pd.read_csv(
            os.path.join(root_dir, 'imu_data.txt.txt'),
            header=None, sep=","
            ).values

        os.makedirs(os.path.join(root_dir, 'raw_ts'), exist_ok=True)
        np.savetxt(
            os.path.join(root_dir, 'raw_ts', 'imu_samples_0.csv'),
            np.concatenate([
                ((data_raw[:, 0] - data_raw[0, 0]) * 1e6).reshape(-1, 1),
                np.zeros((data_raw.shape[0], 1)),
                data_raw[:, 5:8],
                data_raw[:, 2:5],
                ], axis=1
            ),
            header='#timestamp [ns],temperature [degC],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]',
            delimiter=","
        )
    else:
        # 读取原始的 IMU 时间戳
        root_dir = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\data\54"

        data_raw = pd.read_csv(
            os.path.join(root_dir, '31.i.csv'), sep=","
        ).values

        os.makedirs(os.path.join(root_dir, 'raw_ts'), exist_ok=True)
        np.savetxt(
            os.path.join(root_dir, 'raw_ts', 'imu_samples_0.csv'),
            np.concatenate([
                ((data_raw[:, 0] - data_raw[0, 0]) * 1e9).reshape(-1, 1),
                np.zeros((data_raw.shape[0], 1)),
                data_raw[:, 5:8],
                data_raw[:, 8:11],
            ], axis=1
            ),
            header='#timestamp [ns],temperature [degC],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]',
            delimiter=","
        )

def task2():
    ''' 2: 标记出丢包数据所在的位置 '''
    # 读取原始的 IMU 时间戳
    root_dir = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\data\vr_body_20250408_test_x"

    paths = [os.path.join(root_dir, dir) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]

    for path in paths:
        all_dirs = glob.glob(os.path.join(path, "*"))
        subdir = [d for d in all_dirs if not os.path.basename(d).startswith("imu")][0]

        data_paths = glob.glob(os.path.join(subdir, "imu-*.txt"))

        if len(data_paths) != 2:
            raise ValueError(f"{path} IMU NUM ERROR")

        for i in range(len(data_paths)):

            imu = pd.read_csv(data_paths[i], sep=",", header=None).values
            out_path = os.path.dirname(os.path.dirname(data_paths[i]))
            os.makedirs(os.path.join(out_path, os.path.basename(data_paths[i]).split(".")[0]), exist_ok=True)

            print(os.path.join(out_path, os.path.basename(data_paths[i]).split(".")[0]))
            np.save(
                os.path.join(out_path, os.path.basename(data_paths[i]).split(".")[0], 'time_gap.npy'),
                (imu[:, 0] - imu[0, 0]).reshape(-1, 1)
            )
            np.savetxt(
                os.path.join(out_path, os.path.basename(data_paths[i]).split(".")[0], 'imu_samples_0.csv'),
                np.concatenate([
                    (np.arange(imu.shape[0]) * 5 * 1e06).reshape(-1, 1),
                    np.zeros((imu.shape[0], 1)),
                    imu[:, 4:7],
                    imu[:, 1:4],
                ], axis=1
                ),
                header='#timestamp [ns],temperature [degC],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]',
                delimiter=","
            )

def task3():
    ''' 3: 将带有时差标记的点绘制出来 '''
    real_path = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\data\vr_body_20250408_test_x"
    out_path = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\output\0415\vr_body_20250408_test_x_timegap"

    for dirpath, dirnames, filenames in os.walk(out_path):
        if 'not_vio_state.txt.npy' in filenames:
            # print(dirpath)

            points = np.load(os.path.join(dirpath, 'not_vio_state.txt.npy'))

            pos_ts = points[:, -1]
            real_ts = np.load(os.path.join(dirpath.replace(out_path, real_path), 'time_gap.npy'))
            real_ts = real_ts.flatten() * 1e3

            align_ts = real_ts[np.where((real_ts >= pos_ts[0]) & (real_ts <= pos_ts[-1]))]

            print(pos_ts)
            print(align_ts)
            print(pos_ts.shape, align_ts.shape)

            matched_ts = np.full(pos_ts.shape[0], np.nan)
            for i, value in enumerate(pos_ts):
                mask = np.isclose(align_ts, value, atol=1e03)
                if np.any(mask):
                    matched_ts[i] = align_ts[mask][0]

            x, y = points[:, 12], points[:, 13]
            gap = points[np.where(np.isnan(matched_ts))]

            fig = plt.figure(figsize=(12, 6))
            ax0 = fig.add_subplot(1, 2, 1)
            ax0.plot(x, y, 'b-')
            ax0.scatter(gap[:, 12], gap[:, 13], color='red')

            ax1 = fig.add_subplot(1, 2, 2)
            # ax1.scatter(np.arange(len(pos_ts)), pos_ts, s=0.5, c='b')
            ax1.scatter(np.arange(len(align_ts)), align_ts, s=0.5, c='r')

            plt.savefig(os.path.join(dirpath, 'time_gap.png'))
            plt.close(fig)
            # plt.show()

def check_time_gap():
    root_path = r"E:\HaozhanLi\Project\FlatLoc\EqNIO\data\vr_body_20250408_test_x"

    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'time_gap.npy' in filenames:
            time_gap = np.load(os.path.join(dirpath, 'time_gap.npy'))

            print(f"{dirpath}", "丢包数", len(np.where((time_gap[1:, 0] - time_gap[:-1, 0]) != 5.)[0]))


if __name__ == "__main__":
    check_time_gap()