import os
import threading
import csv
import time
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
import quaternion
from scipy.ndimage import gaussian_filter1d

from server.position_fusion_optimize.minimize_error_shape_fit import ShapeFit
from server.position_fusion_optimize.traj_optimize import optimize
from server.utils.data_util import compute_output_time, process_data_source
from server.ws_run import writer
from source.nn.nn_train import add_summary

source_vector = {'gyro', 'acce'}
source_quaternion = {'game_rv'}
source_all = source_vector.union(source_quaternion)


def calculate_angle(x, y):
    # 计算角度（弧度）
    angle_radians = np.arctan2(x, y)
    print('x,y', x, y)
    # 将弧度转换为度数
    angle_degrees = np.degrees(angle_radians)
    print('angle_degrees', angle_degrees)
    # 调整角度范围，使其以 y 轴为 0 度，顺时针方向计算
    angle_degrees = angle_degrees % 360 - 90
    return angle_degrees


def get_target_pos_by_ang(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = math.cos(angle_radians)
    y = math.sin(angle_radians)
    return np.array([x, y])

def write_to_csv(filename,data):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def append_imu_data(gyro, acce, game_rv):
    write_to_csv('abc_gyro.csv',gyro)
    write_to_csv('abc_acce.csv',acce)
    write_to_csv('abc_game_rv.csv',game_rv)

class IMUData:
    def __init__(self):
        self.gyro = []
        self.acce = []
        self.game_rv = []
        self.dts = 0.05
        self.imu_init_gyro_bias = None # gyro_uncalib[0] - gyro[0]
        self.imu_acce_bias = None # acce_calib[0]
        self.imu_acce_scale = None # acce_calib[1]

    def load_data_from_dir(self, dir):
        gyro = np.array(dir['gyro'])
        acce = np.array(dir['acce'])
        game_rv = np.array(dir['game_rv'])

        gyro[:, 0] = gyro[:, 0] / 1e09
        acce[:, 0] = acce[:, 0] / 1e09
        game_rv[:, 0] = game_rv[:, 0] / 1e09

        append_imu_data(gyro, acce, game_rv)

        self.preprocessing_data(gyro, acce, game_rv)

    def load_data_from_file(self, file_path):
        gyro = pd.read_csv(os.path.join(file_path, 'gyro.csv')).values
        acce = pd.read_csv(os.path.join(file_path, 'acce.csv')).values
        game_rv = pd.read_csv(os.path.join(file_path, 'game_rv.csv')).values
        gyro[:, 0] = gyro[:, 0] / 1e09
        acce[:, 0] = acce[:, 0] / 1e09
        game_rv[:, 0] = game_rv[:, 0] / 1e09

        self.preprocessing_data(gyro, acce, game_rv)

    def preprocessing_data(self, gyro, acce, game_rv):
        all_sources = {'gyro': gyro, 'acce': acce, 'game_rv': game_rv}
        output_time = compute_output_time(all_sources)
        self.dts = np.mean(output_time[10::10] - output_time[:-10:10])
        processed_sources = {}
        for source in all_sources.keys():
            if source in source_vector:
                processed_sources[source] = process_data_source(all_sources[source], output_time, 'vector')
            else:
                processed_sources[source] = process_data_source(all_sources[source][:, [0, 4, 1, 2, 3]], output_time,
                                                                'quaternion')
        self.add_new_data(processed_sources)

    def add_new_data(self, processed_sources):

        for row in processed_sources['acce']:
            self.acce.append(row)
        for row in processed_sources['gyro']:
            self.gyro.append(row)
        for row in processed_sources['game_rv']:
            self.game_rv.append(row)

    def get_one_feat(self, window_size, step_size):
        if len(self) >= window_size:
            gyro = np.array(self.gyro[:window_size])
            acce = np.array(self.acce[:window_size])
            game_rv = np.array(self.game_rv[:window_size])
            self.drop_data(step_size)
            return self._get_feat(gyro, acce, game_rv)

    def _get_feat(self, gyro, acce, game_rv):
        ori_q = quaternion.from_float_array(game_rv)
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
        features = np.concatenate([glob_gyro, glob_acce], axis=1, dtype=np.float32).T
        feat_sigma = 0.01
        features = gaussian_filter1d(features, sigma=feat_sigma, axis=0)
        features = torch.from_numpy(features).unsqueeze(0)
        return features

    def drop_data(self, step_size):
        self.acce = self.acce[step_size:]
        self.gyro = self.gyro[step_size:]
        self.game_rv = self.game_rv[step_size:]

    def __len__(self):
        return min(len(self.acce), len(self.gyro), len(self.game_rv))


class DataCache:
    def __init__(self, id):

        self.id = id
        self.dts = 0.05
        self.ang = None

        self.imu_data = IMUData()
        self.shape_fit = ShapeFit()

        self.optimized_uwb = np.array([])
        self.optimized_pos = np.array([])

        self.lock = threading.Lock()

        self.init_cache()

    def init_cache(self):
        dim = 2
        self.uwb_pos = np.array([])
        self.ts_pos = np.array([time.time()] + [0] * dim).reshape(1, dim + 1)

    def com_preds(self, time_pred):
        dim = 2
        temp = np.array(time_pred).reshape(-1, dim + 1)
        new_pos = np.cumsum(temp[:, 1:1 + dim] * self.dts, axis=0) + self.ts_pos[-1, 1:1 + dim]
        temp[:, 1:1 + dim] = new_pos
        self.ts_pos = np.vstack((self.ts_pos, temp))
        if len(self.uwb_pos) != 0:
            self.optimized_pos = np.vstack((self.optimized_pos, self.shape_fit.new_xys_to_lonlat(temp[:, 1:1 + dim])))
        self.plot_data()
    def updata_pos(self, uwb_pos):
        if self.uwb_pos.size == 0:
            self.uwb_pos = np.array([uwb_pos])
        else:
            self.uwb_pos = np.append(self.uwb_pos, [uwb_pos], axis=0)
        # if 0 <= self.ts_pos.shape[0] <= 1002:
        if 0 <= self.ts_pos.shape[0]:
            optimized_pos = self.shape_fit.fit(self.ts_pos[:,:], self.uwb_pos[:,:])
            optimized_uwb = self.uwb_pos[:, [2, 1]]
        # else:
        #     ts_pos = np.concatenate([np.array(self.ts), self.pos], axis=1)
        #     optimized_pos, optimized_uwb = optimize(ts_pos, np.array(self.uwb_pos))
            self.optimized_uwb = optimized_uwb
            self.optimized_pos = optimized_pos
            self.plot_data()

    def rotate_traj_matrix(self, curren_pos, target_pos):
        o_pos = np.array([0, 0])
        vec1 = curren_pos - o_pos
        vec2 = target_pos - o_pos
        if np.allclose(vec2, [0, 0]):
            return np.eye(2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cos_theta = np.dot(vec1, vec2) / ((norm_vec1 * norm_vec2) + 1e-9)
        sin_theta = np.cross(vec1, vec2) / ((norm_vec1 * norm_vec2) + 1e-9)
        theta = np.arctan2(sin_theta, cos_theta)
        c = np.cos(theta)
        s = np.sin(theta)
        rot_matrix = np.array([
            [c, -s],
            [s, c]
        ])
        return rot_matrix

    def save_csv(self):
        with open(f'/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/{self.id}_uwb.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.uwb_pos)
        with open(f'/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/{self.id}_up_pos.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.optimized_pos)
        with open(f'/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/{self.id}_up_uwb.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.optimized_uwb)
        with open(f'/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/{self.id}_pos.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.ts_pos)

    def plot_data(self):
        pos = self.optimized_pos if self.optimized_pos.shape != (0,) else self.ts_pos[:,1:]
        optimized_uwb = self.optimized_uwb

        print(pos.shape)
        print(optimized_uwb.tolist())
        # if self.ang is None and np.linalg.norm(pos[0] - pos[-1]) > 1:
        #     self.ang = self.rotate_traj_matrix(pos[-1], target_pos=get_target_pos_by_ang(200))
        #     print('ang:', self.ang)
        #
        # if self.ang is not None:
        #     pos = np.dot(self.ang, pos.T).T
        plt.figure(figsize=(10, 10))
        if pos.shape != (0,):
            plt.scatter(pos[:, 0], pos[:, 1], c='blue',s=1)

        if optimized_uwb.shape != (0,):
            plt.scatter(optimized_uwb[:, 0], optimized_uwb[:, 1], c='red',s=30)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
