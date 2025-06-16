import os
import time
from os.path import exists

import numpy as np
import quaternion
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils.data_util import compute_output_time, process_data_source

source_vector = {'gyro', 'acce'}


class IMUData:
    def __init__(self):
        self.gyro = np.empty((0, 3))
        self.acce = np.empty((0, 3))
        self.dts = 0.05
        self.imu_init_gyro_bias = None  # gyro_uncalib[0] - gyro[0]
        self.imu_acce_bias = None  # acce_calib[0]
        self.imu_acce_scale = None  # acce_calib[1]

    def preprocessing_data(self, dir):
        output_time = compute_output_time(dir, 200)
        self.dts = np.mean(output_time[10::10] - output_time[:-10:10])
        processed_sources = {}
        for source in dir.keys():
            if source in source_vector:
                processed_sources[source] = process_data_source(dir[source], output_time, 'vector')
            else:
                processed_sources[source] = process_data_source(dir[source][:, [0, 4, 1, 2, 3]], output_time,
                                                                'quaternion')
        self.gyro = np.vstack((self.gyro, processed_sources['gyro']))
        self.acce = np.vstack((self.acce, processed_sources['acce']))

    def get_one_feat(self, window_size, step_size):
        if len(self) >= window_size:
            gyro = self.gyro[:window_size]
            acce = self.acce[:window_size]
            self.drop_data(step_size)
            return self._get_feat(gyro, acce)

    def _get_feat(self, gyro, acce):
        features = np.concatenate([gyro, acce], axis=1, dtype=np.float32).T
        feat_sigma = 0.0001
        features = gaussian_filter1d(features, sigma=feat_sigma, axis=0)
        features = torch.from_numpy(features).unsqueeze(0)
        return features

    def drop_data(self, step_size):
        self.acce = self.acce[step_size:]
        self.gyro = self.gyro[step_size:]

    def __len__(self):
        return min(len(self.acce), len(self.gyro))


class DataCache:
    def __init__(self, id='abc', dim=2):
        self.id = id
        self.dim = dim
        self.dts = 0.05
        self.trajectory = None
        self.velocity = None
        self.imu_data = IMUData()

        self.init_cache()

    def init_cache(self):
        self.trajectory = np.array([time.time()] + [0] * self.dim).reshape(1, self.dim + 1)
        self.velocity = np.array([0] * self.dim).reshape(1, self.dim)

    def com_preds(self, time_pred):
        temp = np.array(time_pred).reshape(-1, self.dim + 1)
        self.velocity = np.vstack((self.velocity, temp[:, 1:1 + self.dim]))
        new_pos = np.cumsum(temp[:, 1:1 + self.dim] * self.imu_data.dts, axis=0) + self.trajectory[-1, 1:1 + self.dim]
        temp[:, 1:1 + self.dim] = new_pos
        self.trajectory = np.vstack((self.trajectory, temp))

    def hand_io(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.visualize(save_path)
        self.save_file(save_path)

    def save_file(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt(os.path.join(save_path, "trajectory.txt"), self.trajectory[:, 1:])
        np.savetxt(os.path.join(save_path, "velocity.txt"), self.velocity)

    def plot_2dtrajectory(self, save_path):
        pos = self.trajectory[:, 1:]
        plt.figure(figsize=(10, 10))
        if pos.shape != (0,):
            plt.scatter(pos[:, 0], pos[:, 1], s=1)
            if True:
                step = 2000
                for i in range(0, pos.shape[0], step):
                    plt.scatter(pos[i, 0], pos[i, 1], s=50, color='red')
                    plt.annotate(f'{i}', (pos[i, 0], pos[i, 1]), fontsize=12, ha='right')
        plt.scatter(pos[-1, 0], pos[-1, 1], s=50, color='red')
        plt.annotate('last', (pos[-1, 0], pos[-1, 1]), fontsize=12, ha='right')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('x (m)', fontsize=18)
        plt.ylabel('y (m)', fontsize=18)
        plt.grid(True)
        plt.axis('equal')
        plt.title(save_path.split('/')[-1], fontsize=18)
        plt.savefig(os.path.join(save_path, "trajectory.png"))
        plt.show()

    def plot_2dvelocity(self, save_path):
        velocity = self.velocity
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(range(velocity.shape[0]), velocity[:, 0], label='x')
        axs[0].set_title('X Velocity')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Velocity')
        axs[0].legend()

        axs[1].plot(range(velocity.shape[0]), velocity[:, 1], label='y')
        axs[1].set_title('Y Velocity')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "velocity.png"))
        plt.show()

    def visualize(self, save_path):
        if self.dim == 2:
            self.plot_2dvelocity(save_path)
            self.plot_2dtrajectory(save_path)
