from os import path as osp

import numpy as np
import pandas
import quaternion
from Sequence.CompiledSequence import CompiledSequence


class RONINVIOSequence(CompiledSequence):
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.imu_freq = 100
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path, verbose=True):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        file = osp.join(data_path, 'SenseINS.csv')
        imu_all = pandas.read_csv(file)

        ts = imu_all[['times']].values

        vio_q = np.array(imu_all[['vio_q_w', 'vio_q_x', 'vio_q_y', 'vio_q_z']].values)
        vio_p = np.array(imu_all[['vio_p_x', 'vio_p_y', 'vio_p_z']].values)

        gyro = np.array(imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values)
        acce = np.array(imu_all[['acce_x', 'acce_y', 'acce_z']].values)

        tmp_vio_gyro_bias = np.array(imu_all[['vio_gyro_bias_x', 'vio_gyro_bias_y', 'vio_gyro_bias_z']].values)
        tmp_vio_acce_bias = np.array(imu_all[['vio_acce_bias_x', 'vio_acce_bias_y', 'vio_acce_bias_z']].values)

        gyro = gyro - tmp_vio_gyro_bias[-1, :]
        acce = acce - tmp_vio_acce_bias[-1, :]

        ori_R = quaternion.from_float_array(vio_q)

        nz = np.zeros(ts.shape)

        gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

        gyro_glob = quaternion.as_float_array(ori_R * gyro_q * ori_R.conj())[:, 1:]
        acce_glob = quaternion.as_float_array(ori_R * acce_q * ori_R.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        self.orientations = quaternion.as_float_array(ori_R)  # [x, y, z, w]
        self.gt_pos = vio_p
        self.targets = (vio_p[self.w:, :self.target_dim] - vio_p[:-self.w, :self.target_dim]) / (ts[self.w:] - ts[:-self.w])

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

