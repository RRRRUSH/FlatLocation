import os

import numpy as np
import pandas
import pandas as pd
import quaternion
import torch

from Sequence.CompiledSequence import CompiledSequence
import os.path as osp

from Sequence.utils.save_as_npy import save_npy


class STEPSequence(CompiledSequence):
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        if path[-1] == '\\':
            path = path[:-1]

        data = pd.read_csv(path, delimiter=',',header=None).values

        ts = data[:, 0].reshape(-1, 1) / 1e03  # s
        gyro = data[:, [1, 2, 3]].astype(np.float64)  # rad/s
        acce = data[:, [4, 5, 6]].astype(np.float64)  # m/s^2
        rv = data[:, [7, 8, 9, 10]].astype(np.float64)
        tango_pos = data[:, [11, 12]].astype(np.float64)  # m
        pos = data[:, [11, 12, 13]].astype(np.float64)  # m

        ori_v = quaternion.from_float_array(rv)

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros((gyro.shape[0], 1)), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros((acce.shape[0], 1)), acce], axis=1))

        gyro_glob = quaternion.as_float_array(ori_v * gyro_q * ori_v.conj())[:, 1:]
        acce_glob = quaternion.as_float_array(ori_v * acce_q * ori_v.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        self.targets = (tango_pos[self.w:, :self.target_dim] - tango_pos[:-self.w, :self.target_dim]) / (
                ts[self.w:] - ts[:-self.w])
        self.gt_pos = tango_pos
        self.orientations = quaternion.as_float_array(ori_v)

        # path = osp.join('/data/imu_data/TLIO/train_data/stepdata', path.split('/')[-2],
        #                 path.split('/')[-1].split('.')[0])
        # # acce_glob, gyro_glob 已经经过四元数旋转
        # save_npy(path, acce_glob, gyro_glob, ori_v, tango_pos, ts)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)
