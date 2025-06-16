import os

import numpy as np
import pandas
import quaternion
from Sequence.CompiledSequence import CompiledSequence
import os.path as osp

from Sequence.utils.save_as_npy import save_npy


class RUNSequence(CompiledSequence):
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
        if path[-1] == '/':
            path = path[:-1]

        data = np.loadtxt(osp.join(path, 'processed/data.csv'),delimiter=',',skiprows=1)
        ts = data[:,1].reshape(-1, 1)
        gyro = data[:,[2,3,4]]
        acce = data[:,[5,6,7]]
        rv = data[:,[8,9,10,11]]
        tango_pos = data[:,[12,13]]

        ori_v = quaternion.from_float_array(rv)

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros((gyro.shape[0], 1)), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros((acce.shape[0], 1)), acce], axis=1))

        gyro_glob = quaternion.as_float_array(ori_v * gyro_q * ori_v.conj())[:, 1:]
        acce_glob = quaternion.as_float_array(ori_v * acce_q * ori_v.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        self.targets = (tango_pos[self.w:, :self.target_dim] - tango_pos[:-self.w, :self.target_dim]) / (ts[self.w:] - ts[:-self.w])
        self.gt_pos = tango_pos
        self.orientations = quaternion.as_float_array(ori_v)


        # path = osp.join('/data/imu_data/TLIO/train_data/rundata', path.split('/')[-1])
        # save_npy(path, acce_glob, gyro_glob, ori_v, tango_pos, ts)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

