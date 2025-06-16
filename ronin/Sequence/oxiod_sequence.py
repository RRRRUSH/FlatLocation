import os
from os import path as osp
import numpy as np
import pandas
import quaternion
from Sequence.CompiledSequence import CompiledSequence
from Sequence.utils.save_as_npy import save_npy, save_txt


class OXIODSequence(CompiledSequence):
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

        imu_all = pandas.read_csv(path)

        ts = imu_all[['time']].values
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        tango_pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values

        game_rv = quaternion.from_float_array(imu_all[['rv_w','rv_x', 'rv_y', 'rv_z']].values)

        ori = game_rv

        # save_txt(path, acce, gyro, ori, ts)

        nz = np.zeros(ts.shape)
        gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

        gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
        acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        self.targets = (tango_pos[self.w:, :self.target_dim] - tango_pos[:-self.w, :self.target_dim]) / (ts[self.w:] - ts[:-self.w])
        self.gt_pos = tango_pos
        self.orientations = quaternion.as_float_array(game_rv)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

