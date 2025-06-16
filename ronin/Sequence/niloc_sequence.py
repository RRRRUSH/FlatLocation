import os
from os import path as osp
import h5py
import numpy as np
import quaternion
from Sequence.CompiledSequence import CompiledSequence


class NILOCSequence(CompiledSequence):
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with h5py.File(f"{data_path}.hdf5", mode='r') as f:
            gyro = np.copy(f['synced/gyro'])
            acce = np.copy(f['synced/acce'])
            ts = np.copy(f['synced/time'])
            tango_pos = np.copy(f['pose/tango_pos'])
            ori = np.copy(f['synced/game_rv'])

        ori_q = quaternion.from_float_array(ori)

        save_path = os.path.join("/home/a/Desktop/git/plot_trj/pos", data_path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)
        np.savetxt(os.path.join(save_path, 'acce.txt'), np.concatenate([ts[:,None] * 1e09, acce],axis=-1))
        np.savetxt(os.path.join(save_path, 'gyro.txt'), np.concatenate([ts[:,None] * 1e09, gyro],axis=-1))
        np.savetxt(os.path.join(save_path, 'game_rv.txt'), np.concatenate([ts[:,None] * 1e09, ori],axis=-1))

        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / (ts[self.w:] - ts[:-self.w])[:, None]

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
        self.targets = glob_v[:, :self.target_dim]
        self.orientations = quaternion.as_float_array(ori_q)
        self.gt_pos = tango_pos
        # save_path= os.path.join("/home/a/Desktop/git/plot_trj/pos",data_path.split("/")[-1])
        # os.makedirs(save_path, exist_ok=True)
        # np.save(os.path.join(save_path, 'feat.npy'), self.features)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)
