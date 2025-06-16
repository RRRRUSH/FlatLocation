from os import path as osp
import numpy as np
from Sequence.CompiledSequence import CompiledSequence


class UnifiedSequence(CompiledSequence):
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

        all_data = np.load(osp.join(path, 'imu0_resampled.npy'))

        ts = all_data[:, 0]
        ts = ts.reshape(-1, 1)
        gyro = all_data[:, 1:4]
        acce = all_data[:, 4:7]
        tango_pos = all_data[:, 11:14]

        self.ts = ts
        self.features = np.concatenate([gyro, acce], axis=1)
        self.targets = (tango_pos[self.w:, :self.target_dim] - tango_pos[:-self.w, :self.target_dim]) / (ts[self.w:] - ts[:-self.w])
        self.gt_pos = tango_pos

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.gt_pos], axis=1)

