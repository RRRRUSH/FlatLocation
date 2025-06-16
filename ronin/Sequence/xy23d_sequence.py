import os.path as osp

import numpy as np
import pandas as pd
import quaternion

from Sequence.CompiledSequence import CompiledSequence
from Sequence.utils.data_precessed import compute_output_time, process_data_source

_nano_to_sec = 1e09


class xy23dsequence(CompiledSequence):
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
        source_vector = {'gyro', 'acce', 'pose'}
        source_quaternion = {'game_rv'}
        source_all = source_vector.union(source_quaternion)
        if path[-1] == '/':
            path = path[:-1]

        all_sources = {}
        for source in source_all:
            source_path = osp.join(path, source + '.txt')
            source_data = pd.read_csv(source_path, delimiter=' ', skiprows=1).values

            source_data[:, 0] = (source_data[:, 0]) / _nano_to_sec
            all_sources[source] = source_data

        output_time = compute_output_time(all_sources)
        processed_sources = {}
        for source in all_sources.keys():
            if source in source_vector:
                processed_sources[source] = process_data_source(all_sources[source], output_time, 'vector')
            else:
                processed_sources[source] = process_data_source(all_sources[source][:, [0, 4, 1, 2, 3]], output_time,
                                                                'quaternion')

        gyro = processed_sources['gyro']
        acce = processed_sources['acce']
        game_rv = processed_sources['game_rv']
        tango_pos = processed_sources['pose']

        ts = output_time
        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt

        ori_q = quaternion.from_float_array(game_rv)
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
        start_frame = self.info.get('start_frame', 0)

        self.ts = ts[start_frame:, None]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        self.targets = glob_v[start_frame:, :self.target_dim]
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)
