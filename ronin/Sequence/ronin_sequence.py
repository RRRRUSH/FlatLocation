import json
import os

import h5py
import numpy as np
import quaternion
from Sequence.CompiledSequence import CompiledSequence
import os.path as osp

from Sequence.utils.save_as_npy import save_npy, save_txt


class RONINSequence(CompiledSequence):
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
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
            data_path, self.max_ori_error, self.grv_only)

        with h5py.File(osp.join(data_path, 'data.hdf5'), mode='r') as f:
            gyro_uncalib = f['synced/gyro_uncalib']
            acce_uncalib = f['synced/acce']
            ts = np.copy(f['synced/time'])

            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
            acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))
            tango_pos = np.copy(f['pose/tango_pos'])
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])

        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q

        dt = (ts[self.w:] - ts[:-self.w])[:, None]
        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt

        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        start_frame = self.info.get('start_frame', 0)
        self.ts = ts[start_frame:]
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
        self.targets = glob_v[start_frame:, :self.target_dim]
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
        self.gt_pos = tango_pos[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)


def angular_velocity_to_quaternion_derivative(q, w):
    omega = np.array([[0, -w[0], -w[1], -w[2]],
                      [w[0], 0, w[2], -w[1]],
                      [w[1], -w[2], 0, w[0]],
                      [w[2], w[1], -w[0], 0]]) * 0.5
    return np.dot(omega, q)


def gyro_integration(ts, gyro, init_q):
    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = output_q[i - 1] + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1]) * dts[
            i - 1]
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q


def select_orientation_source(data_path, max_ori_error=20.0, grv_only=True, use_ekf=True):
    ori_names = ['gyro_integration', 'game_rv']
    ori_sources = [None, None, None]

    with open(osp.join(data_path, 'info.json')) as f:
        info = json.load(f)
        ori_errors = np.array(
            [info['gyro_integration_error'], info['grv_ori_error'], info['ekf_ori_error']])
        init_gyro_bias = np.array(info['imu_init_gyro_bias'])

    with h5py.File(osp.join(data_path, 'data.hdf5'), mode='r') as f:
        ori_sources[1] = np.copy(f['synced/game_rv'])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append('ekf')
                ori_sources[2] = np.copy(f['pose/ekf_ori'])
            min_id = np.argmin(ori_errors[:len(ori_names)])
            # Only do gyro integration when necessary.
            if min_id == 0:
                ts = f['synced/time']
                gyro = f['synced/gyro_uncalib'] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]
