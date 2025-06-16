import json
import os
import random
import warnings
from os import path as osp

import h5py
import numpy as np
import quaternion
from torch.utils.data import Dataset
from Sequence.utils.save_as_npy import save_npy


class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)
        self.index = 0
        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])

            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


def load_cached_sequences(seq_type, root_dir, data_list, cache_path, **kwargs):
    grv_only = kwargs.get('grv_only', True)

    if cache_path is not None and cache_path not in ['none', 'invalid', 'None']:
        if not osp.isdir(cache_path):
            os.makedirs(cache_path)
        if osp.exists(osp.join(cache_path, 'config.json')):
            info = json.load(open(osp.join(cache_path, 'config.json')))
            if info['feature_dim'] != seq_type.feature_dim or info['target_dim'] != seq_type.target_dim:
                warnings.warn('The cached dataset has different feature or target dimension. Ignore')
                cache_path = 'invalid'
            if info.get('aux_dim', 0) != seq_type.aux_dim:
                warnings.warn('The cached dataset has different auxiliary dimension. Ignore')
                cache_path = 'invalid'
            if info.get('grv_only', 'False') != str(grv_only):
                warnings.warn('The cached dataset has different flag in "grv_only". Ignore')
                cache_path = 'invalid'
        else:
            info = {'feature_dim': seq_type.feature_dim, 'target_dim': seq_type.target_dim,
                    'aux_dim': seq_type.aux_dim, 'grv_only': str(grv_only)}
            json.dump(info, open(osp.join(cache_path, 'config.json'), 'w'))

    features_all, targets_all, aux_all = [], [], []
    for i in range(len(data_list)):
        if cache_path is not None and osp.exists(osp.join(cache_path, data_list[i] + '.hdf5')):
            with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5')) as f:
                feat = np.copy(f['feature'])
                targ = np.copy(f['target'])
                aux = np.copy(f['aux'])
        else:
            seq = seq_type(osp.join(root_dir, data_list[i]), **kwargs)
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            if cache_path is not None and osp.isdir(cache_path):
                with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5'), 'x') as f:
                    f['feature'] = feat
                    f['target'] = targ
                    f['aux'] = aux
            if kwargs.get('save_npy_to', None) is not None:
                root_p = os.path.join(kwargs['save_npy_to'], f"{kwargs.get('dataset_name', 'Train')}")
                os.makedirs(root_p, exist_ok=True)
                out_p = os.path.join(root_p, f"{len(os.listdir(root_p))}")
                save_npy(out_p, seq.features[:, 3:], seq.features[:, :3], seq.orientations, seq.gt_pos, seq.ts)
        features_all.append(feat)
        targets_all.append(targ)
        aux_all.append(aux)
    return features_all, targets_all, aux_all
