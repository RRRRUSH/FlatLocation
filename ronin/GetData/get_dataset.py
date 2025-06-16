import json
import math

import numpy as np
from torch.utils.data import ConcatDataset

from Sequence import *

from Sequence.strided_sequence_dataset import StridedSequenceDataset

sequence_classes = {
    "RONINSequence": RONINSequence,
    "RIDISequence": RIDISequence,
    "OXIODSequence": OXIODSequence,
    "TLIOSequence": TLIOSequence,
    "RONINVIOSequence": RONINVIOSequence,
    "NILOCSequence": NILOCSequence,
    "xy23dsequence": xy23dsequence,
    "runsequence":RUNSequence,
    "stepsequence":STEPSequence,
}


class RandomHoriRotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, feat, targ, **kwargs):

        dim=targ.shape[0]
        angle = np.random.random() * self.max_angle
        if dim == 3:
            rm = np.array([[math.cos(angle), -math.sin(angle), 0],
                           [math.sin(angle), math.cos(angle), 0],
                           [0, 0, 1]])
            feat_aug = np.copy(feat)
            targ_aug = np.copy(targ)
            feat_aug[:, :3] = np.matmul(rm, feat[:, :3].T).T
            feat_aug[:, 3:6] = np.matmul(rm, feat[:, 3:6].T).T
            targ_aug[:3] = np.matmul(rm, targ[:3].T).T
        else:
            rm = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
            feat_aug = np.copy(feat)
            targ_aug = np.copy(targ)
            feat_aug[:, :2] = np.matmul(rm, feat[:, :2].T).T
            feat_aug[:, 3:5] = np.matmul(rm, feat[:, 3:5].T).T
            targ_aug[:2] = np.matmul(rm, targ[:2].T).T

        return feat_aug, targ_aug


def get_dataset(dataset_path, data_list_path, seq_type, dataset_name=None, save_npy_to=None):
    with open(data_list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']

    return StridedSequenceDataset(
        seq_type, dataset_path, data_list,None, 10, 200,
        dataset_name=dataset_name, save_npy_to=save_npy_to,
        random_shift=5, transform=RandomHoriRotate(math.pi * 2),
        shuffle=True, grv_only=False, max_ori_error=20, num_workers=8)


def load_datasets_from_json(c_path: str, set_type: str, save_npy=r"E:\HaozhanLi\Project\FlatLoc\ronin\Datasets\test"):
    with open(c_path, 'r') as f:
        config = json.load(f)
    datasets = []
    for dataset_name in config['selected_datasets']:
        dataset_config = config['datasets'][dataset_name]
        list_path = dataset_config[f'{set_type}_list_path']
        dataset_path = dataset_config[f'{set_type}_dataset_path']
        sequence_class = sequence_classes[dataset_config['sequence_class']]
        dataset = get_dataset(
            dataset_path, list_path, sequence_class,
            dataset_name=dataset_name, save_npy_to=save_npy
        )
        datasets.append(dataset)
    return ConcatDataset(datasets)


if __name__ == '__main__':
    config_path = r"E:\HaozhanLi\Project\FlatLoc\ronin\GetData\config.json"
    load_datasets_from_json(config_path, 'train')
