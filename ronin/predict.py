import os
import time
import glob

import numpy as np
import torch
import argparse

# from datacache_without_rv import DataCache
from utils.datacache import DataCache
from model_library.model_factory import get_model
from utils.logging import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def com_data(data, dim=2, save_path='/', show=False):
    window_size = 200
    step_size = 10
    cache = DataCache(dim=dim, show=show)
    cache.imu_data.preprocessing_data(data)
    logging.info('Total imu_data : {}'.format(len(cache.imu_data)))
    feat_all = []
    times_preds = []
    while len(cache.imu_data) >= window_size:
        feat = cache.imu_data.get_one_feat(window_size, step_size)
        feat_all.append(feat[0,:].T[:step_size].tolist())
        with torch.no_grad():
            pred = network(feat.to(device)).cpu().detach().numpy()
        times_preds.append([time.time()] + pred[0, :].tolist())
    cache.com_preds(times_preds)
    times_preds.clear()
    cache.hand_io(save_path)
    feat_all = np.concatenate(feat_all, axis=0)
    np.savetxt(os.path.join(save_path, 'feat.txt'), feat_all, fmt='%.6f')


def load_data_from_txt(dir_path):
    logging.info('load data from txt')
    gyro = np.loadtxt(os.path.join(dir_path, 'gyro.txt'))
    acce = np.loadtxt(os.path.join(dir_path, 'acce.txt'))
    game_rv = np.loadtxt(os.path.join(dir_path, 'game_rv.txt'))
    return {'gyro': gyro, 'acce': acce, 'game_rv': game_rv}


def load_data_from_csv(dir_path):
    logging.info('load data from csv')
    gyro = np.loadtxt(os.path.join(dir_path, 'gyro.csv'), delimiter=',')
    acce = np.loadtxt(os.path.join(dir_path, 'acce.csv'), delimiter=',')
    game_rv = np.loadtxt(os.path.join(dir_path, 'game_rv.csv'), delimiter=',')
    return {'gyro': gyro, 'acce': acce, 'game_rv': game_rv}


def load_data_from_npy(dir_path):
    logging.info('load data from npy')
    data = np.load(os.path.join(dir_path, 'imu0_resampled.npy'))
    data[:, 0] = data[:, 0] / 1e06
    gyro = data[:, [0, 1, 2, 3]]
    acce = data[:, [0, 4, 5, 6]]
    return {'gyro': gyro, 'acce': acce}


if __name__ == "__main__":
    s = time.time()
    dim = 2
    # 3d的基础模型
    # '/home/a/Desktop/git/plot_trj/model/3d_base.pt'
    # 2d的基础模型
    # '/home/a/Desktop/git/plot_trj/model/2d_base.pt'
    # 3d的改进模型
    # '/home/a/Desktop/git/plot_trj/model/1z5data400.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r'E:\HaozhanLi\Project\FlatLoc\ronin_predict\models\RONIN\ronin_ridi_imunet_tlio_oxiod\2d_base.pt')
    parser.add_argument('--data_dir', type=str, default=r'E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\input\lsfq_verify_70s')
    parser.add_argument('--out_dir', type=str, default=r'E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\output\ronin')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--show', action='store_true', default=False)
    
    args = parser.parse_args()

    subdir = [item for item in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(item)]

    for i, sub_data_dir in enumerate(subdir):
        logging.info(f'Processing {i+1} / {len(subdir)}')
        
        network = get_model(args.model_path, dim=args.dim, ronin_instance=args.arch)
        network.eval().to(device)

        logging.info('load model from {}'.format(args.model_path))
        logging.info('load data from {}'.format(sub_data_dir))
        logging.info('Total number of parameters: {}'.format(network.get_num_params()))

        data = load_data_from_txt(sub_data_dir)
        # data = load_data_from_csv(data_dir)
        # data = load_data_from_npy(data_dir)
        com_data(data, dim=args.dim, show=args.show, save_path=os.path.join(
                args.out_dir, os.path.basename(sub_data_dir)
            )) # data中的数据的时间戳单位是s

        logging.info('time cost:{}'.format(time.time() - s))
