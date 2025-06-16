import numpy as np
import pandas as pd

from server.position_fusion_optimize.minimize_error_shape_fit import ShapeFit
from server.position_fusion_optimize.traj_optimize import optimize
from server.utils.coordinate_utils import join_xys_to_lonlat_by_yaw, xys_to_lonlat_by_image_coordinate


def run_optimize(traj_pos, flp_pos, new_traj_pos):
    optimized = optimize(traj_pos, flp_pos)
    # optimized.keys
    # dict_keys(['map', 'dpi', 'origin_traj', 'refined_traj', 'args', 'vel_norm', 'vel_angle', 'ts', 'map_aligned_points',
    # 'origin_params', 'loop', 'prev_pos_energy', 'current_pos_energy', 'variables', 'functions', 'compound_functor',
    # 'starting_params', 'solution', 'lsq', 'refined_yaw', 'status', 'v_pos', 'v_yaw', 'v_scale', 'v_bias', 'v_noise', 'map_data'])
    # print(vars(optimized)['functions'])
    # print(optimized.map_data)
    # print(optimized.refined_traj[:, 0:2])

    imu_lonlat = xys_to_lonlat_by_image_coordinate(optimized.map_data, optimized.refined_traj[:, 0:2])
    new_imu_lonlat = np.array([])
    if new_traj_pos.shape[0] > 0:
        new_imu_lonlat = join_xys_to_lonlat_by_yaw(new_traj_pos, imu_lonlat[-1], optimized.refined_yaw[-1])

    return imu_lonlat, new_imu_lonlat, optimized


def run_shape_fit(traj_pos, flp_pos,new_traj_pos):
    shape_fit = ShapeFit()
    imu_lonlat = shape_fit.fit(traj_pos, flp_pos)

    new_imu_lonlat = np.array([])
    if new_traj_pos.shape[0] > 0:
        new_imu_lonlat = shape_fit.new_xys_to_lonlat(new_traj_pos)

    return imu_lonlat, new_imu_lonlat


if __name__ == '__main__':
    path = r'E:\HaozhanLi\Project\FlatLoc\INS\fusion-dhl\tests\\'
    dir_list = pd.read_csv(f'{path}\list.txt', header=None).values

    for dirs in dir_list:
        dir_name = dirs[0]
        # traj_pos: time, x, y
        traj_pos = pd.read_csv(path + dir_name + '/pred_traj.csv', header=None).values
        # rtk_pos: time, x, y, error
        rtk_pos = pd.read_csv(path + dir_name + '/pred_rtk.csv', header=None).values
        # align timestamp
        # time_pos_index_map: imu_ts, rtk_ts
        time_pos_index_map = []
        for j in range(rtk_pos.shape[0]):
            for i in range(1, traj_pos.shape[0]):
                if i >= traj_pos.shape[0] - 1:
                    continue
                if traj_pos[i, 0] <= rtk_pos[j, 0] <= traj_pos[i + 1, 0]:
                    time_pos_index_map.append([i, j])
                    break
        time_pos_index_map = np.array(time_pos_index_map)
        print('time_pos_index_map', time_pos_index_map.shape)

        optimize_traj, add_traj, _ = run_optimize(
            # traj_pos,
            # rtk_pos
            traj_pos[:time_pos_index_map[:, 0].max()],
            rtk_pos[time_pos_index_map[:, 1].min():time_pos_index_map[:, 1].max()],
            traj_pos[time_pos_index_map[:, 0].max():, [1, 2]]
        )

        optimize_traj = np.vstack((optimize_traj, add_traj))
        optimize_traj = np.concatenate([traj_pos[:, [0]], optimize_traj[:, [0, 1]]], axis=1)
        with open(f'{path}{dir_name}/optimize_traj.csv', 'w') as f:
            for i in range(optimize_traj.shape[0]):
                f.write(f'{optimize_traj[i, 0]},{optimize_traj[i, 1]},{optimize_traj[i, 2]}\n')

        fit_traj, add_traj = run_shape_fit(
            # traj_pos,
            # rtk_pos
            traj_pos[:time_pos_index_map[:, 0].max()],
            rtk_pos[time_pos_index_map[:, 1].min():time_pos_index_map[-1, 1].max()],
            traj_pos[time_pos_index_map[:, 0].max():, [1, 2]]
        )

        fit_traj = np.vstack((fit_traj, add_traj))
        fit_traj = np.concatenate([traj_pos[:, [0]], fit_traj[:, [0, 1]]], axis=1)
        with open(f'{path}{dir_name}/fit_traj.csv', 'w') as f:
            for i in range(fit_traj.shape[0]):
                f.write(f'{fit_traj[i, 0]},{fit_traj[i, 1]},{fit_traj[i, 2]}\n')