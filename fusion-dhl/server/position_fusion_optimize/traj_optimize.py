import argparse
import json

import numpy as np
from matplotlib import pyplot as plt

from server.utils.coordinate_utils import create_image_coordinate_by_lonlat, xys_to_lonlat_by_image_coordinate
from source.optim.optimization_config import Variable
from source.optim.optimizer import Cmd
from source.util.data_loader import trajectory_as_polar_velocity
from source.util.map_util import latlong_coordinates_on_map, match_timestamps, find_sparse
from source.util.other import load_config

def process_flp_data(map, dpi, map_data, points, visualize=True):
    return latlong_coordinates_on_map(map, dpi, map_data, points, visualize)


def optimize(traj, flp, visualize=False):
    traj = np.array(traj)
    flp = np.array(flp)
    with open('./data_paths.json', 'r') as f:
        default_config = json.load(f)
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help="[Optional] file to load configuration from")
    parser.add_argument('--data_path', type=str, default=None, help="Path to folder containing hdf5 and json files")
    parser.add_argument('--data_list', type=str, default=None)
    parser.add_argument('--data_dir', type=str)

    parser.add_argument('--type', type=str, choices=['gt', 'ronin', 'raw'])
    parser.add_argument('--traj_freq', type=int, help='Frequency of trajectory data')

    parser.add_argument('--out_dir', type=str, required=False)
    parser.add_argument('--prefix', type=str)

    # specify a single map_file with data_path or specify map_dir and map_csv_dir (required only if flp priors are used)
    parser.add_argument('--map_path', type=str, default=None, help="Path to map image")
    parser.add_argument('--map_dpi', type=int, default=2.5)
    parser.add_argument('--map_latlong_path', type=str, default=None,
                        help="Constraints to map floorplan with FLP data (default: <map_path>.csv)")
    parser.add_argument('--map_dir', type=str, default=default_config.get('floorplan_dir', None))
    parser.add_argument('--map_csv_dir', type=str, default=None)

    parser.add_argument('--starting_variables', type=str,
                        help='Path to file containing optimization varibles (ensure same number of parameter)')
    parser.add_argument('--pos_prior_type', type=str, default='flp', choices=['flp', 'manual', 'from_file'],
                        help='Position prior type [flp or manual positions are in hdf5 file, or specify input file')
    parser.add_argument('--flp_adjust_radius', type=float, default=1, help="Factor to adjust FLP radius")
    parser.add_argument('--flp_sample', type=int, default=10, help="Sparse FLP points, sample every x seconds")
    parser.add_argument('--pos_prior_file', type=str, default=None,
                        help='Path to file containing positions priors, if pos_prior_type is from file')

    # optimization params
    parser.add_argument('--scale', type=str, choices=Variable.get_var_states(False), help='Status of scale param')
    parser.add_argument('--scale_interval', type=int,
                        help='Variable interval for piecewise linear function of scale (0 '
                             'for constant)')
    parser.add_argument('--bias', type=str, choices=Variable.get_var_states(False), help='Status of bias param')
    parser.add_argument('--bias_interval', type=int, help='Variable interval for piecewise linear function of bias')

    # optimization_functions
    parser.add_argument('--pos_prior_weight', type=float, help="Weight for manual position prior (-1 for no prior)")
    parser.add_argument('--pos_prior_loss', type=float, help="Norm loss function for pos prior (-1 for default)")
    parser.add_argument('--interpolate_kind', type=str, default="quadratic", choices=["linear", "quadratic", "cubic"],
                        help="Interpolation type")

    parser.add_argument('--scale_reg_weight', type=float, help="Weight for scale regularization (-1 for no prior)")
    parser.add_argument('--bias_reg_weight', type=float, help="Weight for bias regularization (-1 for no prior)")

    # optimization parameters
    parser.add_argument('--n_iterations', type=int, default=50)
    parser.add_argument('--loop', action='store_true', help='When set, iterate until the energy is stable (with cmd)')
    parser.add_argument('--verbose', action='store_true', help='When set progress of optimization is printed')
    parser.add_argument('--no_gui', action='store_true', help='If true, run optimization without gui')

    args = parser.parse_args()

    args, _ = load_config('./default_config.json', args)

    ts, vel_norm, vel_angle, traj, _ = trajectory_as_polar_velocity(np.flip(traj[:, 1:3], axis=1), 200, visualize=False,
                                                                    timestamp=traj[:, 0])
    map_data = create_image_coordinate_by_lonlat(flp[:,[1,2]],args.map_dpi)

    map_image = np.random.rand(int(map_data[:,0].max()), int(map_data[:,1].max()), 3)

    flp_data = process_flp_data(map_image, args.map_dpi, map_data, flp, visualize=False)

    if flp_data is not None:
        flp_data[:, -1] /= args.flp_adjust_radius
        if args.flp_sample > 1:
            flp_data = find_sparse(args.flp_sample, flp_data[:, 0], flp_data[:, 1:])
        map_aligned_points = match_timestamps(ts, flp_data[:, 0], flp_data[:, 1:], allow_err=1)
    
    optimizer = Cmd('', '', traj, args, ts, vel_norm, vel_angle, map_aligned_points, {}, loop=False)
    optimizer.map_data = map_data
    if visualize:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.plot(optimizer.refined_traj[:, 0] , -optimizer.refined_traj[:, 1], label='optimizer_traj',color='blue')
        ax.scatter(optimizer.map_aligned_points[:, 1] , -optimizer.map_aligned_points[:, 2], label='optimizer_point',color='red')
        ax.legend()
        # 设置 x 轴和 y 轴刻度标签的字体大小
        plt.xticks(fontsize=14)  # 设置 x 轴刻度标签字体大小为 14
        plt.yticks(fontsize=14)  # 设置 y 轴刻度标签字体大小为 14
        plt.show()

    return optimizer



if __name__ == '__main__':
    # with h5py.File('./data.hdf5', 'r') as f:
    #     traj = np.copy(f['computed/ronin_traj'])
    #     time = np.copy(f['synced/time'])
    #     if traj.shape[-1] == 2:
    #         traj = np.concatenate([traj, np.zeros([traj.shape[0], 1])], axis=-1)
    #     flp = np.copy(f['filtered/flp'])

    #     time = [[i] for i in time]
    #     traj = np.concatenate([time, traj], axis=1)
    
    import pandas as pd
    pos  = pd.read_csv('/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/abc_pos.csv',names=['x','y'])
    time = pd.read_csv('/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/abc_time.csv',names=['time'])
    uwb  = pd.read_csv('/home/a/桌面/git/Fusion-DHL-master/server/pos_csv/abc_uwb.csv',names=['time','x','y','r'])

    traj = np.concatenate([time,pos], axis=-1)
    flp = np.array(uwb)


    # flp = np.array([[time[0][0],39.10192122646628,117.06021988389575,1]])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.plot(traj[:, 1], traj[:, 2], label='traj', color='blue')
    plt.xticks(fontsize=18)  # 设置 x 轴刻度标签字体大小为 14
    plt.yticks(fontsize=18)  # 设置 y 轴刻度标签字体大小为 14
    # ax.scatter(flp[:, 1], flp[:, 2], label='flp', color='red')
    ax.legend()
    plt.show()
    optimize(traj, flp, True)
