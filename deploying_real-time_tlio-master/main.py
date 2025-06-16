import argparse
import asyncio
import warnings
import numpy as np
from numba.core.errors import NumbaPerformanceWarning

from src.hipnuc.cmd_list import cmd_list
from src.tracker.imu_tracker_runner import ImuTrackerRunner
from src.utils.argparse_utils import add_bool_arg
from src.utils.logging import logging

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

if __name__ == "__main__":
    ports = cmd_list()
    parser = argparse.ArgumentParser()

    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")
    io_groups.add_argument("--port", type=str, default="COM11", )
    io_groups.add_argument("--baudrate", type=int, default=921600)
    io_groups.add_argument("--wifi", default=False, action="store_true")
    io_groups.add_argument("--model_path", type=str, default=r"resource/model/model_torchscript.pt")
    io_groups.add_argument("--model_param_path", type=str, default=r'resource/model/parameters.json')

    # ----------------------- filter params -----------------------
    filter_group = parser.add_argument_group("filter tuning:")
    filter_group.add_argument("--update_freq", type=float, default=20.0)  # (Hz)
    filter_group.add_argument(
        "--sigma_na", type=float, default=np.sqrt(1e-3)
    )  # accel noise  m/s^2
    filter_group.add_argument(
        "--sigma_ng", type=float, default=np.sqrt(1e-4)
    )  # gyro noise  rad/s
    filter_group.add_argument(
        "--ita_ba", type=float, default=1e-4
    )  # accel bias noise  m/s^2/sqrt(s)
    filter_group.add_argument(
        "--ita_bg", type=float, default=1e-6
    )  # gyro bias noise  rad/s/sqrt(s)
    filter_group.add_argument(
        "--init_attitude_sigma", type=float, default=1.0 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument(
        "--init_yaw_sigma", type=float, default=0.1 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument("--init_vel_sigma", type=float, default=1.0)  # m/s
    filter_group.add_argument("--init_pos_sigma", type=float, default=0.001)  # m
    filter_group.add_argument(
        "--init_bg_sigma", type=float, default=0.0001
    )  # rad/s  0.001
    filter_group.add_argument("--init_ba_sigma", type=float, default=0.02)  # m/s^2  0.02
    filter_group.add_argument("--g_norm", type=float, default=9.80)
    filter_group.add_argument("--meascov_scale", type=float, default=10.0)
    add_bool_arg(
        filter_group, "initialize_with_vio", default=False
    )  # initialize state with gt state
    add_bool_arg(
        filter_group, "initialize_with_offline_calib", default=False
    )  # initialize bias state with offline calib or 0
    add_bool_arg(
        filter_group, "calib", default=False
    )
    filter_group.add_argument(
        "--mahalanobis_fail_scale", type=float, default=0
    )  # if nonzero then mahalanobis gating test would scale the covariance by this scale if failed

    # ----------------------- debug params -----------------------
    debug_groups = parser.add_argument_group("debug")
    # covariance alternatives (note: if use_vio_meas is true, meas constant with default value 1e-4)
    add_bool_arg(debug_groups, "use_const_cov", default=False)
    debug_groups.add_argument(
        "--const_cov_val_x", type=float, default=np.power(0.1, 2.0)
    )
    debug_groups.add_argument(
        "--const_cov_val_y", type=float, default=np.power(0.1, 2.0)
    )
    debug_groups.add_argument(
        "--const_cov_val_z", type=float, default=np.power(0.1, 2.0)
    )

    add_bool_arg(
        debug_groups, "add_sim_meas_noise", default=False
    )  # adding noise on displacement measurement when using vio measurement
    debug_groups.add_argument(
        "--sim_meas_cov_val", type=float, default=np.power(0.01, 2.0)
    )
    debug_groups.add_argument(
        "--sim_meas_cov_val_z", type=float, default=np.power(0.01, 2.0)
    )
    debug_groups.add_argument(
        "--force_cpu", type=bool, default=False
    )

    args = parser.parse_args()

    np.set_printoptions(linewidth=2000)

    logging.info("Running")
    try:
        trackerRunner = ImuTrackerRunner(args)

        print("model load")
        trackerRunner.async_run_tracker()
        # asyncio.run(trackerRunner.async_run_tracker())
        # trackerRunner.run_tracker()
    except FileExistsError as e:
        print(e)
    except OSError as e:
        print(e)
    except KeyboardInterrupt as e:
        print(e)
