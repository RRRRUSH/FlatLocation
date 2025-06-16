import json
import os
import os.path as osp

import numpy as np
import quaternion


# path = osp.join('/home/a/Desktop/git/TLIO/data/train_data/imunet', path.split('/')[-1])
# save_npy(path, acce_glob, gyro_glob, ori, tango_pos, ts)

def save_npy(path, glob_acce, glob_gyro, ori_q, tango_pos, ts):
    if glob_gyro is not None and glob_acce is not None and ori_q is not None and tango_pos is not None:
        print(path, glob_gyro.shape, glob_acce.shape, ori_q.shape, tango_pos.shape)
        ts *= 1e06
        # tango_pos = np.concatenate([tango_pos, np.zeros((tango_pos.shape[0], 1))], axis=1)
        os.makedirs(path, exist_ok=True)
        np.save(osp.join(path, 'imu0_resampled.npy'),
                np.concatenate(
                    [ts.reshape(-1, 1), glob_gyro, glob_acce, ori_q, tango_pos,
                    tango_pos], axis=-1))
        description = {
            "columns_name(width)": [
                "ts_us(1)",
                "gyr_compensated_rotated_in_World(3)",
                "acc_compensated_rotated_in_World(3)",
                "qxyzw_World_Device(4)",
                "pos_World_Device(3)",
                "vel_World(3)"
            ],
            "num_rows": int(glob_gyro.shape[0]),
            "approximate_frequency_hz": 200.0,
            "t_start_us": float(ts[0]),
            "t_end_us": float(ts[-1])
        }
        with open(osp.join(path, 'imu0_resampled_description.json'), 'w') as f:
            json.dump(description, f, indent=4)
    else:
        print(f"Invalid data! {path}")

def save_txt(path, acce, gyro, ori_q, ts):
    save_path = os.path.join("/home/a/Desktop/git/plot_trj/pos", path.split("/")[-1])
    print(gyro.shape, acce.shape, quaternion.as_float_array(ori_q).shape, ts.shape)
    ts = ts - ts[0]
    os.makedirs(save_path, exist_ok=True)
    np.savetxt(os.path.join(save_path, 'acce.txt'), np.concatenate([ts * 1e09, acce],axis=-1))
    np.savetxt(os.path.join(save_path, 'gyro.txt'), np.concatenate([ts * 1e09, gyro],axis=-1))
    np.savetxt(os.path.join(save_path, 'game_rv.txt'), np.concatenate([ts* 1e09, quaternion.as_float_array(ori_q)],axis=-1))
