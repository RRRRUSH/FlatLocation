from base import *

def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.

    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated

if __name__ == "__main__":
    imu_raw = pd.read_csv(
        r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\predict\raw\vio_calib\0326_0\gyro_accel.csv", 
        sep=","
    ).values
    imu_raw[:, 0] = imu_raw[:, 0] / 1e09

    extr_param = pd.read_csv(
        r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\predict\raw\vio_calib\0326_0\WQ_01_test.csv", 
        sep=","
    ).values
    print(imu_raw.shape, extr_param.shape)

    # CALI 30 hz
    dt = extr_param[1:, 0] - extr_param[:-1, 0]
    print(np.mean(dt), np.std(dt))

    # IMU 500 hz
    dt = imu_raw[1:, 0] - imu_raw[:-1, 0]
    print(np.mean(dt), np.std(dt))

    # sync
    duration = np.where((imu_raw[:, 0] >= extr_param[0, 0]) & (imu_raw[:, 0] <= extr_param[-1, 0]))[0]
    raw_index = np.where(~((imu_raw[:, 0] >= extr_param[0, 0]) & (imu_raw[:, 0] <= extr_param[-1, 0])))[0]
    print(raw_index)

    # downsample
    imu_sync = imu_raw[duration]

    extr_sync = interpolate_vector_linear(extr_param[:, 11:17], extr_param[:, 0], imu_sync[:, 0])
    print(imu_sync.shape, extr_sync.shape)
    
    plt.subplot(2, 1, 1)
    plt.scatter(imu_sync[:, 0], imu_sync[:, 1], label="gyro_raw_x", s=2, c='b')
    plt.scatter(imu_sync[:, 0], imu_sync[:, 2], label="gyro_raw_y", s=2, c='r')
    plt.scatter(imu_sync[:, 0], imu_sync[:, 3], label="gyro_raw_z", s=2, c='g')
    plt.legend()

    imu_calib = np.zeros(imu_sync.shape)
    imu_calib[:, 1:] = imu_sync[:, 1:] - extr_sync[:, :]
    
    plt.subplot(2, 1, 2)
    plt.scatter(imu_sync[:, 0], imu_calib[:, 1], label="gyro_calib_bx", s=2, c='b')
    plt.scatter(imu_sync[:, 0], imu_calib[:, 2], label="gyro_calib_by", s=2, c='r')
    plt.scatter(imu_sync[:, 0], imu_calib[:, 3], label="gyro_calib_bz", s=2, c='g')
    plt.legend()
    plt.show()

    res = np.zeros((imu_raw.shape[0], 7))
    res[raw_index, :] = imu_raw[raw_index, :]
    res[duration, :] = imu_calib[:, :]

    np.savetxt(r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\predict\raw\vio_calib\0326_0\imu_calib.csv", res, delimiter=",")