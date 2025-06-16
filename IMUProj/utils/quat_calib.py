import numpy as np
import quaternion
import pandas as pd
import matplotlib.pyplot as plt

def find_offset(q_ext, q_imu):
    """
    计算IMU相对于外部设备的姿态偏移量q_offset
    :param q_ext: 四元数，外部设备的姿态
    :param q_imu: 四元数，IMU的姿态
    :return: 四元数，IMU相对于外部设备的姿态偏移量q_offset
    """
    # 假设我们知道在某个时刻两者的姿态一致，那么q_ext == q_imu * q_offset^-1
    # 因此，我们可以求解得到q_offset = q_imu^-1 * q_ext
    q_offset = q_imu.inverse() * q_ext
    return q_offset

def calibrate_imu(q_ext, q_imu, q_offset):
    """
    根据外部设备的姿态角校准IMU的姿态角
    :param q_ext: 四元数，外部设备的姿态
    :param q_imu: 四元数，IMU的姿态
    :param q_offset: 四元数，IMU相对于外部设备的姿态偏移量
    :return: 四元数，校准后的IMU姿态
    """
    # 使用公式q_calibrated = q_ext * q_offset^-1 * q_imu 进行校准
    q_calibrated = q_ext * q_offset.inverse() * q_imu
    return q_calibrated


ref = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\input\quat_interp_test\nvidia_10r_interpq\game_rv.txt", sep=" ", header=None).values
imu = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\input\quat_interp_test\nvidia_10r_rawq\game_rv.txt", sep=" ", header=None).values

raw = imu.copy()

# 示例数据：假设我们知道在某一时刻外部设备和IMU的姿态如下
ref_q = quaternion.from_float_array(np.mean(ref[:1, -4:], axis=0)) if ref.shape[0] > 1 else quaternion.from_float_array(ref[0, -4:])  # 外部设备的姿态
imu_q = quaternion.from_float_array(np.mean(imu[:1, -4:], axis=0)) if imu.shape[0] > 1 else quaternion.from_float_array(imu[0, -4:])  # IMU的姿态

# 第一步：寻找姿态偏移量q_offset
q_offset = find_offset(ref_q, imu_q)
print("q_offset:", q_offset)

for i in range(1, len(ref)):
    ref_q = quaternion.from_float_array(ref[i, -4:])  # 外部设备的姿态
    imu_q = quaternion.from_float_array(imu[i, -4:])  # IMU的姿态

    # 第二步：使用找到的姿态偏移量对新的IMU姿态进行校准
    calibrated_q_imu = calibrate_imu(ref_q, imu_q, q_offset)

    imu[i, -4:] = quaternion.as_float_array(calibrated_q_imu)

plt.figure(figsize=(10, 8))

for idx in range(1, 5):
    plt.subplot(4, 1, idx)
    plt.plot(imu[:, 0], imu[:, idx], label=f'Calibrated q{idx-1}')
    plt.plot(raw[:, 0], raw[:, idx], label=f'Raw q{idx-1}')
    plt.legend()

plt.show()
    
