import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def rotate_quaternion(r1, r2, q_o):
    R_total = r1 @ r2

    q_rotation = R.from_matrix(R_total).as_quat()

    q_target = R.from_quat(q_rotation).inv() * R.from_quat(q_o) * R.from_quat(q_rotation)
    q_target = q_target.as_quat()

    return q_target

def task1():
    " 左手坐标系转右手坐标系 "

    # xyzw
    q_original = np.array([0.6803997, 0.0090357597, 0.05518873138, 0.730704322])

    R1 = R.from_euler('z', -90, degrees=True).as_matrix()
    R2 = R.from_euler('y', -90, degrees=True).as_matrix()

    q_target = rotate_quaternion(R1, R2, q_original)

    # w xyz
    q_right = q_target[-1], -q_target[0], q_target[1], -q_target[2]

    print("left", q_original[-1], q_original[0], q_original[1], q_original[2])
    print("right", q_right)

def task2():
    " rot "
    imu = pd.read_csv(
        r"E:\HaozhanLi\Project\FlatLoc\EqNIO\test\temp\txt_to_csv\1\imu_output.csv",
        sep=",").dropna()
    label = pd.read_csv(
        r"E:\HaozhanLi\Project\FlatLoc\EqNIO\test\temp\txt_to_csv\1\pos_input.csv",
        sep=",")

    q_i = imu[['qw', 'qx', 'qy', 'qz']]
    q_l = label[['qw', 'qx', 'qy', 'qz']]

    rot_m = np.array([0.998961, 0.00464107, 0.045336, 0.0452105, 0.0242917, -0.998682, -0.00573625, 0.999694, 0.0240567])
    rot_m = rot_m.reshape(3, 3)

    rot_q = R.from_matrix(rot_m).as_quat()
    q_rotted = R.from_quat(rot_q).inv() * R.from_quat(q_l[['qx', 'qy', 'qz', 'qw']]) * R.from_quat(rot_q)
    q_rotted = q_rotted.as_quat()

    print(q_l.shape)

    plt.subplot(4, 1, 1)
    plt.title('IMU Quaternion')
    plt.scatter(np.arange(q_i.shape[0]), q_i['qw'], s=1, label='w')
    plt.subplot(4, 1, 2)
    plt.scatter(np.arange(q_i.shape[0]), q_i['qx'], s=1, label='x')
    plt.subplot(4, 1, 3)
    plt.scatter(np.arange(q_i.shape[0]), q_i['qy'], s=1, label='y')
    plt.subplot(4, 1, 4)
    plt.scatter(np.arange(q_i.shape[0]), q_i['qz'], s=1, label='z')
    plt.legend()
    plt.show()

    plt.subplot(4, 1, 1)
    plt.title('Label Quaternion')
    plt.scatter(np.arange(q_l.shape[0]), q_l['qw'], s=1, label='w')
    plt.subplot(4, 1, 2)
    plt.scatter(np.arange(q_l.shape[0]), q_l['qx'], s=1, label='x')
    plt.subplot(4, 1, 3)
    plt.scatter(np.arange(q_l.shape[0]), q_l['qy'], s=1, label='y')
    plt.subplot(4, 1, 4)
    plt.scatter(np.arange(q_l.shape[0]), q_l['qz'], s=1, label='z')
    plt.legend()
    plt.show()

    plt.subplot(4, 1, 1)
    plt.title('Rotted Quaternion')
    plt.scatter(np.arange(q_rotted.shape[0]), q_rotted[:, 3], s=1, label='w')
    plt.subplot(4, 1, 2)
    plt.scatter(np.arange(q_rotted.shape[0]), q_rotted[:, 0], s=1, label='x')
    plt.subplot(4, 1, 3)
    plt.scatter(np.arange(q_rotted.shape[0]), q_rotted[:, 1], s=1, label='y')
    plt.subplot(4, 1, 4)
    plt.scatter(np.arange(q_rotted.shape[0]), q_rotted[:, 2], s=1, label='z')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    task2()