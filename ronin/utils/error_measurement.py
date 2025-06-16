import os

import matplotlib.pyplot as plt

import numpy as np

Absolute_trajectory = [[0, 0], [0, 41], [-17, 41], [-17, 3], [-4, 3], [-4, 0], [0, 0]]


def CZ(pose, num_points=4000):
    x = [point[0] for point in pose]
    y = [point[1] for point in pose]
    t = np.linspace(0, len(pose) - 1, num_points)
    x_new = np.interp(t, np.arange(len(x)), x)
    y_new = np.interp(t, np.arange(len(y)), y)
    new_pose = list(zip(x_new, y_new))
    return np.array(new_pose) * 0.8


def XZ(points, theta):
    angle = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    points_transposed = points.T
    rotated_points_transposed = np.dot(rotation_matrix, points_transposed)
    return rotated_points_transposed.T


def get_ate(pose1, pose2, index1=None, index2=None):
    assert pose1.shape == pose2.shape
    if index1 is None:
        ate = np.linalg.norm(pose1 - pose2, axis=1).mean()
        return ate
    else:
        error = np.linalg.norm(pose1[index1] - pose2[index2])
        return error


# traj = np.loadtxt("/home/a/Desktop/git/plot_trj/pos/011/trajectory.txt")
traj = np.load("/home/a/Desktop/git/TLIO/outputs/llio256/output_ours/0001/not_vio_state.txt.npy")[:,12:14]

interpolated_trajectory = CZ(Absolute_trajectory, traj.shape[0])

rotated_points = XZ(traj * 1.12, -81)           # TLIO
# rotated_points = XZ(traj * 1.12, 0)  # RONIN
error = {"error1": np.inf, "error2": np.inf, "error3": np.inf, "error4": np.inf, "error5": np.inf, "error6": np.inf}
for index in range(0, traj.shape[0]):
    error["error1"] = min(error["error1"], get_ate(interpolated_trajectory, rotated_points, index1=int((traj.shape[0]) * 0.167), index2=index))
    error["error2"] = min(error["error2"], get_ate(interpolated_trajectory, rotated_points, index1=int((traj.shape[0]) * 0.334), index2=index))
    error["error3"] = min(error["error3"], get_ate(interpolated_trajectory, rotated_points, index1=int((traj.shape[0]) * 0.5), index2=index))
    error["error4"] = min(error["error4"], get_ate(interpolated_trajectory, rotated_points, index1=int((traj.shape[0]) * 0.667), index2=index))
    error["error5"] = min(error["error5"], get_ate(interpolated_trajectory, rotated_points, index1=int((traj.shape[0]) * 0.834), index2=index))
    error["error6"] = min(error["error5"], get_ate(interpolated_trajectory, rotated_points, index1=-1, index2=index))
print(error)

plt.figure()
plt.plot(interpolated_trajectory[:, 0], interpolated_trajectory[:, 1], c='black')
plt.plot(rotated_points[:, 0], rotated_points[:, 1], c='red')


plt.axis('equal')
plt.grid(True)
plt.title("LLIO")
# plt.title("RONIN")
plt.show()
