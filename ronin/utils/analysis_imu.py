import math
import os.path
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import window


def vis(data, path, name):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(range(data.shape[0]), data[:, 0], label=f'x,mean:{np.mean(data[:,0]):.3f}')
    axs[0].set_title(f'X {name}')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(f'{name}')
    axs[0].legend()
    axs[1].plot(range(data.shape[0]), data[:, 1], label=f'y,mean:{np.mean(data[:,1]):.3f}')
    axs[1].set_title(f'Y {name}')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(f'{name}')
    axs[1].legend()
    axs[2].plot(range(data.shape[0]), data[:, 2], label=f'z,mean:{np.mean(data[:,2]):.3f}')
    axs[2].set_title(f'Z {name}')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel(f'{name}')
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}.png"))
    plt.show()
    plt.close()


def vis_feat(data, path, name):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(range(data.shape[0]), data[:, 0], label=f'x,mean:{np.mean(data[:,0]):.3f}')
    axs[0].set_title(f'X feat_{name}')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(f'{name}')
    axs[0].legend()
    axs[1].plot(range(data.shape[0]), data[:, 1], label=f'y,mean:{np.mean(data[:,1]):.3f}')
    axs[1].set_title(f'Y feat_{name}')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(f'{name}')
    axs[1].legend()
    axs[2].plot(range(data.shape[0]), data[:, 2], label=f'z,mean:{np.mean(data[:,2]):.3f}')
    axs[2].set_title(f'Z feat_{name}')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel(f'{name}')
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"feat_{name}.png"))
    plt.show()
    plt.close()


def vis_pressure(data, path, name):
    plt.figure(figsize=(32, 8))
    height = calculate_altitude(data)
    plt.plot(range(len(height)), height, label='height')
    plt.savefig(os.path.join(path, f"{name}.png"))
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def calculate_altitude(pressure):
    JZQY = 101325
    GD = []
    windows_size = 1
    step_size = 1
    for i in range(windows_size):
        GD.append(44330 * (1 - math.pow(pressure[i] / JZQY, 1.0 / 5.25)) * 1)
    for i in range(windows_size, len(pressure) - windows_size, step_size):
        pm = np.mean(pressure[i - windows_size:i])
        for j in range(step_size):
            p = pressure[i + j]
            h = 44330 * (1 - math.pow(p / pm, 1.0 / 5.25)) * 1 + GD[i - windows_size]
            GD.append(h)

    return GD


if __name__ == '__main__':
    path = "/home/a/Desktop/git/plot_trj/pos/1022"
    acce = np.loadtxt(os.path.join(path, 'acce.txt'))[:, [1, 2, 3]]
    gyro = np.loadtxt(os.path.join(path, 'gyro.txt'))[:, [1, 2, 3]]
    vis(gyro, path, "gyro")
    vis(acce, path, "acce")

    f = np.loadtxt(os.path.join(path, 'feat.txt'))
    feat_acce = f[:, [3, 4, 5]]
    feat_gyro = f[:, [0, 1, 2]]
    vis_feat(feat_gyro, path, "gyro")
    vis_feat(feat_acce, path, "acce")

    pressure = np.loadtxt(os.path.join(path, 'pressure.txt'))
    vis_pressure(pressure[:, 1], path, "pressure")
