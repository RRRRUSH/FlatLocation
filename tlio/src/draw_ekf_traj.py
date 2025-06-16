import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def draw_ekf_3dpos(f):
    pos = f[:, 12:15]
    fig = plt.figure("position-3d")
    ax = fig.add_subplot(111, projection="3d")
    plt.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="filter3d")
    return fig


def draw_ekf_2dpos(f, sub_path):
    pos = f[:, 12:15]
    fig, ax = plt.subplots()  # Create a new figure and axis each time
    ax.plot(pos[:, 0], pos[:, 1], label="filter2d")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()
    ax.axis("equal")
    ax.grid()
    ax.title.set_text(os.path.basename(sub_path))
    return fig

def save_photo(path, show, pos3d=False):
    subdir = [item for item in glob.glob(os.path.join(path, "*")) if os.path.isdir(item)]

    for i, sub_path in enumerate(subdir):
        print(f"Plotting {i+1}/{len(subdir)}: ", sub_path)
        f = np.load(os.path.join(sub_path, "not_vio_state.txt.npy"))
        if pos3d:
            fig = draw_ekf_3dpos(f)
        else:
            fig = draw_ekf_2dpos(f, sub_path)

        if show:
            plt.show()
        fig.savefig(os.path.join(sub_path, "position2d.png"))
        plt.close(fig)  # Close the figure to free up memory
def plt_gt():
    gt = np.load(r"E:\HaozhanLi\Project\FlatLoc\tlio\data\test\raw\137102747096458\imu0_resampled.npy")

    plt.plot(gt[:, 11], gt[:, 12])
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--state_path", type=str,
                        default=r"E:\HaozhanLi\Project\FlatLoc\tlio\data\test\o2")
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    save_photo(args.state_path, args.show)
    # plt_gt()