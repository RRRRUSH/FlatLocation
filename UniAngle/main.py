import os

import numpy as np
import pandas as pd

from utils.plotter import Plotter
from utils.multi_angle_fusion import AngleFusion

CONFIG = dict()

np.set_printoptions(linewidth=300)

def config_init():
    config = {
        # "ANTENNAS_NUM": 6,
        # "ANTENNAS_SERIAL": "A,B,C,D,E,F",
        # "ROOT_DIR": "./data/0424/20.txt", # 天线面左侧为负，id逆时针递减
        # # "ROOT_DIR": "./data/0507/30_test_0507.txt", # 天线面左侧为负，id逆时针递增
        # # "ROOT_DIR": "./data/generate/120.txt",  # 天线面左侧为负，id逆时针递增

        "ANTENNAS_NUM": 4,
        "ROOT_DIR": "./data/0506_new/merge/310.txt",  # 天线面左侧为负，id逆时针递增
        "REAL_ANGLE": 310,
        "OUTPUT_DIR": "./output/0507",
    }

    CONFIG.update(config)
    return config

def save_result(output_path, angle, score, real_angle):
    print(output_path)
    std_window = 5000

    savedf = np.zeros((len(angle.keys()), 12))
    for i, ts in enumerate(angle.keys()):
        savedf[i, 0] = ts
        savedf[i, 1:8] = score[ts][0, :]
        savedf[i, 8] = angle[ts][0, -1]
        savedf[i, 9] = real_angle

        delta_angle = np.abs(angle[ts][0, -1] - real_angle) % 360
        savedf[i, 10] = np.minimum(delta_angle, 360 - delta_angle)

        mask = (savedf[:, 0] >= np.max(ts - std_window, 0)) & (savedf[:, 0] <= ts)
        savedf[i, 11] = np.std(savedf[mask][:, 10])

    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(savedf).to_csv(
        os.path.join(output_path, f'{real_angle}.csv'), index=False, header=[
            "TIMESTAMP", "SCOPE", "TIME", "STD", "DENSITY", "HISTORY",
            "RSSI", "SCORE", "ANGLE", "REAL ANGLE", "ERROR", "STD"]
    )

def main():
    config_init()

    af = AngleFusion(
        data_dir=CONFIG.get("ROOT_DIR"),
        antenna_nums=CONFIG.get("ANTENNAS_NUM"),
        score_weights=[0, 0, 1, 0, 0, 0],
        calibration=False
    )

    angle_res, score_res = af.get_result()
    plotter = Plotter(
        angle_result=angle_res,
        score_result=score_res,
        antenna_nums=CONFIG.get("ANTENNAS_NUM"),
    )
    plotter.show()

    # save_result(
    #     CONFIG.get("OUTPUT_DIR"),
    #     angle_res, score_res,
    #     CONFIG.get("REAL_ANGLE")
    # )

if __name__ == '__main__':
    main()