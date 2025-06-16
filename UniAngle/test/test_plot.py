import numpy as np
import matplotlib.pyplot as plt


#
# data = np.arange(-90, 61, 1)
# val = np.cos(np.deg2rad(data))
# print(data, val)
#
# plt.plot(np.deg2rad(data), val)
#
# def s(deg):
#     print(np.cos(np.deg2rad(deg)))
#     plt.scatter(np.deg2rad(deg), np.cos(np.deg2rad(deg)))
#     plt.scatter(np.deg2rad(-deg), np.cos(np.deg2rad(-deg)))
#
# s(21.17)
# s(48.79)
# s(60)
#
# plt.show()

def pdoa2aoa(pdoa, deno):
    # -196.1+360 65.58223843017042
    # -174.6    -75.93013225242788

    phi = np.deg2rad(pdoa)

    lamda = 0.07
    d = lamda / 2

    param = (phi * lamda) / (2 * np.deg2rad(deno) * d)

    theta = np.rad2deg(np.arcsin(param))

    return theta

def show_diff():
    pdoa = np.arange(-180, 181, 1)
    aoa_180 = pdoa2aoa(pdoa, 180)
    aoa_220 = pdoa2aoa(pdoa, 220)

    diff = np.abs(aoa_180 - aoa_220)

    fig, ax1 = plt.subplots()

    ax1.plot(pdoa, aoa_180, label='AOA 180', color='blue')
    ax1.plot(pdoa, aoa_220, label='AOA 220', color='red')
    ax1.set_xlabel('PDOA')
    ax1.set_ylabel('AOA (degrees)')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(pdoa, diff, label='|AOA_180 - AOA_220|', color='green', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.grid(True)
    plt.show()

if __name__=="__main__":
    # aoas = [8, -30, -60, -90, -115, -140, -157, -165]
    # for aoa in aoas:
    #     print(pdoa2aoa(aoa, 180))
    show_diff()