from math import pi

import numpy as np


def get_parents_score(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    number = data.shape[0]
    data = data[:,2:5]
    data_rr = data[data[:, 0] == 0]
    data_ra = data[data[:, 0] == 1]
    rr_l = data_rr.shape[0]
    ra_l = data_ra.shape[0]
    if rr_l>0 and ra_l>0:
        max_r = max(np.max(data_rr[:,1:3]), max(data_ra[:,1]))
    elif rr_l>0:
        max_r = np.max(data_rr[:,1:3])
    elif ra_l>0:
        max_r = max(data_ra[:,1])
    else:
        max_r = 0
    max_a = pi/3
    # if aa_l>0 and ra_l>0:
    #     max_a = max(np.max(data_aa[:,1:3]), max(data_ra[:,2]))
    # elif aa_l>0:
    #     max_a = np.max(data_aa[:,1:3])
    # elif ra_l>0:
    #     max_a = max(data_ra[:,2])
    # else:
    #     max_a = 0


    score = np.zeros(number)
    for i in range(number):
        if data[i, 0] == 0:
            score[i] = (1 - (data[i,1]/(2 * max_r))) * 0.5 + (1 - (data[i,2]/(2 * max_r))) * 0.5
        elif data[i, 0] == 1:
            score[i] = (1 - (data[i,1]/(2 * max_r))) * 0.5 + (1 - (data[i,2]/max_a)) * 0.5
        elif data[i, 0] == 2:
            score[i] = (1 - (data[i,1]/max_a)) * 0.5 + (1 - (data[i,2]/max_a)) * 0.5

    return score


if __name__ == '__main__':

    z = [[0, 0, 0, 2, 4, 5, 1],
        [0, -1, 1, 1, 1.57, 1, 10],
        [-1, 0, 0, 1, 5, 10, 5]]

    z1 = [[0, 0, 2, 4.71, 3.14, 5, 1],
       [0, -1, 1, 1, 4.71, 1, 10],
        [-1, 0, 1, 1, 3.14, 10, 5]]
    s = get_parents_score(z1)
    print(s, type(s), len(s))