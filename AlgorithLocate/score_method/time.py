import numpy as np


def get_time_score(data,delta_alpha=0.2):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data = data[:, 5:7]
    delta_time_score = 1 - np.square(data[:,0] - data[:,1])/100
    delta_time_score[delta_time_score < 0] = 0
    # print(delta_time_score)

    result = np.copy(data)
    result[data < 3] = 1
    result[data >= 60] = -1
    mask = (data >= 3) & (data < 60)
    result[mask] = -(result[mask] / 30) + 1
    score = np.sum(result,axis=1)/2

    score2 = score * (1-delta_alpha) + delta_time_score * delta_alpha
    # print("multy:",score2)
    return score2


if __name__ == '__main__':

    z = [
        [0, 0, 2, 4.71, 3.14, 5, 1],
        [0, -1, 1, 1, 4.71, 1, 10],
        [-1, 0, 1, 1, 3.14, 10, 5]
    ]

    y=get_time_score(data=z)
    print(y)