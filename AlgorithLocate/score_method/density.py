import math

import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_density_score(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    points = data[:,:2]
    number = points.shape[0]
    K = min(int (math.sqrt(number * 2)), number - 1)
    # print(K)

    # K+1 即包括点自身
    k_method = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree').fit(points)
    dis, _ = k_method.kneighbors(points)
    # print(dis)
    dis_all_points = np.sum(dis, axis=1)
    score = 1 - dis_all_points/(np.sum(dis_all_points)+1e-6)

    mark = data[:, 2]
    mark_alpha = 0.9
    weight = np.power(mark_alpha, mark)
    return score * weight


if __name__ == '__main__':
    z = [
        [-2.18036,0.657144,0,35,2,18,27,],
        [10.274,43.5806,0,35,24,18,28,],
        [-1.92936,-0.795384,0,35,24,18,28,],
        [0.23561,-0.520488,0,2,24,27,28,],
    ]

    print(get_density_score(data=z))