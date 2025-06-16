import math
from math import pi

import numpy as np
from sklearn.neighbors import NearestNeighbors

# 权重占比 半径与角度、密集度、历史位置、时间
score_weight = [0.2, 1, 0.1, 0.4]

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


def get_history_score(data, history_location=None):

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if history_location is None:
        return np.zeros(data.shape[0],)
    else:
        history_location = history_location[:2]
        if not isinstance(history_location, np.ndarray):
            history_location = np.array(history_location)

    points = data[:,:2]

    dis = points - history_location
    dis = np.linalg.norm(dis,axis=1)
    # print(f"dis:{dis}")
    score = 1 - dis / (np.max(dis)+1e-6)
    score[score > 0.75] = 0.75

    return score


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


def get_score(data, history_location):

    ra_score = get_parents_score(data)
    density_score = get_density_score(data)
    history_score = get_history_score(data, history_location=history_location)
    time_score = get_time_score(data)
    print(f"ra_score: {ra_score}")
    print(f"density_score: {density_score}")
    print(f"history_score: {history_score}")
    print(f"time_score: {time_score}")
    score = ra_score * score_weight[0] + density_score * score_weight[1] + history_score * score_weight[2] + time_score * score_weight[3]
    return score / sum(score_weight)
