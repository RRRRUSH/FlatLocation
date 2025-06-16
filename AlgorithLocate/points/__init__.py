import math

import numpy as np
from score_method import score

# from points.methed.circle_circle import get_cc_points
# from points.methed.circle_line import get_cl_points
# from points.methed.line_line import get_ll_points

from methed.circle_circle import get_cc_points
from methed.circle_line import get_cl_points
from methed.line_line import get_ll_points



def get_xx_points(p1,p2):
    if p1[2] >= 0 and p2[2] >=0:
        return get_cc_points(p1, p2)
    elif p1[3] >= 0 and p2[3] >=0:
        return get_ll_points(p1, p2)
    elif p1[2] >= 0 and p2[3] >=0:
        return get_cl_points(p1, p2)
    elif p1[3] >= 0 and p2[2] >=0:
        return get_cl_points(p2,p1)
    else:
        raise "距离或者角度存在数据异常"



def get_points(data):
    """
    :param data: [[x, y, dis, ture_angle, time, measure_angle],...]
    :return:data: [[x, y, mark, r_a1, r_a2, time1, time2]]
    """
    data_number = len(data)
    points = []
    if data_number <= 1:
        return points
    for i in range(data_number):
        for j in range(i+1, data_number):
            points_ij = get_xx_points(data[i], data[j])
            if len(points_ij) > 0:
                points.extend(points_ij)
    return np.array(points)

if __name__ == '__main__':


    c1 = [0,1,-1,4.71,0]
    c2 = [1,0,-1,3.14,0]
    c3 = [0,0,1,-1,0]
    error = [[-2.4,23.2,25,-1,13,-1,],
[-2.4,23.2,-1,6.19592,50,0.0872665,],
[0,1.6,2,-1,41,-1,],
[0,1.6,-1,1.22173,9,0.349066,],
[25.6,23.2,19,-1,19,-1,],
[29.6,14.4,35,-1,30,-1,],]

    test = [[25.6,23.2,20,-1,16,-1,],
[-2.4,23.2,-1,6.0912,15,0.191986,],]

    res = get_points(error)
    score.get_score(res, [0, 0])
    print(np.round(res[:,0:2],3))