import math

import numpy as np

def get_out_virtual_points(x1,y1,r1,x2,y2,r2,d):
    x_direction, y_direction = (x1 - x2) / d, (y1 - y2) / d
    delta = d - r1 - r2
    dis = d - r1 - delta / 2
    x = x2 + x_direction * dis
    y = y2 + y_direction * dis
    return x, y


def get_in_virtual_points(x1,y1,r1,x2,y2,r2,d):
    if r2 > r1:
        x_direction, y_direction = (x1 - x2) / d, (y1 - y2) / d
        delta = r2 - d -  r1
        dis = r2 - delta/2
        x = x2 + x_direction * dis
        y = y2 + y_direction * dis
    else:
        x_direction, y_direction = (x2 - x1) / d, (y2 - y1) / d
        delta = r1 - d - r2
        dis = r1 - delta / 2
        x = x1 + x_direction * dis
        y = y1 + y_direction * dis
    return x, y


def get_cc_points(c1,c2,vir=True):
    """

    :param vir:
    :param c1:
    :param c2:
    :return: [[x, y, mark=0, r1, r2, time],[...]]
    """
    x1,y1,r1,_,time1,_ = c1
    x2,y2,r2,_,time2,_ = c2

    result = []

    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # print(f"dis: {d}")
    assert d > 0, "基站重合"

    if d > r1 + r2:
        if vir:
            vir_x, vir_y = get_out_virtual_points(x1,y1,r1,x2,y2,r2,d)
            result.append([vir_x, vir_y, 0, r1, r2, time1, time2])
    elif d < abs(r1 - r2):
        if vir:
            vir_x, vir_y = get_in_virtual_points(x1,y1,r1,x2,y2,r2,d)
            result.append([vir_x, vir_y, 0, r1, r2, time1, time2])
    else:
        # pass
        a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r1 ** 2 - a ** 2)
        x0 = x1 + a * (x2 - x1) / d
        y0 = y1 + a * (y2 - y1) / d
        x3 = x0 + h * (y2 - y1) / d
        y3 = y0 - h * (x2 - x1) / d
        x4 = x0 - h * (y2 - y1) / d
        y4 = y0 + h * (x2 - x1) / d
        if x3 == x4 and y3 == y4:
            result.append([x3, y3, 0, r1, r2, time1, time2])
        else:
            result.append([x3, y3, 0, r1, r2, time1, time2])
            result.append([x4, y4, 0, r1, r2, time1, time2])
    return result


def get_cl_points(c1, l1):
    """
    :param c1:
    :param l1:
    :return: [[x, y, mark=1, r, angle, time],[...]]
    """
    result = []
    x1,y1,r,_,time1,_ = c1
    x2,y2,_,angle,time2,measure_angle = l1


    k = math.tan(angle)
    # print(k)


    c = y2 - k * x2
    a = 1 + k ** 2
    b = 2 * (k * (c - y1) - x1)
    c_c = x1 ** 2 + (c - y1) ** 2 - r ** 2


    discriminant = b ** 2 - 4 * a * c_c

    if discriminant < 0:
        pass
    elif discriminant == 0:
        x3 = -b / (2 * a)
        y3 = k * x3 + (y2 - k * x2)
        dx3, dy3 = x3-x2,y3-y2
        dot_kv3 = math.cos(angle)*dx3 + math.sin(angle)*dy3
        # print(f"dot_kv3: {dot_kv3}")
        if dot_kv3 > 0:
            result.append([x3, y3, 1, r, angle, time1, time2])
    else:
        x3 = (-b + math.sqrt(discriminant)) / (2 * a)
        y3 = k * x3 + (y2 - k * x2)
        x4 = (-b - math.sqrt(discriminant)) / (2 * a)
        y4 = k * x4 + (y2 - k * x2)

        # 删掉基站后面的点
        dx3, dy3 = x3-x2,y3-y2
        dx4, dy4 = x4-x2,y4-y2
        dot_kv3 = math.cos(angle)*dx3 + math.sin(angle)*dy3
        dot_kv4 = math.cos(angle)*dx4 + math.sin(angle)*dy4
        if dot_kv3>0:
            result.append([x3, y3, 1, r, measure_angle, time1, time2])
        if dot_kv4>0:
            result.append([x4, y4, 1, r, measure_angle, time1, time2])

    return result


def get_ll_points(l1,l2):
    """
    :param l1:
    :param l2:
    :return: [[x, y, mark=2, angle1, angle2, time],[...]]
    """
    result = []
    x1, y1, _, angle1, time1, measure_angle1 = l1
    x2, y2, _, angle2, time2, measure_angle2 = l2
    k1 = math.tan(angle1)
    k2 = math.tan(angle2)
    x = ((y2 - y1) + (k1 * x1 - k2 * x2)) / (k1 - k2)
    y = k1 * x + (y1 - k1 * x1)

    dx1, dy1 = x - x1, y - y1
    dx2, dy2 = x - x2, y - y2
    dot_kv1 = math.cos(angle1) * dx1 + math.sin(angle1) * dy1
    dot_kv2 = math.cos(angle2) * dx2 + math.sin(angle2) * dy2
    if dot_kv1 >0 and dot_kv2 > 0:
        result.append([x, y, 2, measure_angle1, measure_angle2, time1, time2])
    return result


def get_xx_points(p1, p2):
    if p1[2] >= 0 and p2[2] >= 0:
        return get_cc_points(p1, p2)
    elif p1[3] >= 0 and p2[3] >= 0:
        return get_ll_points(p1, p2)
    elif p1[2] >= 0 and p2[3] >= 0:
        return get_cl_points(p1, p2)
    elif p1[3] >= 0 and p2[2] >= 0:
        return get_cl_points(p2, p1)
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
        for j in range(i + 1, data_number):
            points_ij = get_xx_points(data[i], data[j])
            if len(points_ij) > 0:
                points.extend(points_ij)
    return np.array(points)


if __name__ == '__main__':
    c1 = [0, 1, -1, 4.71, 0]
    c2 = [1, 0, -1, 3.14, 0]
    c3 = [0, 0, 1, -1, 0]
    error = [[25.6, 23.2, 14, -1, 7.8, -1],
             [-2.4, 23.2, 16, -1, 9.0, -1],
             [0, 0, 26, -1, 23.7, -1],
             [-2.4, 23.2, -1, 6.2482787221397, 4.1, 0.03490658503988659]]

    test = [[25.6, 23.2, 20, -1, 16, -1, ],
            [-2.4, 23.2, -1, 6.0912, 15, 0.191986, ], ]

    res = get_points(test)
    print(np.round(res[:, :], 8))