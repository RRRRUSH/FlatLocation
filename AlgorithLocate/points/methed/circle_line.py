import math

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
