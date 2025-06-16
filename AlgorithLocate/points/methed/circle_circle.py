import math


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


if __name__ == '__main__':
    x1, y1, r1 = 0,0,1
    x2, y2, r2 = 0,10,5
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(get_out_virtual_points(x1,y1,r1,x2,y2,r2,d))
