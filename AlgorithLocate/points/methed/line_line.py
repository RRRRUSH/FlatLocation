import math


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

