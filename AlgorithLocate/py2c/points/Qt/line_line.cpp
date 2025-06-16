#include "dxblocation.h"
#include <tuple>
#include <math.h>

/*
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
*/

node* get_ll_points(double* l1, double* l2){
    node* result = new node;
    double x1,y1,angle1,time1,measure_angle1,x2,y2,angle2,time2,measure_angle2;
    double k1,k2,x,y;
    double dx1,dy1,dot_kv1,dx2,dy2,dot_kv2;

    std::tie(x1, y1, angle1, time1, measure_angle1) = std::make_tuple(l1[0], l1[1], l1[3], l1[4], l1[5]);
    std::tie(x2, y2, angle2, time2, measure_angle2) = std::make_tuple(l2[0], l2[1], l2[3], l2[4], l2[5]);
    k1 = tan(angle1);
    k2 = tan(angle2);
    x = ((y2 - y1) + (k1 * x1 - k2 * x2)) / (k1 - k2);
    y = k1 * x + (y1 - k1 * x1);

    dx1 = x - x1;
    dy1 = y - y1;
    dx2 = x - x2;
    dy2 = y - y2;
    dot_kv1 = cos(angle1) * dx1 + sin(angle1) * dy1;
    dot_kv2 = cos(angle2) * dx2 + sin(angle2) * dy2;
    if (dot_kv1 > 0 && dot_kv2 > 0){
        double tmp0[] = {x, y, 2, measure_angle1, measure_angle2, time1, time2};
        std::copy(tmp0, tmp0 + 7, result->info);
    }
    return result;
}