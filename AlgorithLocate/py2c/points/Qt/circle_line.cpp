#include "dxblocation.h"
#include <tuple>
#include <math.h>

/*
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
*/

node* get_cl_points(double* c1, double* l1){
    node* result = new node;
    double x1,y1,r,time1,x2,y2,angle,time2,measure_angle;
    double a,b,c,c_c,discriminant,x3,y3,x4,y4;
    double dx3,dy3,dot_kv3,dx4,dy4,dot_kv4;
    int flag = 0;

    std::tie(x1, y1, r, time1) = std::make_tuple(c1[0], c1[1], c1[2], c1[4]);
    std::tie(x2, y2, angle, time2, measure_angle) = std::make_tuple(l1[0], l1[1], l1[3], l1[4], l1[5]);
    double k = tan(angle);

    c = y2 - k * x2;
    a = 1 + k * k;
    b = 2 * (k * (c - y1) - x1);
    c_c = x1 * x1 + pow((c - y1),2) - r * r;

    discriminant = b * b - 4 * a * c_c;

    if (discriminant < 0) return 0;
    else if (discriminant == 0){
        x3 = -b / (2 * a);
        y3 = k * x3 + (y2 - k * x2);
        dx3 = x3 - x2;
        dy3 = y3 - y2;
        dot_kv3 = cos(angle) * dx3 + sin(angle) * dy3;

        if (dot_kv3 > 0){
            double tmp0[] = {x3, y3, 1, r, angle, time1, time2};
            std::copy(tmp0, tmp0 + 7, result->info);
        }
    }
    else{
        x3 = (-b + sqrt(discriminant)) / (2 * a);
        y3 = k * x3 + (y2 - k * x2);
        x4 = (-b - sqrt(discriminant)) / (2 * a);
        y4 = k * x4 + (y2 - k * x2);

        //删掉基站后面的点
        dx3 = x3-x2;
        dy3 = y3-y2;
        dx4 = x4-x2;
        dy4 = y4-y2;
        dot_kv3 = cos(angle) * dx3 + sin(angle) * dy3;
        dot_kv4 = cos(angle) * dx4 + sin(angle) * dy4;
        //存疑
        if (dot_kv3 > 0){
            double tmp0[] = {x3, y3, 1, r, measure_angle, time1, time2};
            std::copy(tmp0, tmp0 + 7, result->info);
            flag++;
        }

        if (dot_kv4 > 0){
            double tmp1[] = {x4, y4, 1, r, measure_angle, time1, time2};
            if(flag == 0)
                std::copy(tmp1, tmp1 + 7, result->info);
            else{//需要加个node
                node* current = new node;
                result->next = current;
                std::copy(tmp1, tmp1 + 7, result->next->info);
            }
        }
    }
    return result;
}