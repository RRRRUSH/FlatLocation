#include "dxblocation.h"
#include <tuple>
#include <math.h>
#include <iostream>


int getNodeCount(node* head) {
    int count = 0;
    node* current = head;
    while (current != nullptr) {
        count++;
        current = current->next;
    }
    return count;
}


node* get_cc_points(double* c1, double* c2){
    node* result = new node;
    double x1,y1,r1,time1,x2,y2,r2,time2;
    double a,h,x0,y0,x3,y3,x4,y4;

    std::tie(x1, y1, r1, time1) = std::make_tuple(c1[0], c1[1], c1[2], c1[4]);
    std::tie(x2, y2, r2, time2) = std::make_tuple(c2[0], c2[1], c2[2], c2[4]);
    double d = sqrt(pow((x2 - x1),2) + pow((y2 - y1),2));

    if(d <= 0){
        std::cout<<"基站重合"<<std::endl;
        return 0;
    }
    else if(d < (r1 + r2) && d > abs(r1 - r2)){
        a = (pow(r1,2) - pow(r2,2) + pow(d,2)) / (2 * d);
        h = sqrt(pow(r1,2) - pow(a,2));
        x0 = x1 + a * (x2 - x1) / d;
        y0 = y1 + a * (y2 - y1) / d;
        x3 = x0 + h * (y2 - y1) / d;
        y3 = y0 - h * (x2 - x1) / d;
        x4 = x0 - h * (y2 - y1) / d;
        y4 = y0 + h * (x2 - x1) / d;

        if(x3 == x4 && y3 == y4){
            double tmp0[] = {x3, y3, 0, r1, r2, time1, time2};
            std::copy(tmp0, tmp0 + 7, result->info);
        }
        else{
            double tmp0[] = {x3, y3, 0, r1, r2, time1, time2};
            double tmp1[] = {x4, y4, 0, r1, r2, time1, time2};
            node* current = new node;
            result->next = current;
            std::copy(tmp0, tmp0 + 7, result->info);
            std::copy(tmp1, tmp1 + 7, result->next->info);
        }
    }
    return result;
}


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

node* get_xx_points(double* p1, double* p2){
    if (p1[2] >= 0 && p2[2] >= 0)
        return get_cc_points(p1, p2);
    else if (p1[3] >= 0 && p2[3] >= 0)
        return get_ll_points(p1, p2);
    else if (p1[2] >= 0 && p2[3] >= 0)
        return get_cl_points(p1, p2);
    else if (p1[3] >= 0 && p2[2] >= 0)
        return get_cl_points(p2, p1);
    else
        std::cout<<"距离或者角度存在数据异常"<<std::endl;
        return nullptr;
}

/*
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
*/
node* get_points(double** data, int data_number){//int data_number = sizeof(data) / sizeof(data[0]),是行数
    node* points = nullptr;
    node* tail = nullptr;
    node* points_ij = new node;
    if (data_number <= 1) return nullptr;
    for (int i = 0; i < data_number; i++){
        for (int j = i+1; j < data_number; j++){
            points_ij = get_xx_points(data[i], data[j]);
            if(getNodeCount(points_ij) > 0){
                if (points == nullptr) {
                    points = points_ij;
                    tail = points_ij;
                    while(tail->next != nullptr) tail = tail->next;
                } else {
                    tail->next = points_ij;
                    while(tail->next != nullptr) tail = tail->next;
                }
            }
        }
    }
    return points;
}

int main(){

    double p1[6] = {25.6, 23.2, 14, -1, 6.6, -1};
    double p2[6] = {-2.4, 23.2, 16, -1, 7.8, -1};
    int num_0 = 2;
    //num_1 = 0;
    // 动态分配多维数组
    double** p = new double*[num_0];
    for (int i = 0; i < num_0; i++) {
        p[i] = new double[6];
    }
    // 将一维数组的元素赋值给多维数组
    double* arrays_0[] = {p1,p2};
    for (int i = 0; i < num_0; i++) {
        for (int j = 0; j < 6; j++) {
            p[i][j] = arrays_0[i][j];
        }
    }
    //node* points = get_xx_points(p1,p2);
    node* points = get_points(p,num_0);

    while(points != nullptr){
        for(int i = 0; i < 7; i++)
            std::cout<< points->info[i] << " ";
        std::cout << std::endl;
        points = points->next;
    }
    
    double c1[] = {0, 1, -1, 4.71, 0, 4.71};
    double c2[] = {1, 0, -1, 3.14, 0, 3.14};
    double c3[] = {0, 0, 1, -1, 0, -1};
    int num_1 = 3;
    // 动态分配多维数组
    double** c = new double*[num_1];
    for (int i = 0; i < num_1; i++) {
        c[i] = new double[6];
    }
    // 将一维数组的元素赋值给多维数组
    double* arrays[] = {c1, c2, c3};
    for (int i = 0; i < num_1; i++) {
        for (int j = 0; j < 6; j++) {
            c[i][j] = arrays[i][j];
        }
    }

    node* res = get_points(c,num_1);
    while(res != nullptr){
        for(int i = 0; i < 7; i++)
            std::cout<< res->info[i] << " ";
        std::cout << std::endl;
        res = res->next;
    }
   return 0;
}
