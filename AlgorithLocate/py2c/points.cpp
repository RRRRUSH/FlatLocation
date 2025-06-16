#include <tuple>
#include <math.h>
#include "loc.h"

void get_out_virtual_points(double x1, double y1,double r1,double x2,double y2,double r2,double d, double& x, double& y) {
    double x_direction, y_direction, delta, dis;
    x_direction = (x1 - x2) / d;
    y_direction = (y1 - y2) / d;
    delta = d - r1 - r2;
    dis = d - r1 - delta / 2;
    x = x2 + x_direction * dis;
    y = y2 + y_direction * dis;
}

void get_in_virtual_points(double x1,double y1,double r1,double x2,double y2,double r2,double d, double& x, double& y) {
    double x_direction, y_direction, delta, dis;
    if(r2 > r1){
        x_direction = (x1 - x2) / d;
        y_direction = (y1 - y2) / d;
        delta = r2 - d - r1;
        dis = r2 - delta/2;
        x = x2 + x_direction * dis;
        y = y2 + y_direction * dis;
    }
    else{
        x_direction = (x2 - x1) / d;
        y_direction = (y2 - y1) / d;
        delta = r1 - d - r2;
        dis = r1 - delta/2;
        x = x1 + x_direction * dis;
        y = y1 + y_direction * dis;
    }
}

node* get_cc_points(double* c1, double* c2, bool vir) {
    node* result = new node;
    double x1,y1,r1,time1,x2,y2,r2,time2;
    double a,h,x0,y0,x3,y3,x4,y4;
    double vir_x,vir_y;

    std::tie(x1, y1, r1, time1) = std::make_tuple(c1[0], c1[1], c1[2], c1[4]);
    std::tie(x2, y2, r2, time2) = std::make_tuple(c2[0], c2[1], c2[2], c2[4]);
    double d = sqrt(pow((x2 - x1),2) + pow((y2 - y1),2));

    if(d <= 0){
        // qDebug()<<"基站重合"<<std::endl;
        return nullptr;
    }
    else{
        if(d > r1 + r2){
            if(vir){
                get_out_virtual_points(x1,y1,r1,x2,y2,r2,d,vir_x,vir_y);
                double tmp0[] = {vir_x, vir_y, 0, r1, r2, time1, time2};
                std::copy(tmp0, tmp0 + 7, result->info);
            } else {
                return nullptr;
            }
        }
        else if (d < abs(r1 - r2)){
            if(vir){
                get_in_virtual_points(x1,y1,r1,x2,y2,r2,d,vir_x,vir_y);
                double tmp0[] = {vir_x, vir_y, 0, r1, r2, time1, time2};
                std::copy(tmp0, tmp0 + 7, result->info);
            } else {
                return nullptr;
            }
        }
        else{
            a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
            h = sqrt(r1 * r1 - a * a);
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
    }
    return result;
}

node* get_cl_points(double* c1, double* l1) {
    node* result = new node;
    double x1,y1,r,time1,x2,y2,angle,time2,measure_angle;
    double a,b,c,c_c,discriminant,x3,y3,x4,y4;
    double dx3,dy3,dot_kv3,dx4,dy4,dot_kv4;

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

        if (dot_kv3 > 0 && dot_kv4 > 0){
            double tmp0[] = {x3, y3, 1, r, measure_angle, time1, time2};
            double tmp1[] = {x4, y4, 1, r, measure_angle, time1, time2};
            std::copy(tmp0, tmp0 + 7, result->info);
            node* current = new node;
            result->next = current;
            std::copy(tmp1, tmp1 + 7, result->next->info);
        }

        else if (dot_kv3 > 0 && dot_kv4 <= 0){
            double tmp0[] = {x3, y3, 1, r, measure_angle, time1, time2};
            std::copy(tmp0, tmp0 + 7, result->info);
        }
        else if (dot_kv3 <= 0 && dot_kv4 > 0){
            double tmp1[] = {x4, y4, 1, r, measure_angle, time1, time2};
            std::copy(tmp1, tmp1 + 7, result->info);
        }
        else return nullptr;
    }
    return result;
}


node* get_ll_points(double* l1, double* l2) {
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

node* get_xx_points(double* p1, double* p2) {
    if (p1[2] >= 0 && p2[2] >=0)
        return get_cc_points(p1, p2,true);
    else if (p1[3] >= 0 && p2[3] >=0)
        return get_ll_points(p1, p2);
    else if (p1[2] >= 0 && p2[3] >=0)
        return get_cl_points(p1, p2);
    else if (p1[3] >= 0 && p2[2] >=0)
        return get_cl_points(p2, p1);
    else
        return nullptr;
}

int get_node_count(node* head) {
    int count = 0;
    node* current = head;
    while (current != nullptr) {
        count++;
        current = current->next;
    }
    return count;
}

node* get_points(double** data, int data_number) {
    //int data_number = sizeof(data) / sizeof(data[0]),是行数

    node* points = nullptr;
    node* tail = nullptr;
    node* points_ij = new node;
    if (data_number <= 1) return nullptr;
    for (int i = 0; i < data_number; i++){
        for (int j = i+1; j < data_number; j++){
            points_ij = get_xx_points(data[i], data[j]);
            if(get_node_count(points_ij) > 0){
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