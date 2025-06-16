#include "dxblocation.h"
#include <tuple>
#include <math.h>
#include <iostream>

/*
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
        delta = r2 - d - r1
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
*/
void get_out_virtual_points(double x1,double y1,double r1,double x2,double y2,double r2,double d, double& x, double& y){
    double x_direction, y_direction, delta, dis;
    x_direction = (x1 - x2) / d;
    y_direction = (y1 - y2) / d;
    delta = d - r1 - r2;
    dis = d - r1 - delta / 2;
    x = x2 + x_direction * dis;
    y = y2 + y_direction * dis;
}


void get_in_virtual_points(double x1,double y1,double r1,double x2,double y2,double r2,double d, double& x, double& y){
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

/*
def get_cc_points(c1,c2,vir=True):
    """

    :param vir:
    :param c1:
    :param c2:
    :return: [[x, y, mark=0, r1, r2, time1, time2],[...]]
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
*/

node* get_cc_points(double* c1, double* c2, bool vir = true){
    node* result = new node;
    double x1,y1,r1,time1,x2,y2,r2,time2;
    double a,h,x0,y0,x3,y3,x4,y4;
    double vir_x,vir_y;

    std::tie(x1, y1, r1, time1) = std::make_tuple(c1[0], c1[1], c1[2], c1[4]);
    std::tie(x2, y2, r2, time2) = std::make_tuple(c2[0], c2[1], c2[2], c2[4]);
    double d = sqrt(pow((x2 - x1),2) + pow((y2 - y1),2));

    if(d <= 0){
        std::cout<<"基站重合"<<std::endl;
        return nullptr;
    }
    else{
        if(d > r1 + r2){
            if(vir){
                get_out_virtual_points(x1,y1,r1,x2,y2,r2,d,vir_x,vir_y);
                double tmp0[] = {vir_x, vir_y, 0, r1, r2, time1, time2};
                std::copy(tmp0, tmp0 + 7, result->info);
            }
        }
        else if (d < abs(r1 - r2)){
            if(vir){
                get_in_virtual_points(x1,y1,r1,x2,y2,r2,d,vir_x,vir_y);
                double tmp0[] = {vir_x, vir_y, 0, r1, r2, time1, time2};
                std::copy(tmp0, tmp0 + 7, result->info);
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

int main(){
    /*
    x1, y1, r1 = 0,0,1
    x2, y2, r2 = 0,10,5
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(get_out_virtual_points(x1,y1,r1,x2,y2,r2,d))
    */
   int x1, y1, r1, x2, y2, r2;
   double vir_x,vir_y;
   std::tie(x1, y1, r1) = std::make_tuple(0,0,1);
   std::tie(x2, y2, r2) = std::make_tuple(0,10,5);
   double d = sqrt(pow((x2 - x1),2) + pow((y2 - y1),2));
   get_out_virtual_points(x1,y1,r1,x2,y2,r2,d,vir_x,vir_y);
   std::cout << vir_x << " " << vir_y << std::endl;
   return 0;
}