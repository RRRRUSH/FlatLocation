#ifndef DXBLOCATION_H
#define DXBLOCATION_H

#include <stdint.h>

#define RANGE_TIMEOUT   60              /*测距超时时间,单位秒*/
#define WAIT_MOTION_TIMEOUT 30          /*等待运动静止的超时时间*/
#define PI              (3.1415926)
#define GO_STEP         30               /*从老位置到新位置需要几秒*/
#define RECORD_RANGE_CNT    10          /*定位器能记录几个信标的测距结果*/
#define STEP_METER      (1.0f)          /*单步运动0.5米*/

class CPoint
{
public:
    float x;
    float y;
    CPoint &operator =(CPoint &p)
    {
        this->x = p.x;
        this->y = p.y;
        return *this;
    }
};

typedef struct
{
    uint8_t    id[3];
    float   x;          /*当前位置*/
    float   y;          /*当前位置*/
    float   expect_x;   /*期望的位置*/
    float   expect_y;   /*期望的位置*/
    uint8_t go_progess; /*行进进度*/
}device_location_t;

/* NEW */
typedef struct node
{
    double info[7];
    node* next = nullptr;
}node;
/* NEW */

#endif // DXBLOCATION_H
