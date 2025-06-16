#ifndef DXBLOCATION_H
#define DXBLOCATION_H

#include "taxdef.h"
#include "QTimer"
#include <QObject>
#include <QDateTime>

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

class range_pair_t
{
public:
    uint8_t     id1[3];
    uint8_t     id2[3];
    float       dis;
    uint32_t    time;
    bool        is_new_dis;

    range_pair_t &operator =(const range_pair_t &rp)
    {
        memcpy(this->id1, rp.id1, 3);
        memcpy(this->id2, rp.id2, 3);
        this->dis = rp.dis;
        this->time = rp.time;
        this->is_new_dis = rp.is_new_dis;
        return *this;
    }
};

class RangeInfo
{
public:
    uint8_t id[3];
    float   dis;
    QDateTime range_time;
    uint32_t    time_pass;
    bool        is_new_dis;
    uint16_t    angle;
    uint32_t    angle_time_pass;
    bool        is_new_angle;
    RangeInfo()
    {
        memset(id, 0, sizeof(id));
        dis = -1;
        range_time = QDateTime::currentDateTime();
        time_pass = 0;
        is_new_dis = true;
        angle_time_pass = 0;
        angle = 65535;
        is_new_angle = true;
    }

    RangeInfo &operator=(const RangeInfo &ri)
    {
        memcpy(this->id, ri.id, 3);
        this->dis = ri.dis;
        this->range_time = ri.range_time;
        this->time_pass = ri.time_pass;
        this->is_new_dis = ri.is_new_dis;
        this->angle = ri.angle;
        this->angle_time_pass = ri.angle_time_pass;
        return *this;
    }
};

class MotionState
{
public:
    uint8_t     motion_state_list[RANGE_TIMEOUT];   /*保存测距有效期内的运动静止状态，0：静止，1：运动，255：未知*/

    MotionState()
    {
        memset(motion_state_list,0xFF,sizeof(motion_state_list));
    }

    MotionState &operator =(const MotionState &state)
    {
        memcpy(this->motion_state_list, state.motion_state_list, sizeof(uint8_t)*RANGE_TIMEOUT);
        return *this;
    }
};

class BeaconDevice
{
public:
    uint8_t id[3];  /*信标编号*/
    float   x;      /*信标坐标*/
    float   y;      /*信标坐标*/
    uint16_t  heading;      /*信标朝向*/
    uint16_t  roll;         /*信标俯仰角*/
    uint16_t  yaw;          /*信标旋转角*/

    BeaconDevice()
    {
        memset(id, 0, sizeof(id));
        x = 0;
        y = 0;
        heading = 0;
        roll = 0;
        yaw = 0;
    }

    BeaconDevice &operator= (const BeaconDevice &beacon)
    {
        memcpy(this->id, beacon.id, 3);
        this->x = beacon.x;
        this->y = beacon.y;
        this->heading = beacon.heading;
        this->roll = beacon.roll;
        this->yaw = beacon.yaw;

        return *this;
    }
};

class DwqDevice
{
public:
    uint8_t     id[3];      /*定位器编号*/
    float       x;          /*当前位置*/
    float       y;          /*当前位置*/
    float       expect_x;   /*期望的位置*/
    float       expect_y;   /*期望的位置*/
    uint8_t     go_progess; /*行进进度*/
    RangeInfo   range_list[RECORD_RANGE_CNT];  /*测距列表*/
    int         range_cnt;
    MotionState motion;     /*运动状态*/
    int         loc_pass;   /*到上次使用距离值的时间*/
    bool        has_loc;    /*是否定到过位置*/
    DwqDevice()
    {
        memset(id, 0, sizeof(id));
        x = 0;
        y = 0;
        expect_x = 0;
        expect_y = 0;
        go_progess = 0;
        range_cnt = 0;
        loc_pass = 30;
        has_loc = false;
    }

    DwqDevice &operator= (const DwqDevice &dwq)
    {
        memcpy(this->id, dwq.id, 3);
        this->x = dwq.x;
        this->y = dwq.y;
        this->expect_x = dwq.expect_x;
        this->expect_y = dwq.expect_y;
        this->go_progess = dwq.go_progess;
        this->motion = dwq.motion;
        for(int i = 0; i < RECORD_RANGE_CNT; ++i)
            this->range_list[i] = dwq.range_list[i];
        this->range_cnt = dwq.range_cnt;
        this->loc_pass = dwq.loc_pass;
        this->has_loc = dwq.has_loc;
        return *this;
    }
};

class DXBLocation : public QObject
{
    Q_OBJECT
private:
    QList<DwqDevice>            dwq_list;               /*定位器位置列表*/
    QList<BeaconDevice>         beacon_list;            /*信标位置列表*/
    QList<uint32_t>             dwqWhitefilter;

    bool                        default_heading;
    bool                        enable_whitefilter;     /*使能白名单*/
    bool                        default_motion;
    bool                        delay_deal;             /* 延时处理测距结果 */
    quint64                     time_cnt;
public:
    explicit DXBLocation(QObject *parent = 0);
    ~DXBLocation();

    /*重置算法记录*/
    void reset();

    /*设置未收到运动静止报文时，定位器的默认运动静止状态*/
    void set_default_motion(bool motion);

    void set_default_heading(bool is_default_heading);

    /*设置是否延迟处理测距报文以等待运动静止状态回传*/
    void set_delay_deal_range(bool delay);

    /*使能白名单*/
    void set_whitefilter(bool enable);

    /*输入基站坐标信息*/
    void input_beacon(QList<BeaconDevice> &beaconlist);

    /*输入定位白名单，只有白名单中的定位器定位数据*/
    void input_dwq_filter(QList<uint32_t> &list);

    /*实时输入定位信标信息*/
    void intput(const uint8_t *srcid, const around_t *around_list, int around_cnt);
    void intputAngle(const uint8_t *srcid, const p5101_around_t *around_list, int around_cnt,int compass,int pitch,int rotation);
    /*输入定位器过去的运动状态，1BIT代表4秒时间的运动静止状态，总共是前32秒的运动静止状态*/
    void input_dwq_motions(const uint8_t *srcid, uint8_t motions);

    /*获得最新的定位器坐标*/
    QList<DwqDevice>        getNewDwqLocation(void);

    /*需定时1秒调用1次*/
    void timer_task(void);
private:

    bool isSystemDWXB(const uint8_t *id);

    /*判断是否是白名单中的定位器*/
    bool isDwqInWhiteFilter(const uint8_t *id);

    /*更新定位器当前位置和期望位置*/
    void updateDwqLocation(DwqDevice *dwq, CPoint &cp, CPoint &ep);
    void updateDwqLocation(DwqDevice *dwq, CPoint &ep);

    /*获取信标位置*/
    bool getBeaconLocation(const uint8_t * beacon_id, CPoint &p);

    /*获取定位器的上一个位置*/
    bool getDwqLastLocation(const DwqDevice *dwq, CPoint &p);

    /*可以依据定位信息进行定位了,T1:最新的测距时刻，T2:上一次测距时刻,注：T1<=T2*/
    bool could_cal_loation_by_motion(const DwqDevice *dwq, int t1, int t2);

    /*判断是否有新的测距数据*/
    bool has_new_range_info(const DwqDevice *dwq);

    /*计算定位器的新的坐标位置*/
    void cal_dwq_new_location(DwqDevice *dwq);

    /*对定位器的测距结果按照时间先后排序*/
    void sortDwqRangeInfo(const uint8_t *src_id);

    /*查找定位器和定位信标的测距对，没有则创建一个*/
    RangeInfo *getRangePair(const uint8_t *beacon_id, const uint8_t *dwq_id);

    /*查找该设备的历史定位记录，如果没有则创建一个*/
    DwqDevice *getDwqDevice(const uint8_t *id);

    /* 获得一个圆弧上随机的一个点坐标*/
    CPoint  getRandomPointOnCrcle(float x, float y, float radius);

    /*获得圆弧上到指定点距离最近的点坐标*/
    CPoint getShortestPoint2PointBy1Crcle(float x, float y, float radius, CPoint p);

    /*获得2个圆弧交点到指定点距离最近的点坐标*/
    CPoint getShortestPoint2PointBy2Crcle(CPoint p1, float r1, CPoint p2, float r2, CPoint lastp, bool has_last);

    /*获得2个圆弧交点*/
    int getPointsBy2Crcle(CPoint p1, float r1, CPoint p2, float r2, CPoint &rp1, CPoint &rp2);

    /*获得3个圆弧定位到的点*/
    CPoint getPointsBy3Crcle(CPoint p1, float r1, CPoint p2, float r2, CPoint p3, float r3);

};

#endif // DXBLOCATION_H
