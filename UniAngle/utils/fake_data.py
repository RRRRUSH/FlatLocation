import matplotlib.pyplot as plt
import numpy as np
import random

# test: 测试基站半径对于角度测量产生的影响
def test1():
    beta = 0

    y = 100
    x = 0.07

    for theta in range(1, 90):
        theta_temp = np.deg2rad(theta)

        beta = np.rad2deg(np.arctan((y*np.sin(theta_temp) / (x+y*np.cos(theta_temp)))))
        print(theta, beta, "gap:", (theta-beta))

def pdoa2aoa(pdoa):
    # -196.1+360 65.58223843017042
    # -174.6    -75.93013225242788

    phi = np.deg2rad(pdoa)
    lamda = 0.07
    d = lamda / 2

    param = (phi * lamda) / (2 * np.pi * d)

    theta = np.rad2deg(np.arcsin(param))

    return theta

# 新增方法：AOA 转换为 PDOA
def aoa2pdoa(aoa):
    """
    根据 AOA 计算 PDOA
    """
    theta = np.deg2rad(aoa)  # 将 AOA 转换为弧度
    lamda = 0.07
    d = lamda / 2

    # 根据公式反向计算 PDOA
    phi = (2 * np.pi * d * np.sin(theta)) / lamda
    pdoa = np.rad2deg(phi)

    return pdoa

# 新增方法：生成假数据
def generate_fake_data(antenna_id, rssi_base, rssi_range, pdoa_base, pdoa_range, timestamp):
    """
    根据输入参数生成假数据，格式为：天线编号，RSSI，PDOA，AOA，0，时间戳
    """
    # 生成随机RSSI
    rssi = random.uniform(rssi_base - rssi_range, rssi_base + rssi_range)
    # 生成随机PDOA
    pdoa = random.uniform(pdoa_base - pdoa_range, pdoa_base + pdoa_range)
    # 生成随机AOA
    aoa = pdoa2aoa(pdoa)

    # 返回格式化数据
    return f"{antenna_id},{rssi:.2f},{pdoa:.2f},{aoa:.2f},0,{timestamp}"

def sys_angle_2_aoa(angle,ant):
    if angle >= ant*60+90 and angle <= (ant*60-90 +360)%360:
        aoa = -angle + ant*60
    else:
        aoa = angle -180 - ant*60
    return aoa

# 新增方法：生成一段时间的数据
def generate_time_series_data(base_timestamp, real_angle):
    """
    生成一段时间的数据，包含10次大循环，每次循环内调用6次generate_fake_data方法
    """
    all_data = []  # 存储所有生成的数据
    ant_lost = 6
    for ant_no_lost in range(6):  # 外层循环6次
        for outer_loop in range(10):  # 外层循环10次
            current_timestamp = base_timestamp + outer_loop * 110 + ant_no_lost * 1100   # 每次循环时间戳增加110ms
            print(current_timestamp)

            # 0
            if ant_lost != 0:
                fake_data = generate_fake_data(
                    antenna_id=0,
                    rssi_base=-55,
                    rssi_range=5,
                    pdoa_base=aoa2pdoa(sys_angle_2_aoa(real_angle,0)),
                    pdoa_range=10,
                    timestamp=current_timestamp
                )
                all_data.append(fake_data)  # 将生成的数据添加到列表中
            # 1
            if ant_lost != 1:
                fake_data = generate_fake_data(
                    antenna_id=1,
                    rssi_base=-60,
                    rssi_range=5,
                    pdoa_base=aoa2pdoa(sys_angle_2_aoa(real_angle,1)),
                    pdoa_range=15,
                    timestamp=current_timestamp
                )
                all_data.append(fake_data)  # 将生成的数据添加到列表中
            # 2
            if ant_lost != 2    :
                fake_data = generate_fake_data(
                    antenna_id=2,
                    rssi_base=-60,
                    rssi_range=5,
                    pdoa_base=aoa2pdoa(sys_angle_2_aoa(real_angle,2)),
                    pdoa_range=15,
                    timestamp=current_timestamp
                )
                all_data.append(fake_data)  # 将生成的数据添加到列表中
            # 3
            if ant_lost != 3:
                fake_data = generate_fake_data(
                    antenna_id=3,
                    rssi_base=-64,
                    rssi_range=5,
                    pdoa_base=aoa2pdoa(sys_angle_2_aoa(real_angle,3)),
                    pdoa_range=15,
                    timestamp=current_timestamp
                )
                all_data.append(fake_data)  # 将生成的数据添加到列表中
            # 4
            if ant_lost != 4:
                fake_data = generate_fake_data(
                    antenna_id=4,
                    rssi_base=-60,
                    rssi_range=5,
                    pdoa_base=aoa2pdoa(sys_angle_2_aoa(real_angle,4)),
                    pdoa_range=15,
                    timestamp=current_timestamp
                )
                all_data.append(fake_data)  # 将生成的数据添加到列表中
            # 5
            if ant_lost != 5:
                fake_data = generate_fake_data(
                    antenna_id=5,
                    rssi_base=-60,
                    rssi_range=5,
                    pdoa_base=aoa2pdoa(sys_angle_2_aoa(real_angle,5)),
                    pdoa_range=15,
                    timestamp=current_timestamp
                )
                all_data.append(fake_data)  # 将生成的数据添加到列表中
    return current_timestamp, all_data  # 返回所有生成的数据

def show_pdoa_aoa():
    lamda = 0.07
    d = lamda / 2

    points = []
    for alpha in range(-360, 360):
        alpha_rad = np.deg2rad(alpha % 180)

        points.append([alpha, np.arcsin((alpha_rad * lamda) / (2 * np.pi * d))])

    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], s=3)
    plt.show()

if __name__ == "__main__":
    # # print(pdoa2aoa(155))
    # # print(aoa2pdoa(59.44))  # 测试 AOA 转换为 PDOA
    #
    # # 调用新方法生成一段时间的数据
    # angle = 360
    # all = []
    # start = 0
    # for _ in range(10):
    #     ts, time_series_data = generate_time_series_data(start, angle)
    #     all.extend(time_series_data)
    #     start = ts
    #
    # with open(f"../data/generate/{angle}.txt", 'w') as f:
    #     f.write("\n".join(all))
    # # for data in time_series_data:
    # #     print(data)  # 打印生成的数据

    # show_pdoa_aoa()
    print(aoa2pdoa(90))