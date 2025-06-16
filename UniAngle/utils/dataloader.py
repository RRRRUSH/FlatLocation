import numpy as np
import pandas as pd
import bisect


def pdoa2aoa(pdoa):
    """ convert pdoa to aoa """
    phi = np.deg2rad(pdoa.astype(float))
    lamda = 0.07
    d = lamda / 2

    param = (phi * lamda) / (2 * np.pi * d)

    result = np.empty_like(param)
    mask_valid = (param <= 1) & (param >= -1)
    result[mask_valid] = np.rad2deg(np.arcsin(param[mask_valid]))
    result[~mask_valid] = np.nan

    return result

def from_group(path: str, ant_nums: int) -> dict:
    """
    将 dataraw 中的数据按 1s 分组，仅提取每秒前 100ms 的集中输出数据，
    """
    dataraw = pd.read_csv(path, header=None, sep=',').values
    dataraw = dataraw.astype(float)

    delta_ts = dataraw[1:, -1] - dataraw[:-1, -1]
    threshold = np.average([np.max(delta_ts), np.min(delta_ts)])

    gap_indices = np.where(delta_ts > threshold)[0]
    group_start_index = 0
    result = {}
    for i in gap_indices:
        result[dataraw[i, -1]] = dataraw[group_start_index:(i+1), :]
        group_start_index = i+1

    for ts in result.keys():
        temp = [list() for _ in range(ant_nums)]
        for ant_id in range(ant_nums):
            temp[ant_id] = result[ts][result[ts][:, 0] == float(ant_id)]
        result[ts] = temp

    return result

def from_stream(path: str, ant_nums: int, window_size: int = 500) -> dict:
    """
    嵌入式端不处理，实时记录输出的所有数据行 e.g.
         0,   -54.5,   45.8,   -15.5,    0.0,4909592
         1,   -58.8,  -109.1,   51.0,    0.0,4909592
         3,   -64.9,   80.9,   -17.1,    0.0,4909592
         4,   -62.6,  -120.9,   44.4,    0.0,4909592
         0,   -54.2,   46.3,   -15.7,    0.0,4909623
         1,   -58.5,  -112.0,   52.6,    0.0,4909659

    return:
        key: time
        value: measurements of antennas in one time window, shape(ant_nums, n, 6)

        e.g.
        time stamp 5921.0
        0 [array([ 0.000e+00, -5.450e+01,  4.600e+00, -1.500e+00,  0.000e+00,  5.858e+03]),
            array([ 0.000e+00, -5.430e+01,  4.800e+00, -1.600e+00,  0.000e+00,  5.895e+03])]
        1 [array([ 1.000e+00, -6.120e+01, -1.271e+02,  6.140e+01,  0.000e+00,  5.858e+03]),
            array([ 1.000e+00, -6.110e+01, -1.254e+02,  6.040e+01,  0.000e+00,  5.895e+03])]
        2 []
        3 [array([ 3.000e+00, -6.440e+01,  7.250e+01, -1.520e+01,  0.000e+00,  5.895e+03])]
        4 [array([ 4.000e+00, -6.510e+01,  1.865e+02,  7.830e+01,  0.000e+00,  5.895e+03])]
        5 [array([ 5.000e+00, -5.750e+01, -1.918e+02, -4.940e+01,  0.000e+00,  5.822e+03])]

    """
    with open(path, 'r') as f:
        data_raw = f.readlines()
    if len(data_raw) == 0:
        print(f"{path} file is empty!")
        exit()

    data_all = []
    for data_line in data_raw:
        if "(Connection lost)" in data_line.strip():
            break
        data_all.append([float(item.strip()) for item in data_line.split(",")])

    result = dict()
    if not data_all:
        return result

    data_all = np.array(data_all)
    timeline = np.array(np.arange(data_all[0, -1], data_all[-1, -1] + 1, 1))

    ts_i, time_step = 0, window_size
    while ts_i < len(timeline):
        if ts_i + time_step > len(timeline):
            time_step = len(timeline) - ts_i

        beg, end = timeline[ts_i], timeline[ts_i + time_step - 1]

        group = data_all[np.where((beg <= data_all[:, -1]) & (data_all[:, -1] <= end))]
        temp = [list() for _ in range(ant_nums)]

        for meas in group:
            # 后处理计算 aoa
            meas[3] = pdoa2aoa(meas[2])
            temp[int(meas[0])].append(meas)

        result[end] = temp

        ts_i += time_step

    return result

def load_group_data(path: str, ant_nums: int) -> dict:
    """
    get valid measurements from group data

    input: path of the data file and num of antennas
    return: a dict of data,
        key: timestamp(float),
        value: a numpy array of shape (nums of valid measurement, 6)
            [[ src_1, rssi_f_1, pdoa_i_1, aoa_f_1, aoa_std_1, time_1]
                                ...
            [ src_n, rssi_f_n, pdoa_i_n, aoa_f_n, aoa_std_n, time_n]]
    """
    data = from_group(path, ant_nums)
    assert ant_nums is not None, "ant_nums is None"

    valid_id_mark = [-1] * ant_nums
    for ts in sorted(data.keys()):
        for ant_id in range(ant_nums):
            if len(data[ts][ant_id]) > 0:
                valid_id_mark[ant_id] = ts
            else:
                if valid_id_mark[ant_id] != -1:
                    data[ts][ant_id] = data[valid_id_mark[ant_id]][ant_id]

            data[ts][ant_id][:, 3] = pdoa2aoa(data[ts][ant_id][:, 2])
            data[ts][ant_id] = data[ts][ant_id][~np.isnan(data[ts][ant_id][:, 3].astype(float))]

            if len(data[ts][ant_id]) > 0:
                data[ts][ant_id][:, -2] = np.std(data[ts][ant_id][:, 3])

    result = {ts: np.concatenate(data[ts]) for ts in data}
    return result


def load_stream_data(path: str, ant_nums: int, calib_angle: bool = False) -> dict:
    data = from_stream(path, ant_nums, 1)
    assert ant_nums is not None, "ant_nums is None"

    # Step 1: Preprocess data to filter empty measurements
    result = {}
    for ts in data:
        temp = []
        for ant_meases in data[ts]:
            if len(ant_meases) > 0:
                temp.append(ant_meases)
        if temp:
            result[ts] = np.concatenate(temp)

    # Convert to sorted lists for efficient access
    sorted_ts = sorted(result.keys())
    sorted_data = [result[ts] for ts in sorted_ts]
    n = len(sorted_ts)
    output = {}

    # Step 2: Sliding window with binary search
    valid_measurement_duration = 1000
    for i in range(n):
        current_ts = sorted_ts[i]
        start_ts = max(current_ts - valid_measurement_duration, sorted_ts[0])
        left_idx = bisect.bisect_left(sorted_ts, start_ts)

        # Collect all valid data in the window [left_idx, i]
        window_data = []
        for j in range(left_idx, i + 1):
            if len(sorted_data[j]) > 0:
                window_data.append(sorted_data[j])

        if window_data:
            output[current_ts] = np.concatenate(window_data)
            ids = set(output[current_ts][:, 0])
            for ant_id in ids:
                meas_group_indices = output[current_ts][:, 0] == float(ant_id)
                output[current_ts][meas_group_indices, -2] = np.std(output[current_ts][meas_group_indices, 2])
            output[current_ts] = output[current_ts][~np.isnan(output[current_ts][:, 3])]
        else:
            output[current_ts] = np.array([])  # Return empty array if no

    return output

if __name__=="__main__":
    # # test read from stream
    # res_dict = from_stream(r"../data/0424/0.txt", 6)
    # for key in res_dict:
    #     print(key)
    #     for index, item in enumerate(res_dict[key]):
    #         print(index, item)
    np.set_printoptions(linewidth=300)

    # # test load_stream_data
    # res_dict = load_stream_data(r"../data/0424/0.txt", 6)
    #
    # for key in res_dict:
    #     print(key)
    #     print(res_dict[key])
    data = load_group_data(r"../data/0510_360/merge/0.txt", 4)
    print(data)