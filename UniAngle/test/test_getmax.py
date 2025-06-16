import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

def concatenate_every_n_rows(arr, n=6):
    if len(arr) % n != 0:
        padding = (n - (len(arr) % n)) % n
        arr = np.pad(arr, ((0, padding), (0, 0)), mode='constant', constant_values=0)
    return arr.reshape(-1, n * arr.shape[1])

def preprocess(data_path):
    data = pd.read_excel(
        data_path, sheet_name="Sheet1", header=None).dropna().values

    targ_ls = []
    for i in range(data.shape[0]):
        if "plan" in data[i, 0]:
            data[i, 0] = np.nan
            continue
        
        temp = np.array([item for item in "".join(data[i, 0].split(",")).split(" ") if item != ""])
        targ_ls.append(temp[[0, 2, 4, 6, -1]])

    targ_ls = np.array(targ_ls)
    targ_ls = concatenate_every_n_rows(targ_ls, 6)
    
    return targ_ls


def main():
    data = preprocess(r"C:\WORK\Projects\UniAngle\data\360_all.xlsx")

    serial = "A,D,E,H,G,C".split(",")
    header = "id,rssi_f,pdoa_i,aoa_f,aoa_std"

    for i in range(6):
        data[:, i*5] = np.array([i]).repeat(data.shape[0])
    
    np.set_printoptions(edgeitems=10, linewidth=300)
    data = data.astype(np.float16)

    cal_info = dict()

    # step1: 角度框架转换
    for i in range(6):
        temp = np.zeros((data.shape[0], 8))
        temp[:, :4] = data[:, 1+5*i:5+5*i]
        temp[:, -4] = (-data[:, 3+5*i] + i*60)%360
        temp[:, -3] = (180 + data[:, 3+5*i] + i*60)%360
        cal_info[serial[i]] = temp

    # step2: 排列组合，计算得分
    combs = list(itertools.combinations(cal_info.keys(), 2))

    for line_index in range(data.shape[0]):
        for comb in combs:
            rssi = cal_info[comb[0]][line_index, 0] - cal_info[comb[1]][line_index, 0]
            if rssi > 0:
                cal_info[comb[0]][line_index, -2] += np.abs(rssi)
                cal_info[comb[1]][line_index, -1] += np.abs(rssi) * 0.5
            else:
                cal_info[comb[0]][line_index, -1] += np.abs(rssi) * 0.5
                cal_info[comb[1]][line_index, -2] += np.abs(rssi)

        
    # step3: 得分比较
    res_dic = dict()
    for s in serial:
        # 得到每个天线对应 正 / 负 分更大的一组
        res_dic[s] = np.zeros((cal_info[s].shape[0], 3))
        res = cal_info[s][:, -2] - cal_info[s][:, -1]
        
        positive = np.where(res >= 0)[0]   # 正分更大的索引
        negtive = np.where(res < 0)[0]   # 负分更大的索引

        res_dic[s][positive, 0] = cal_info[s][positive, -4] 
        res_dic[s][negtive, 0] = cal_info[s][negtive, -3]

        res_dic[s][:, 1] = cal_info[s][:, -2] 
        res_dic[s][:, 2] = cal_info[s][:, -1]

    logs = []
    for line in range(res_dic[serial[0]].shape[0]):
        id, ang, max_s = 0, 0, 0
        for s in serial:
            angle_score = max(res_dic[s][line, 1], res_dic[s][line, 2])
            if max_s < angle_score:
                id, ang, max_s = s, res_dic[s][line, 0], angle_score
        
        logs.append((id, ang, max_s))
    
    logs = np.array(logs)

    plt.scatter(np.arange(logs.shape[0]), logs[:, 1].astype(float), s=1)
    plt.show()

    
if __name__=="__main__":
    main()
