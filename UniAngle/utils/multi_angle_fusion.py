import os

os.environ["OMP_NUM_THREADS"] = "1"

import time

import numpy as np

from sklearn.neighbors import NearestNeighbors
from utils.dataloader import load_stream_data, load_group_data

class Weights:
    def __init__(self, score_weights: list):
        self.weights_dict = dict()
        self.max_angle = 60
        self.max_time = 1000 # ms

        self.angles_limit = [-40, 60]

        assert score_weights is not None
        self.score_weights = np.array([score_weights]) / np.sum(score_weights)
        print("<<SCORES_WEIGHTS>>", self.score_weights)

        self.history = None

    def get_scores(self, ts, meas, history_result):
        """ 根据当前时刻的测量结果计算得分 """
        # 角度范围 流逝时间 标准差 密度 历史角度 信号强度分类
        scores = np.zeros((meas.shape[0], 7))
        scores[:, 0] = self.angle_scope(meas[:, 3])
        scores[:, 1] = self.elapsed_time(ts - meas[:, 5])
        scores[:, 2] = self.meas_std(meas)
        scores[:, 3] = self.density(meas[:, -1])
        if history_result is not None:
            scores[:, 4] = self.history_angle(ts, meas[:, -1], history_result)
        scores[:, 5] = self.rssi_cluster(meas[:, 1])

        scores[:, -1] = np.sum(scores[:, :-1] * self.score_weights, axis=1)

        return scores

    def angle_scope(self, aoa):
        aligned_aoa = np.zeros_like(aoa)
        left_limit, right_limit = self.angles_limit
        # METHOD 1
        if left_limit > -self.max_angle:
            aligned_aoa[aoa <= left_limit] = aoa[aoa <= left_limit] - self.max_angle - left_limit
        if right_limit < self.max_angle:
            aligned_aoa[aoa >= right_limit] = aoa[aoa >= right_limit] + self.max_angle - right_limit

        return np.cos(np.deg2rad(aligned_aoa))

    def elapsed_time(self, t):
        return 1 - t / self.max_time

    def meas_std(self, meas):
        # 根据天线当前窗口内的数据量来定义方差项的可信度，不同天线不一样
        ants = set(meas[:, 0])

        max_std_num = 0
        std_score = np.zeros(meas.shape[0])

        for ant_id in ants:
            mark = (meas[:, 0] == float(ant_id))
            if meas[mark].shape[0] > max_std_num:
                max_std_num = meas[mark].shape[0]
        for ant_id in ants:
            mark = (meas[:, 0] == float(ant_id))
            std_score[mark] = (meas[mark].shape[0] / max_std_num) * (1 - meas[mark, 4] / (np.max(meas[:, 4]) + 1e-06))
        return std_score

    def density(self, meas_angle):
        # angle to vector
        angle_vectors = np.array([[np.cos(theta), np.sin(theta)] for theta in meas_angle])
        K = np.min([meas_angle.shape[0] - 1, int(np.sqrt(2*meas_angle.shape[0]))])

        nn = NearestNeighbors(n_neighbors=K+1).fit(angle_vectors)
        k_neighbors_dis, _ = nn.kneighbors(angle_vectors)
        angles_dis = np.sum(k_neighbors_dis, axis=1)

        density = 1 - angles_dis / (np.max(angles_dis) + 1e-06)

        return density

    def history_angle(self, ts_now, meas_angle, history_result):
        # ts, self.angle_result[ts][-1], self.score_result[ts][-1]

        history_ts, history_meas, history_score = history_result

        delta_angle = np.abs(meas_angle - history_meas[-1]) % 360
        delta_angle = np.minimum(delta_angle, 360 - delta_angle)

        score = 1 - delta_angle / np.max(delta_angle + 1e-6)

        history_score_result = score * history_score[-1] * (1 - (ts_now-history_ts) / self.max_time)
        return history_score_result

    def rssi_cluster(self, rssi):

        sorted_indices = np.argsort(rssi)
        rssi_max = rssi[sorted_indices[-1]] + 100

        return (rssi + 100) / rssi_max

class AngleFusion:
    def __init__(self, data_dir: str, antenna_nums: int, score_weights: list = None, calibration: bool = False):

        assert data_dir is not None
        assert antenna_nums is not None

        self.root_dir = data_dir
        self.antenna_nums = antenna_nums

        self.score_weights = score_weights

        self.angle_result = dict()
        self.score_result = dict()

        self.fusion(calibration)

    def fusion(self, calibration):
        """
            key: timestamp(float),
            value: a numpy array of shape (nums of valid measurement, 6)
                [[ src_1, rssi_f_1, pdoa_i_1, aoa_f_1, aoa_std_1, time_1, pos_mark, pos_angle_1]
                 [ src_1, rssi_f_1, pdoa_i_1, aoa_f_1, aoa_std_1, time_1, neg_mark, neg_angle_1]
                                        ...
                 [ src_n, rssi_f_n, pdoa_i_n, aoa_f_n, aoa_std_n, time_n, pos_mark, pos_angle_n]
                 [ src_n, rssi_f_n, pdoa_i_n, aoa_f_n, aoa_std_n, time_n, neg_mark, neg_angle_n]
        """
        beg = time.time()
        self.valid_meas_dict = load_group_data(self.root_dir, self.antenna_nums)
        print("load_stream_data spend", time.time() - beg)

        self.weights = Weights(self.score_weights)

        history_result = None

        for ts, meas in self.valid_meas_dict.items():
            if len(meas) != 0:
                sys_coordinate_meas = self.convert_coordinate(meas)
                self.valid_meas_dict[ts] = sys_coordinate_meas

                scores = self.weights.get_scores(ts, sys_coordinate_meas, history_result)

                scores_indices = np.argsort(scores[:, -1])[::-1]
                if len(scores_indices) > 20:
                    group_high = scores_indices[:10]
                    group_low = scores_indices[-10:]
                else:
                    low = int(len(scores_indices) / 2)
                    group_high = scores_indices[:low]
                    group_low = scores_indices[-len(scores_indices)+low:]
                high_low_indices = np.concatenate([group_high, group_low])
                self.angle_result[ts] = self.valid_meas_dict[ts][high_low_indices]
                self.score_result[ts] = scores[high_low_indices]

                history_result = ts, self.angle_result[ts][-1], self.score_result[ts][-1]
            else:
                history_result = None

    def convert_coordinate(self, meas):
        ant_id, aoa = meas[:, 0], meas[:, 3]

        # aoa 的正负取决于天线左侧为正还是右侧为正
        # 左侧为负则取 -aoa，反之取 aoa

        # # 6-ant_id 表示天线顺序（012345）为逆时针递减
        # pos_angle = (-aoa + (6-ant_id) * (360 / self.antenna_nums)) % 360
        # neg_angle = (aoa + 180 + (6-ant_id) * (360 / self.antenna_nums)) % 360

        # ant_id 表示天线顺序（012345）为逆时针递增的
        pos_angle = (aoa + ant_id * (360 / self.antenna_nums)) % 360
        # neg_angle = (-aoa + 180 + ant_id * (360 / self.antenna_nums)) % 360

        pos_mark = np.ones((pos_angle.shape[0], 1))
        # neg_mark = -np.ones((neg_angle.shape[0], 1))

        with_pos_angle = np.concatenate(
            [meas[:, :], pos_mark, pos_angle.reshape(-1, 1)], axis=1)
        # with_neg_angle = np.concatenate(
        #     [meas[:, :], neg_mark, neg_angle.reshape(-1, 1)], axis=1)

        return np.concatenate([with_pos_angle], axis=0)
        # return np.concatenate([with_pos_angle, with_neg_angle], axis=0)

    def cali_raw(self, calibration):
        pass

    def get_result(self):
        return self.angle_result, self.score_result


if __name__=="__main__":
    af = AngleFusion(r"../data/0506/merge/10.txt", 4, [1, 1, 1, 1, 1, 1])
    np.set_printoptions(linewidth=300)
    # print(af.valid_meas_dict)
    #
    # for ts in af.valid_meas_dict.keys():
    #     print(ts)
    #     print(af.valid_meas_dict[ts])

    angle_result, score_result = af.get_result()

    for ts in angle_result.keys():
        print(ts)
        print(angle_result[ts])
