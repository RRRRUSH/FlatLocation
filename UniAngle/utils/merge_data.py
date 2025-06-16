import glob
import os.path

import numpy as np
import pandas as pd


class DataMerger:
    def __init__(self, path, without_windows_ts=False):
        self.path = path
        self.without_windows_ts = without_windows_ts

        self.write_id_totxt()
        self.merge_ants_with_ts()

    def write_ant_id(self, subdir):
        txt_paths = glob.glob(os.path.join(subdir, "*.txt"))
        ant_id = os.path.basename(subdir)

        os.makedirs(os.path.join(subdir, 'withid'), exist_ok=True)
        for p in txt_paths:
            print(f'Processing: {p}')
            data = pd.read_csv(p, sep=",", header=None, on_bad_lines='skip').values
            data[:, 0] = np.array([int(ant_id)] * data.shape[0])
            pd.DataFrame(data).to_csv(
                os.path.join(subdir, 'withid', os.path.basename(p)), sep=",", header=False, index=False
            )

    def write_id_totxt(self):
        subdirs = [p for p in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, p))]
        for subdir in subdirs:
            self.write_ant_id(os.path.join(self.path, subdir))

    def merge_ants_with_ts(self):
        data_withid = []
        for i in os.walk(self.path):
            if "withid" in i[0]:
                data_withid.append(i)

        os.makedirs(os.path.join(self.path, 'merge'), exist_ok=True)
        for txt in data_withid[0][-1]:
            dfs = []
            for i in range(4):
                datapath = os.path.join(data_withid[i][0], txt)
                if self.without_windows_ts:
                    dfs.append(pd.read_csv(datapath, header=None, delimiter=",").values)
                else:
                    withts = pd.read_csv(datapath, header=None, delimiter=",").values
                    ts = np.char.replace(withts[:, -1].astype(str), '0.00[', '')
                    ts = np.char.strip(ts.astype(str))
                    withts[:, -2] = pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S.%f]').astype(np.int64) // 1e6
                    # 为了显示取 7 位时间戳
                    withts[:, -2] = withts[:, -2] % 1e07
                    dfs.append(withts)

            all_ants = np.concatenate(dfs, axis=0)
            pd.DataFrame(all_ants[np.argsort(all_ants[:, -2])][:, :-1]).to_csv(
                os.path.join(self.path, 'merge', txt), sep=',', header=False, index=False
            )


if __name__=="__main__":
    root_path = r"E:\HaozhanLi\Project\UniAngle\data\0510_360"
    DataMerger(root_path, without_windows_ts=True)
