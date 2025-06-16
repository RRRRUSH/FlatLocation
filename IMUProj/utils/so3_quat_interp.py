
import pandas as pd
from so3_utils import *

def interpolate(x, t, t_int):
    """
    Interpolate ground truth at the sensor timestamps
    """

    x_int = np.zeros((t_int.shape[0], x.shape[1]))
    # quaternion interpolation
    t_int = torch.Tensor(t_int - t[0])
    t = torch.Tensor(t - t[0])
    qs = SO3.qnorm(torch.Tensor(x[:, :]))
    x_int[:, :] = SO3.qinterp(qs, t, t_int).numpy()
    return x_int

def down_sampling():
    raw = pd.read_csv(
        r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\input\quat_interp_test\nvidia_10r_rawq\data.csv", sep=",", header=None
        )

    mask = np.zeros(raw.shape[0], dtype=bool)
    mask[::10] = True

    for cols in raw.columns[-4:]:
        raw.loc[~mask, cols] = 0

    return raw

def main():
    ds_df = down_sampling()
    print(ds_df.head())

    quat_20hz = np.array(ds_df.iloc[::10, -4:]) # x
    ts_20hz = np.array(ds_df.iloc[::10, 0]) # t 20hz timestamp
    ts_200hz = np.array(ds_df.iloc[:, 0]) # t_int 200hz timestamp

    print(ts_20hz.shape, ts_200hz.shape, quat_20hz.shape)

    x_int = interpolate(quat_20hz, ts_20hz, ts_200hz)

    ds_df.iloc[:, -4:] = x_int
    print(ds_df.head())
    ds_df.to_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\input\quat_interp_test\nvidia_10r_interpq\data.csv", index=False, header=False)

if __name__=="__main__":
    main()