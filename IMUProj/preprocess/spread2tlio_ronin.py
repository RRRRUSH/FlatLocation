from base import *

from scipy.spatial.transform import Rotation as R

class DataCache():
    def __init__(self, acce, gyro, q, outpath):
        self.out = outpath

        min_shape = min(acce.shape[0], gyro.shape[0])

        ts = np.arange(min_shape) * 5 / 1e03 * 1e09 # s
        
        self.ts = ts.reshape(-1, 1)
        assert np.all(np.diff(ts) >= 0), 'Timestamp array is not sorted'

        self.gyro = gyro[:min_shape, :]
        self.acce = acce[:min_shape, :]
        self.rv = q[:min_shape, :]

        self.ronin()
        self.tlio()
        
    def ronin(self):
        # 保存处理后的数据
        os.makedirs(self.out, exist_ok=True)
        np.savetxt(os.path.join(self.out, "acce.txt"), np.concatenate([self.ts / 1e09, self.acce], axis=1), fmt='%.9f')
        np.savetxt(os.path.join(self.out, "gyro.txt"), np.concatenate([self.ts / 1e09, self.gyro], axis=1), fmt='%.9f')
        np.savetxt(os.path.join(self.out, "game_rv.txt"), np.concatenate([self.ts / 1e09, self.rv], axis=1), fmt='%.9f')

    def tlio(self):
        imu_samples_0 = np.concatenate([self.ts, np.zeros((self.ts.shape[0], 1)), self.gyro, self.acce], axis=1)
        # 保存组合后的数据
        np.savetxt(os.path.join(self.out, 'imu_samples_0.csv'), imu_samples_0, delimiter=',',
                header='#timestamp [ns],temperature [degC],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]')

def ch040(in_path, out_path):
    data = pd.read_csv(in_path, sep=",").values

    DataCache(data[:, 3:6] * 9.8, data[:, 6:9] / 57.3, data[:, -9:-5], out_path)


def h1(in_path, out_path):
    data = pd.read_csv(in_path, sep=",", header=None).values
    
    DataCache(data[:, 3:6], np.deg2rad(data[:, :3]), data[:, 6:10], out_path)
    
def flat3_rect(in_path, out_path):
    data = pd.read_csv(in_path, sep=",", header=None).values

    DataCache(data[:, 1:4], data[:, 4:7], data[:, 7:11], out_path)

def rot_test(in_path, out_path):
    data = pd.read_csv(in_path, sep=",", header=None).dropna().values

    # label = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\label.csv", sep=",").values

    # 取label_1的姿态转imu给ronin推理
    raw_tx = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\tx\label_1.csv", sep=",").values
    befor_vicon = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\tx\pos_input.csv", sep=",").values
    q_l = raw_tx[:, 4:8]
    min_shape = min(data.shape[0], q_l.shape[0])
    q_l_inv = R.from_quat(q_l[:min_shape, [1, 2, 3, 0]]).inv()

    # # info 旋转
    # body2imu = np.array([0.99896, -0.045276, 0.00531674, 0.0451366, 0.998694, 0.0239288, -0.0063932, -0.0236639, 0.9997]).reshape(3, 3)
    # info_b2i = R.from_matrix(body2imu)

    # g2v = np.array([1, 4.20117e-07, -0.000171892, 0, 0.999997, 0.00244407, 0.000171892, -0.00244407, 0.999997]).reshape(3, 3)
    # info_g2v = R.from_matrix(g2v)

    # pos = label[:, 1:4]
    # posin_v = info_g2v.apply(pos)
    # posin_i = info_b2i.apply(posin_v)

    # fig = plt.figure()
    # fig.add_subplot(111, projection="3d")
    # plt.plot(rotted_pos[:, 0], rotted_pos[:, 1], rotted_pos[:, 2], color='red')

    plt.plot(befor_vicon[:, 1], -befor_vicon[:, 3], color='blue', label="ground truth")

    plt.plot(raw_tx[:, 1], raw_tx[:, 2], color='black', label="raw")
    plt.show()

    exit()

    rotted = np.zeros((min_shape, data.shape[1] - 1))

    rotted[:min_shape, 0] = np.arange(min_shape) * 0.005
    rotted[:min_shape, 1:4] = q_l_inv.apply(data[:min_shape, 1:4]) # acc
    rotted[:min_shape, 4:7] = q_l_inv.apply(data[:min_shape, 4:7]) # gyro
    rotted[:min_shape, [8, 9, 10, 7]] = (q_l_inv.inv() * R.from_quat(data[:min_shape, [8, 9, 10, 7]]) * q_l_inv).as_quat(True) # quat

    DataCache(rotted[:, 1:4], rotted[:, 4:7], rotted[:, 7:11], out_path)

    # np.savetxt(
    #     r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\imu_inv_rot.txt",
    #     rotted, header="ts,ax,ay,az,gx,gy,gz,qw,qx,qy,qz", delimiter=",", comments=""
    # )
    # DataCache(data[:, 1:4], data[:, 4:7], data[:, [10,7,8,9]], out_path)

def show_euler():
    data = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\tx\label_1.csv").values
    raw = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\tx\pos_input.csv").values
    imu = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\tx\imu_output.csv").dropna().values

    # xyzw
    quat = data[:, [5 , 6, 7, 4]]
    raw_q = raw[:, 4:8]
    imu_q = imu[:, [-3, -2, -1, -4]]
    pos = data[:, 1:4]

    euler = R.from_quat(quat).as_euler("xyz", degrees=True)
    raw_euler = R.from_quat(raw_q).as_euler("xzy", degrees=True)

    rot = R.from_euler("x", -90, degrees=True)
    raw_euler = (rot.inv() * R.from_euler("xzy", raw_euler, degrees=True) * rot).as_euler("xyz", degrees=True)

    imu_euler = R.from_quat(imu_q).as_euler("xyz", degrees=True)

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(euler)), euler[:, 0], color="r")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(euler)), euler[:, 1], color="r")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(euler)), euler[:, 2], color="r")
    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(raw_euler)), raw_euler[:, 0], color="b")
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(raw_euler)), raw_euler[:, 1], color="b")
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(raw_euler)), raw_euler[:, 2], color="b")
    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(imu_euler)), imu_euler[:, 0])
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(imu_euler)), imu_euler[:, 1])
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(imu_euler)), imu_euler[:, 2])
    plt.show()
    
    # yaw = np.deg2rad(euler[:, 2])
    # vec = np.array([np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))]).T

    # nums = 1700
    # step = 100

    # plt.quiver(pos[:nums:step, 0], pos[:nums:step, 1], vec[:nums:step, 0], vec[:nums:step, 1], color="r")
    # plt.plot(pos[:nums, 0], pos[:nums, 1])
    # plt.show()

def lidar():
    data = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_1\154\imu.csv").values

    DataCache(data[:, 1:4], data[:, 4:7], data[:, -4:], r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_1\154")



if __name__ == "__main__":
    # batch()

    # sn = "0325_4"
    # root_p = rf"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\raw\xxb_lsf_imucali\{sn}"
    # out_p = rf"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\xxb_lsf_imucali\{sn}"

    # # h1(
    # #     in_path=r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\raw\test\data2.txt",
    # #     out_path=r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0407_3\222"
    # # )

    # paths = glob.glob(os.path.join(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\raw\mag_test\0410_1", "*.csv"))
    # for p in paths:

    #     ch040(
    #         in_path=p,
    #         out_path=os.path.join(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0410_1", os.path.basename(p).split(".")[0])
    #     )

    # rot_test(
    #     in_path=r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\imu.txt",
    #     out_path=r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1"
    # )

    # show_euler()

    lidar()