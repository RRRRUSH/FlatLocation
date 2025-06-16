import os
import numpy as np
import pandas as pd

def cbn(q):
    """ 将四元数转换为旋转矩阵，转换关系要求q为单位四元数 """
    # 将输入转换为 NumPy 数组并展平为一维
    q = np.asarray(q).flatten()

    if len(q) != 4:
        raise ValueError("输入的四元数必须是 4 元素向量。")
    
    # 提取四元数分量
    q0, q1, q2, q3 = q
    
    # 计算各分量的平方
    q00 = q0 * q0
    q11 = q1 * q1
    q22 = q2 * q2
    q33 = q3 * q3
    
    # 初始化 3x3 零矩阵
    c = np.zeros((3, 3))
    
    # 填充旋转矩阵元素
    c = np.array([
        [   q00 + q11 - q22 - q33, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [ 2 * (q1 * q2 + q0 * q3),   q00 - q11 + q22 - q33, 2 * (q2 * q3 - q0 * q1)],
        [ 2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1),   q00 - q11 - q22 + q33],
    ], dtype=np.float32)

    return c

def getoula(q):
    """ 将四元数转换为欧拉角（以度为单位） """
    # 确保 q 是 NumPy 数组并展平为一维
    q = np.asarray(q).flatten()
    if len(q) != 4:
        raise ValueError("输入的四元数必须是 4 元素向量。")
    
    # 获取旋转矩阵 cnb（转置 cbn(q)）
    cnb = cbn(q).T
    
    # 初始化欧拉角
    ou = np.zeros(3)
    
    # 计算欧拉角（弧度）ZXY 旋转顺序
    ou[0] = np.arctan2(-cnb[1, 0], cnb[1, 1])
    ou[1] = np.arctan2(cnb[1, 2], np.sqrt(cnb[0, 2]**2 + cnb[2, 2]**2))
    ou[2] = np.arctan2(-cnb[0, 2], cnb[2, 2])
    
    # 将欧拉角从弧度转换为度
    ou = ou * 180 / np.pi
    
    return ou

def getfk(atti, accn):
    """ 根据姿态和加速度生成 8x8 的状态转移矩阵 F """
    # 将加速度转换为 NumPy 数组并展平为一维
    accn = np.asarray(accn).flatten()
    if len(accn) != 3:
        raise ValueError("加速度 accn 必须是 3 元素向量。")
    
    F = np.zeros((8, 8))
    
    fE = accn[0]  # 东方向加速度
    fN = accn[1]  # 北方向加速度
    fU = accn[2]  # 天方向加速度
    
    # 构造姿态对速度影响的子矩阵 Fav
    Fav = np.array([
        [0,   -fU,  fN],
        [fU,   0,  -fE],
        [-fN,  fE,   0]
    ])
    
    F[0:2, 2:5] = Fav[0:2, :]
    
    # 计算旋转矩阵 cbnm
    cbnm = cbn(atti)
    
    # 将陀螺仪零偏对姿态的影响赋值给 F 的第 3-5 行、第 6-8 列
    F[2:5, 5:8] = -cbnm
    
    return F

def kal(Z, H, Pk, Q, R, Phi):
    """ 卡尔曼滤波的预测和更新步骤
    Z : array_like 观测向量。
    H : array_like 观测矩阵。
    Pk : array_like 先验协方差矩阵。
    Q : array_like 过程噪声协方差矩阵。
    R : array_like 观测噪声协方差矩阵。
    Phi : array_like 状态转移矩阵。
    
    返回:
    X : ndarray
        估计的状态向量。
    Pk : ndarray
        后验协方差矩阵。
    """
    # 预测协方差矩阵 Pkk
    Pkk = Phi @ Pk @ Phi.T + Q
    
    # 计算卡尔曼增益 K
    K = Pkk @ H.T @ np.linalg.inv(H @ Pkk @ H.T + R)
    
    # 估计状态 X
    X = K @ Z
    
    # 计算 IKH = I - K*H
    IKH = np.eye(Phi.shape[0]) - K @ H
    
    # 更新后验协方差矩阵 Pk
    Pk = IKH @ Pkk @ IKH.T + K @ R @ K.T
    
    return X, Pk

def qupdate(qa, th):
    """ Update quaternion qa based on rotation vector th.
    
    Parameters:
    qa : array_like
        Original quaternion, a 4-element vector.
    th : array_like
        Rotation vector, a 3-element vector.
    
    Returns:
    qb : ndarray
        Updated quaternion.
    """
    # Ensure inputs are NumPy arrays and flattened
    qa = np.asarray(qa).flatten()
    th = np.asarray(th).flatten()
    if len(qa) != 4 or len(th) != 3:
        raise ValueError("qa must be a 4-element vector, th must be a 3-element vector.")
    
    # Calculate the norm of the rotation vector th
    thabs = np.sqrt(th[0]**2 + th[1]**2 + th[2]**2)
    
    # Construct the THM matrix (4x4)
    THM = np.array([
        [    0, -th[0], -th[1], -th[2]],
        [th[0],      0,  th[2], -th[1]],
        [th[1], -th[2],      0,  th[0]],
        [th[2],  th[1], -th[0],      0],
    ])
    
    # Construct the A matrix
    A = np.eye(4) * np.cos(thabs * 0.5)
    
    # Update A based on the magnitude of thabs
    if thabs < 1e-6:
        A += THM * 0.5
    else:
        A += THM * (np.sin(thabs * 0.5) / thabs)
    
    # Compute the updated quaternion qb
    qb = A @ qa
    
    return qb

def setoula(yawdeg, pitchdeg, rolldeg):
    """
    Convert Euler angles to quaternion with rotations in the order: yaw (z), pitch (x), roll (y).
    
    Parameters:
    yawdeg : float
        Yaw angle in degrees (rotation around z-axis).
    pitchdeg : float
        Pitch angle in degrees (rotation around x-axis).
    rolldeg : float
        Roll angle in degrees (rotation around y-axis).
    
    Returns:
    q : ndarray
        Quaternion [w, x, y, z].
    """
    # Initialize quaternion as identity [w, x, y, z]
    q = np.array([1.0, 0.0, 0.0, 0.0])

    q = qupdate(q, np.array([       0,       0, yawdeg]) * (np.pi / 180))
    q = qupdate(q, np.array([pitchdeg,       0,      0]) * (np.pi / 180))
    q = qupdate(q, np.array([       0, rolldeg,      0]) * (np.pi / 180))
    
    return q

def inst_main(data):

    L = len(data)

    # 卡尔曼滤波参数
    Pk1 = np.diag([0.1, 0.1, 3e-4, 3e-4, 3e-4, 0, 0, 0])
    Q0 = np.diag([1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 0, 0, 0])
    Phi1 = np.eye(8)
    R = np.diag([10, 10])
    H = np.eye(2, 8)
    Q1 = np.zeros((8, 8))
    Z1 = np.zeros((2, 1))
    X1 = np.zeros((8, 1))

    # 初始值计算
    macc = np.mean(data[0:20, 3:6], axis=0)
    pitch0 = np.arctan2(macc[1], np.sqrt(macc[0]**2 + macc[2]**2)) / np.pi * 180
    roll0 = np.arctan2(-macc[0], macc[2]) / np.pi * 180
    yaw0 = 0
    
    atti1 = setoula(yaw0, pitch0, roll0)
    speed1 = np.array([[0], [0]])
    dTins = 0.005
    dataA = np.zeros((L, 30))
    biasgyro = np.zeros((3, 1))

    # 导航循环
    for k in range(L):
        gyro = data[k, 0:3].reshape(-1, 1)  # 角速度
        accb = data[k, 3:6].reshape(-1, 1)  # 加速度
        
        gyro1 = gyro + biasgyro
        atti1 = qupdate(atti1, gyro1 * dTins)  # 更新姿态（四元数）
        Cbn = cbn(atti1)  # 计算方向余弦矩阵
        accn = Cbn @ accb  # 将加速度转换到导航系
        speed1 = speed1 + dTins * accn[0:2]  # 更新速度（仅水平分量）
        
        Fk = getfk(atti1, accn)  # 获取状态转移矩阵
        IFk = np.eye(8) + Fk * dTins
        Phi1 = IFk @ Phi1
        
        Q1 = Q1 + Q0 * dTins
        
        if True:  # 卡尔曼滤波更新
            Z1 = speed1
            X1, Pk1 = kal(Z1, H, Pk1, Q1, R, Phi1)  # 假设 kal 函数已定义
            X1[4] = 0  # 注意：MATLAB 的 X1(5) 在 Python 中是 X1[4]（0-based indexing）
            Phi1 = np.eye(8)
            Q1 = np.zeros((8, 8))
            speed1 = speed1 - X1[0:2]
            atti1 = qupdate(atti1, (Cbn.T @ X1[2:5]).flatten())  # 修正姿态
            biasgyro = biasgyro - X1[5:8]
        
        # 数据保存
        dataA[k, 0:9] = np.concatenate((getoula(atti1), atti1, speed1.flatten()))  # 假设 getoula 已定义
        dataA[k, 9:11] = Z1.flatten()
        dataA[k, 11:19] = X1.flatten()
        
    # 保存四元数数据到文件
    np.savetxt('qf_1.txt', dataA[:, 3:7], fmt='%10.6f')

def main():

    raw_data = pd.read_csv(r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\test\0417_0\1\tx\imu_output.csv").values
    ol = inst_main(raw_data)
    

if __name__=="__main__":
    main()