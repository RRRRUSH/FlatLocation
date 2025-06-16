import numpy as np


def weighted_average_angle(angles_deg, weights):
    """
    计算加权平均角度，处理0°/360°突变问题。

    Args:
        angles_deg (list): 角度列表（单位：度）
        weights (list): 对应权重列表（正数）

    Returns:
        float: 平均角度（0°~360°）
    """
    # 转换为弧度
    angles_rad = np.deg2rad(angles_deg)

    # 检查权重合法性
    if len(weights) != len(angles_deg):
        raise ValueError("角度和权重的长度必须一致！")
    if np.sum(weights) == 0:
        raise ValueError("权重总和不能为零！")

    # 计算加权的sin和cos和
    weighted_sin = np.sum(weights * np.sin(angles_rad))
    weighted_cos = np.sum(weights * np.cos(angles_rad))

    # 计算平均角度
    avg_angle_rad = np.arctan2(weighted_sin, weighted_cos)
    avg_angle_deg = np.rad2deg(avg_angle_rad) % 360

    return avg_angle_deg


# 示例数据（包含0点突变，且带有权重）
angles = [355, 5, 10]
weights = [1, 2, 1]  # 例如：第二个角度权重更高

result = weighted_average_angle(angles, weights)
print(f"加权平均角度：{result:.1f}°")
# 输出：3.8°（因为5°权重更大，结果更接近5°）