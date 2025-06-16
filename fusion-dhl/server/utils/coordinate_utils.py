import math

import numpy as np
from geographiclib.geodesic import Geodesic
from geopy import Point
from pyproj import Transformer
from geopy.distance import geodesic

def get_target_pos_by_ang(angle_degrees, is_radians = False):
    if is_radians:
        angle_radians = angle_degrees
    else:
        angle_radians = math.radians(angle_degrees)
    x = math.cos(angle_radians)
    y = math.sin(angle_radians)
    return np.array([x, y])

def rotate_traj_matrix(current_pos, target_pos):
    o_pos = np.array([0, 0])
    vec1 = current_pos - o_pos
    vec2 = target_pos - o_pos

    # 检查 vec2 是否为零向量
    if np.allclose(vec2, [0, 0]):
        return np.eye(2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    cos_theta = np.dot(vec1, vec2) / ((norm_vec1 * norm_vec2) + 1e-9)
    sin_theta = np.cross(vec1, vec2) / ((norm_vec1 * norm_vec2) + 1e-9)

    theta = np.arctan2(sin_theta, cos_theta)

    # 计算旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)

    rot_matrix = np.array([
        [c, -s],
        [s, c]
    ])

    return rot_matrix

def get_utm_zone_from_longitude(longitude):
    if not (-180 <= longitude <= 180):
        raise ValueError("经度值必须在 -180 到 180 度之间")

    zone = int((longitude + 180) / 6) + 1
    print(zone)
    return zone

utm_crs = f"+proj=utm +zone={get_utm_zone_from_longitude(117)} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
transformer_wgs84_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
transformer_utm_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

def create_image_coordinate_by_lonlat(lonlat, dpi=2.5):
    # 提取经度和纬度
    lons = [point[0] for point in lonlat]
    lats = [point[1] for point in lonlat]

    # 计算经度和纬度的最小值和最大值
    min_lon = min(lons) - 0.000001
    max_lon = max(lons) + 0.000001
    min_lat = min(lats) - 0.000001
    max_lat = max(lats) + 0.000001

    width = Geodesic.WGS84.Inverse(min_lon, min_lat, min_lon, max_lat)['s12'] * dpi
    height = Geodesic.WGS84.Inverse(min_lon, min_lat, max_lon, min_lat)['s12'] * dpi

    return np.array([
        [0, 0, max_lon, min_lat],
        [0, height, min_lon, min_lat],
        [width, 0, max_lon, max_lat],
        [width, height, min_lon, max_lat]
    ])


def xys_to_lonlat_by_image_coordinate(image_coordinate, xys):
    max_lon = image_coordinate[:,2].max()
    min_lat = image_coordinate[:,3].min()

    lonlats = []

    utm_x, utm_y = transformer_wgs84_to_utm.transform(min_lat, max_lon)

    for x, y in xys:
        new_utm_x = utm_x + x
        new_utm_y = utm_y - y
        lon, lat = transformer_utm_to_wgs84.transform(new_utm_x, new_utm_y)
        lonlats.append((lon, lat))

    return np.array(lonlats)

def get_final_heading_angle(lonlats):
    if len(lonlats) < 2:
        raise ValueError("至少需要两个点来计算航向角")

    # 获取最后两个点的经纬度
    last_point = lonlats[-1,[1,0]]
    second_last_point = lonlats[-2,[1,0]]

    # 计算方位角
    initial_bearing = geodesic(second_last_point, last_point).ELLIPSOID[-1]

    return initial_bearing

def calculate_arrow_endpoint(lonlat, heading_angle, distance):
    """
    根据经纬度、航向角和距离计算箭头的终点
    :param lonlat: 起点的经纬度 (经度, 纬度)
    :param heading_angle: 航向角 (度)
    :param distance: 箭头的长度 (单位: 米)
    :return: 箭头终点的经纬度 (经度, 纬度)
    """
    destination = geodesic(meters=distance).destination(Point(lonlat[1], lonlat[0]), heading_angle)
    return (destination.longitude, destination.latitude)

def join_xys_to_lonlat_by_yaw(xys, lonlat, yaw, scale=1):
    utm_x, utm_y = transformer_wgs84_to_utm.transform(lonlat[0], lonlat[1])
    target = get_target_pos_by_ang(yaw, True)
    ang_matrix = rotate_traj_matrix(xys[0], target)
    xys[:] = xys[:] - xys[0]
    xys = np.dot(ang_matrix, xys.T).T

    lonlats = []
    for i in range(xys.shape[0]):
        x = xys[i, 0] * scale + utm_x
        y = xys[i, 1] * scale + utm_y
        lon, lat = transformer_utm_to_wgs84.transform(x, y)
        lonlats.append((lon, lat))

    # lonlats_array = np.array(lonlats)
    #
    # final_heading_angle = get_final_heading_angle(lonlats_array)
    #
    # # 假设箭头的长度为 100 米
    # arrow_length = 100
    # last_point = lonlats_array[-1]
    # arrow_endpoint = calculate_arrow_endpoint(last_point, final_heading_angle, arrow_length)
    # lonlats_array = lonlats_array[:-1]
    # # 拼接箭头终点到 lonlats_array
    # lonlats_array = np.vstack([lonlats_array, arrow_endpoint])

    return np.array(lonlats)
