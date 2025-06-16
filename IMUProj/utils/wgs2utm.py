import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Transformer

def wgs84_to_utm(lon, lat):
    # 确定UTM区域号
    utm_zone = int((lon + 180) / 6) + 1
    
    # 确定南北半球
    hemisphere = 'N' if lat >= 0 else 'S'
    
    # 创建WGS84到UTM的转换器
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{utm_zone}" if hemisphere == 'N' else f"EPSG:327{utm_zone}", always_xy=True)
    
    # 进行坐标转换
    utm_x, utm_y = transformer.transform(lon, lat)
    
    return utm_x, utm_y, utm_zone, hemisphere

if __name__=="__main__":
    path = r"E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\raw\rtk_imu\2025-04-10_192659\pred_rtk.txt"

    data = pd.read_csv(path, header=None, sep=",").values

    for i in range(len(data)):
        lat, lon = data[i, 1], data[i, 2]
        x, y, zone, hemisphere = wgs84_to_utm(lon, lat)
        data[i, 1], data[i, 2] = x, y
        
    data[:, 1], data[:, 2] = data[:, 1] - data[0, 1], data[:, 2] - data[0, 2]
    pd.DataFrame(data).to_csv(
        path.split(".")[0]+"_utm.txt", 
        header=False, sep=","
    )
    
