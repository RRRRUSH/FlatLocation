import time

from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans

from server.utils.coordinate_utils import transformer_wgs84_to_utm, transformer_utm_to_wgs84, \
    get_target_pos_by_ang, rotate_traj_matrix, get_final_heading_angle, calculate_arrow_endpoint


class ShapeFit:
    last_optimal_adjust_args = None
    origin_xy = None
    origin_utm_xy = None
    origin_lonlat = None
    ang_matrix = None
    time_pos_index_map = np.array([])
    utm_shape = np.array([])
    maybe_points = np.array([])


    def new_xys_to_lonlat(self, xys):
        if self.origin_lonlat is None:
            return
        xys[:] = xys[:] - self.origin_xy[:]
        if self.last_optimal_adjust_args is not None:
            rotated_xy = np.dot(self.ang_matrix, xys.T).T
            xys[:, 0] = rotated_xy[:, 0] * self.last_optimal_adjust_args[1] + self.last_optimal_adjust_args[2]
            xys[:, 1] = rotated_xy[:, 1] * self.last_optimal_adjust_args[1] + self.last_optimal_adjust_args[3]
        lonlat = np.array(self.xys_to_lonlat(xys))

        # lonlats_array = np.array(lonlat)
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

        return lonlat

    def xys_to_lonlat(self, xys):
        lonlats = []
        for xy in xys:
            # 计算新的 UTM 坐标
            new_utm_x = self.origin_utm_xy[0] + xy[0]
            new_utm_y = self.origin_utm_xy[1] + xy[1]
            # 将新的 UTM 坐标转换回 WGS84 坐标
            lon, lat = transformer_utm_to_wgs84.transform(new_utm_x, new_utm_y)
            lonlats.append([lon, lat])
        return lonlats

    def fit(self, shape1, shape2):
        if shape2.shape[0] == 0:
            return shape1
        shape1_ts = shape1[:,0]
        shape1 = shape1[:, [1,2]]

        def error_function(args):
            a = args[0]
            s = args[1]
            x = args[2]
            y = args[3]
            a_m = rotate_traj_matrix(shape1[1], get_target_pos_by_ang(a))
            map_pos = []
            for i in range(self.time_pos_index_map.shape[0]):
                map_pos.append(shape1[self.time_pos_index_map[i, 0]])
            map_pos = np.array(map_pos)
            rotated_traj_pos = np.dot(a_m, map_pos.T).T
            rotated_traj_pos[:,0] = rotated_traj_pos[:,0] * s + self.origin_utm_xy[0]+x
            rotated_traj_pos[:,1] = rotated_traj_pos[:,1] * s + self.origin_utm_xy[1]+y
            errors = []
            err_pos = []
            for i in range(self.time_pos_index_map.shape[0]):
                d = np.linalg.norm(rotated_traj_pos[i] - self.utm_shape[i,[0,1]])
                d2 = d - self.utm_shape[i][2]
                errors.append(d2)
                err_pos.append(self.utm_shape[i,[4,3,0,1,2]])

            sorted_errors, sorted_err_pos = zip(*sorted(zip(errors, err_pos), key=lambda pair: pair[0]))
            errors = list(sorted_errors)
            err_pos = list(sorted_err_pos)
            # length = len(errors)
            # for i in range(5,length,5):
            #     errors = errors[1:-1]
            #     err_pos = err_pos[1:-1]
            self.maybe_points = np.array(err_pos)
            return sum(errors) if errors else 0

        if self.time_pos_index_map.shape[0] > 0:
            last_pos_index = self.time_pos_index_map[-1,0] + 1
            last_uwb_index = self.time_pos_index_map[-1, 1] + 1
        else:
            last_pos_index = 0
            last_uwb_index = 0

        temp_uwb_pos_index_map = []
        for i in range(last_pos_index,shape1_ts.shape[0]):
            if i >= shape1_ts.shape[0] - 1:
                break
            for j in range(last_uwb_index,shape2.shape[0]):
                if shape1_ts[i] <= shape2[j, 0] <= shape1_ts[i + 1]:
                    temp_uwb_pos_index_map.append([i, j])

        temp_utm_shape = []
        for i in temp_uwb_pos_index_map:
            lonlat = shape2[i[1], [2, 1, 3, 0]]
            utm_x, utm_y = transformer_wgs84_to_utm.transform(lonlat[0], lonlat[1])
            temp_utm_shape.append([utm_x, utm_y, lonlat[2], lonlat[3],i[1]])

        if self.utm_shape.shape[0] > 0:
            if len(temp_utm_shape)>0:
                self.utm_shape = np.vstack((self.utm_shape, np.array(temp_utm_shape)))
        else:
            self.utm_shape = np.array(temp_utm_shape)

        if self.time_pos_index_map.shape[0] > 0:
            if len(temp_uwb_pos_index_map)>0:
                self.time_pos_index_map = np.vstack((self.time_pos_index_map, np.array(temp_uwb_pos_index_map)))
        else:
            self.time_pos_index_map = np.array(temp_uwb_pos_index_map)

        if self.time_pos_index_map.shape[0] >= 1:
            if self.origin_xy is None:
                self.origin_xy = np.array(shape1[self.time_pos_index_map[0, 0]])
                self.origin_lonlat = shape2[self.time_pos_index_map[0, 1], [2, 1]]
                self.origin_utm_xy = self.utm_shape[0]
            shape1[:] = shape1[:] - self.origin_xy[:]

        if self.time_pos_index_map.shape[0] > 1:
            # 使用 differential_evolution 进行全局优化
            st = time.time()
            bounds = [(0, 360), (0, 2), (-1000,1000), (-1000,1000)]
            # if self.last_optimal_adjust_args is None:
            result = differential_evolution(error_function,
                                            bounds,
                                            strategy='best1bin',
                                            maxiter=100,
                                            popsize=10,
                                            tol=0.01,
                                            x0=[0,1,0,0] if self.last_optimal_adjust_args is None else self.last_optimal_adjust_args
                                            )
            print('differential_evolution Elapsed time:',time.time() - st)
            # else:
            #     result = minimize(error_function, np.array(self.last_optimal_adjust_args), method='L-BFGS-B', bounds=bounds)
            #     print('minimize Elapsed time:',time.time() - st)

            # 获取最优调整值
            optimal_angle = result.x[0]
            optimal_scale = result.x[1]
            optimal_o_x = result.x[2]
            optimal_o_y = result.x[3]
            self.last_optimal_adjust_args = [optimal_angle, optimal_scale,optimal_o_x, optimal_o_y]
            print(f"Optimal angle: {optimal_angle}")
            print(f"Optimal scale: {optimal_scale}")
            print(f"Optimal offset x: {optimal_o_x}")
            print(f"Optimal offset y: {optimal_o_y}")

            # 使用最优调整值
            self.ang_matrix = rotate_traj_matrix(shape1[1], get_target_pos_by_ang(optimal_angle))
            shape1 = np.dot(self.ang_matrix, shape1.T).T
            shape1[:,0] = shape1[:,0] * optimal_scale + optimal_o_x
            shape1[:,1] = shape1[:,1] * optimal_scale + optimal_o_y
            return np.array(self.xys_to_lonlat(shape1))
        elif self.time_pos_index_map.shape[0] == 1:
            return np.array(self.xys_to_lonlat(shape1))
        else:
            return shape1


def means_fit(xys, n_clusters=10, max_intracluster_distance=100, min_cluster_size=5):
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=0)
    kmeans.fit(xys)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    # 计算每个簇的大小
    cluster_sizes = np.bincount(labels)

    # 确定大集群
    large_clusters = np.where(cluster_sizes >= min_cluster_size)[0]
    valid_clusters = []

    for i in large_clusters:
        cluster_points = xys[labels == i]
        if len(cluster_points) < 2:
            continue
        intracluster_distances = np.max(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2))
        if intracluster_distances <= max_intracluster_distance:
            valid_clusters.append(i)

    return np.array(cluster_centers[valid_clusters])

