#!/usr/bin/env python3

import os
import queue
import socket
import sys
import threading
import time
import numpy as np
import pandas as pd
import psutil
import serial
import glob
import datetime
from matplotlib import pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from src.tracker.imu_tracker import ImuTracker
from src.tracker.scekf import propagate_rvt_and_jac, propagate_covariance
from src.utils.dotdict import dotdict
from src.utils.logging import logging
from src.hipnuc.hipnuc_serial_parser import hipnuc_parser

GRAVITY = 9.80665
R2D = 57.29577951308232


class ImuTrackerRunner:
    """
    This class is responsible for going through a dataset, feed imu tracker and log its result
    """

    def __init__(self, args):

        # imu_calib = ImuCalib.from_offline_calib()
        self.ser = None
        self.port = args.port
        self.baudrate = args.baudrate
        self.wifi = args.wifi
        self.t_us = 5
        self.stop_event = 1
        filter_tuning = dotdict(
            {
                "g_norm": args.g_norm,
                "sigma_na": args.sigma_na,
                "sigma_ng": args.sigma_ng,
                "ita_ba": args.ita_ba,
                "ita_bg": args.ita_bg,
                "init_attitude_sigma": args.init_attitude_sigma,  # rad
                "init_yaw_sigma": args.init_yaw_sigma,  # rad
                "init_vel_sigma": args.init_vel_sigma,  # m/s
                "init_pos_sigma": args.init_pos_sigma,  # m
                "init_bg_sigma": args.init_bg_sigma,  # rad/s
                "init_ba_sigma": args.init_ba_sigma,  # m/s^2
                "meascov_scale": args.meascov_scale,
                "use_const_cov": args.use_const_cov,
                "const_cov_val_x": args.const_cov_val_x,  # sigma^2
                "const_cov_val_y": args.const_cov_val_y,  # sigma^2
                "const_cov_val_z": args.const_cov_val_z,  # sigma^2
                "add_sim_meas_noise": args.add_sim_meas_noise,
                "sim_meas_cov_val": args.sim_meas_cov_val,
                "sim_meas_cov_val_z": args.sim_meas_cov_val_z,
                "mahalanobis_fail_scale": args.mahalanobis_fail_scale,
            }
        )
        self.warm_ekf()
        # ImuTracker object
        self.tracker = ImuTracker(
            model_path=args.model_path,
            model_param_path=args.model_param_path,
            update_freq=args.update_freq,
            filter_tuning_cfg=filter_tuning,
            imu_calib=None,
            force_cpu=args.force_cpu,
        )

    def warm_ekf(self):
        ts = time.time()
        A_aug = np.ascontiguousarray(np.eye(15))
        B_aug = np.zeros((15, 6))
        dt_us = 5000
        Sigma = np.eye(15)
        W = np.eye(6)
        Q = np.eye(6)
        Sigma_kp1 = propagate_covariance(
            A_aug, B_aug, dt_us * 1e-6, Sigma, W, Q
        )
        R_k = np.eye(3)
        v_k = np.zeros((3, 1))
        p_k = np.zeros((3, 1))
        b_gk = np.zeros((3, 1))
        b_ak = np.zeros((3, 1))
        gyr = np.zeros((3, 1))
        acc = np.ones((3, 1))
        g = 9.8
        dt_us = 5000
        R_kp1, v_kp1, p_kp1, Akp1 = propagate_rvt_and_jac(
            R_k, v_k, p_k, b_gk, b_ak, gyr, acc, g, dt_us * 1e-6
        )
        logging.info("warm time: %f", time.time() - ts)

    last_time = 0
    lose_count = 0
    def async_read_from_serial(self, queue, sv_queue):
        try:
            serial_parser = hipnuc_parser()
            self.ser = serial.Serial(self.port, int(self.baudrate))
            while self.stop_event:
                data = self.ser.read(82)
                hipnuc_frames = serial_parser.parse(data)
                if hipnuc_frames:
                    for i, frame in enumerate(hipnuc_frames):
                        if self.last_time != 0 and frame.system_time_ms - self.last_time > 5:
                            self.lose_count += (frame.system_time_ms - self.last_time) / 5 - 1
                            print(f"已丢失{self.lose_count}条数据,丢包率: {self.lose_count / (self.t_us / 5 + self.lose_count) * 100}%")
                        self.last_time = frame.system_time_ms

                        self.t_us = self.t_us + 5
                        acc = np.array(frame.acc).reshape((3, 1)) * GRAVITY
                        gyr = np.array(frame.gyr).reshape((3, 1)) / R2D
                        q = np.array(frame.quat).reshape((4, 1))
                        queue.put((self.t_us, acc, gyr))
                        sv_queue.put((self.t_us, acc, gyr, q))

        except KeyboardInterrupt:
            print("Program interrupted by user src async_read_from_serial")

        except (serial.SerialException, PermissionError) as e:
            print(f"Error: {e} src async_read_from_serial")
            print(f"To run this script with superuser privileges, use the 'sudo' command:")
            print(f"Example: sudo python main.py read --port {self.port} --baudrate {self.baudrate}")
            sys.exit(1)

    def read_from_wifi(self,queue, sv_queue):
        try:

            tcpPort = 6666
            tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
            tcp_server.bind(('0.0.0.0', tcpPort))
            tcp_server.listen(4)
            serial_parser = hipnuc_parser()

            while self.stop_event:
                # 建立客户端连接
                client_socket, client_address = tcp_server.accept()
                print(f"客户端 {client_address} 已连接")

                while self.stop_event:
                    try:
                        data = client_socket.recv(82)
                        if data:
                            hipnuc_frames = serial_parser.parse(data)
                            if hipnuc_frames:
                                for i, frame in enumerate(hipnuc_frames):
                                    if self.last_time != 0 and frame.system_time_ms - self.last_time > 5:
                                        self.lose_count += (frame.system_time_ms - self.last_time) / 5 - 1
                                        print(f"已丢失{self.lose_count}条数据,丢包率: {self.lose_count / (self.t_us / 5 + self.lose_count) * 100}%")
                                    self.last_time = frame.system_time_ms

                                    self.t_us = self.t_us + 5
                                    acc = np.array(frame.acc).reshape((3, 1)) * GRAVITY
                                    gyr = np.array(frame.gyr).reshape((3, 1)) / R2D
                                    q = np.array(frame.quat).reshape((4, 1))
                                    queue.put((self.t_us, acc, gyr))
                                    sv_queue.put((self.t_us, acc, gyr, q))
                        else:
                            print("客户端关闭连接")
                            break
                    except socket.timeout:
                        print("当前无数据")
                    except Exception as e:
                        print(f"发生错误: {e}")
                        break
                client_socket.close()
                print(f"客户端 {client_address} 连接已关闭")
        except KeyboardInterrupt:
            print("Program interrupted by user src async_read_from_serial")

    def async_write_to_file(self, sv_queue):
        root_path = r"/home/jetson/tlio_data"
        paths = glob.glob(root_path + "//" + r"*.csv")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data_{timestamp}.csv"
        try:
            with open(os.path.join(root_path, filename), 'w') as f:
                while self.stop_event:
                    us, acc, gyro, quat = sv_queue.get()
                    acc_flat = acc.flatten()  # [acc_x, acc_y, acc_z]
                    gyr_flat = gyro.flatten()  # [gyr_x, gyr_y, gyr_z]
                    q_flat = quat.flatten()      # [q_w, q_x, q_y, q_z]
                
                    data_row = [str(us)] + [str(x) for x in acc_flat] + [str(x) for x in gyr_flat] + [str(x) for x in q_flat]
                    line = ','.join(data_row) + '\n'

                    f.write(line)
        except Exception as e:
            print(f"SAVE ERROR {e}")
            exit()

    def async_process_data(self, queue, plot_queue):

        try:
            idx = 0
            while self.stop_event:
                t_us, acc_raw, gyr_raw = queue.get()
                idx += 1
                p = self.get_pos(acc_raw, gyr_raw, int(t_us * 1e03))
                if p is not None:
                    plot_queue.put((p[0], p[1]))

        except KeyboardInterrupt:
            print("Program interrupted by user src async_process_data")

    def async_plot_data(self, data_queue, plot_queue):
        # 创建PyQtGraph绘图部件
        self.win = pg.GraphicsLayoutWidget(show=True, size=(600, 600))
        self.win.setWindowTitle('Real-time Position Plot')

        # 创建绘图区域
        self.plot = self.win.addPlot(title="Position Tracking")
        self.plot.setLabel('left', 'Position Y (m)')
        self.plot.setLabel('bottom', 'Position X (m)')
        self.plot.setAspectLocked(True)  # 保持纵横比1:1
        self.plot.showGrid(x=True, y=True)

        # 禁用鼠标交互（禁止缩放和平移）
        self.plot.setMouseEnabled(x=False, y=False)

        # 设置最小比例
        min_scale = 15  # 定义最小比例值
        self.plot.setRange(xRange=(-min_scale, min_scale), yRange=(-min_scale, min_scale), padding=0)
        self.plot.disableAutoRange()  # 禁用自动调整范围

        # 初始化绘图曲线
        self.curve = self.plot.plot(pen='y', name='Position')

        # 初始化数据存储
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.lim = 10

        # 创建文本项显示队列信息（PyQtGraph使用Qt的文本项）
        self.text_item = pg.TextItem("", anchor=(0, 1), color='w')
        self.plot.addItem(self.text_item)

        # 创建定时器用于定期更新（替代原来的循环）
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.update_plot(data_queue, plot_queue))
        self.timer.start(250)  # 约33fps

        # 启动Qt事件循环（如果不在Qt主线程中需要单独处理）
        QtWidgets.QApplication.instance().exec_()

    def update_plot(self, data_queue, plot_queue):
        try:
            # 从队列获取最新数据（非阻塞方式）
            temp_data_x = None
            temp_data_y = None

            while not plot_queue.empty():
                x, y = plot_queue.get_nowait()
                temp_data_x = x
                temp_data_y = y

            if temp_data_x is not None and temp_data_y is not None:
                self.data_x = np.append(self.data_x, temp_data_x)
                self.data_y = np.append(self.data_y, temp_data_y)
                # 更新曲线数据
                self.curve.setData(self.data_x, self.data_y)

                # 动态调整坐标轴范围
                current_max = max(np.max(np.abs(self.data_x)), np.max(np.abs(self.data_y)))
                if current_max > self.lim:
                    self.lim = current_max + 1
                    self.plot.setXRange(-self.lim, self.lim)
                    self.plot.setYRange(-self.lim, self.lim)

        except KeyboardInterrupt:
            self.timer.stop()
            self.win.close()
            print("Program interrupted by user")
    def async_run_tracker(self):
        def initialize_at_first_update(this):
            logging.info(f"Re-initialize filter at first update")
            ts = time.time()
            self.reset_filter_state_pv()
            logging.info(f"Re-initialize using time {time.time() - ts}")

        self.tracker.callback_first_update = initialize_at_first_update
        data_queue = queue.Queue()
        plot_queue = queue.Queue()
        save_queue = queue.Queue()

        if self.wifi:
            producer_thread = threading.Thread(target=self.read_from_wifi, args=(data_queue, save_queue), name='serial reader')
        else:
            producer_thread = threading.Thread(target=self.async_read_from_serial, args=(data_queue, save_queue), name='serial reader')
        consumer_thread = threading.Thread(target=self.async_process_data, args=(data_queue, plot_queue), name='model inferencer')
        save_thread = threading.Thread(target=self.async_write_to_file, args=(save_queue, ), name='save to csv')

        producer_thread.daemon = True
        consumer_thread.daemon = True
        save_thread.daemon = True

        producer_thread.start()
        consumer_thread.start()
        save_thread.start()
        try:
            self.async_plot_data(data_queue, plot_queue)
        except KeyboardInterrupt:
            print("\n程序终止。")
        finally:
            self.stop_event = 0
            producer_thread.join()
            consumer_thread.join()
            save_thread.join()
            print("所有线程已停止")

    def get_pos(self, acc_raw, gyr_raw, t_us):

        self.tracker.on_imu_measurement(t_us, gyr_raw, acc_raw)
        _, _, p, _, _ = self.tracker.filter.get_evolving_state()
        return p.T.tolist()[0]


    def reset_filter_state_pv(self):
        """ Reset filter states p and v with zeros """
        state = self.tracker.filter.state
        ps = []
        for i in state.si_timestamps_us:
            ps.append(np.zeros((3, 1)))
        p = np.zeros((3, 1))
        v = np.zeros((3, 1))
        self.tracker.filter.reset_state_and_covariance(
            state.si_Rs, ps, state.s_R, v, p, state.s_ba, state.s_bg
        )

    def get_thread_cpu_usage(self):
        current_process = psutil.Process(os.getpid())
        threads = current_process.threads()
        cpu_info = {}

        cpu_cores = psutil.cpu_count(logical=True)

        for thread in threads:
            thread_id = thread.id  # 操作系统级别的线程 ID
            thread_cpu_percent = current_process.cpu_percent(interval=0.1)
            normalized_cpu_percent = thread_cpu_percent / cpu_cores
            cpu_num = psutil.Process(os.getpid()).cpu_num()
            cpu_info[thread_id] = {
                "cpu_num": cpu_num,  # 线程运行的 CPU 核心编号
                "cpu_percent": normalized_cpu_percent  # 线程在单个核心上的 CPU 占用率
            }

        for thread in threading.enumerate():
            thread_id = thread.native_id  # 获取操作系统级别的线程 ID
            thread_name = thread.name
            thread_cpu = cpu_info.get(thread_id, {"cpu_num": -1, "cpu_percent": 0})
            print(
                f"线程名称: {thread_name}, 线程ID: {thread_id}, CPU核心: {thread_cpu['cpu_num']}, CPU占用率: {thread_cpu['cpu_percent']}%")
        return cpu_info

    def get_cpu_using(self):
        cpu_percentages = psutil.cpu_percent(percpu=True, interval=0.1)
        for i, percent in enumerate(cpu_percentages):
            if percent > 0:
                print(f"CPU 核心 {i} 使用率: {percent}%")
        total_cpu = sum(cpu_percentages)
        print(f"总 CPU 占用率: {total_cpu}%")
        print('--------')

    def test(self):
        data = pd.read_csv(r"C:\Users\29586\Downloads\data_20250224_161828.csv", header=None,delimiter=',').values
        traj = []
        for i in data:
            t_us, acc_raw, gyr_raw = i[0], np.array(i[1:4]).reshape((3, 1)), np.array(i[4:7]).reshape((3, 1))
            p = self.get_pos(acc_raw, gyr_raw, int(t_us * 1e03))
            traj.append([p[0], p[1]])
        pd.DataFrame(traj).to_csv(r"C:\Users\29586\Downloads\data_20250224_161828_traj.csv", header=None, index=None)
        print('推理结束')
        exit()