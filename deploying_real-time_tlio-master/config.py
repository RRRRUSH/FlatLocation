# -*- coding: utf-8 -*-            
# @Time: 2025/2/13 16:17
# @Author: Zhou Xiang
# @FileName: config.py
# @Software: PyCharm
import yaml

from pydantic import BaseModel
import yaml


class IOConfig(BaseModel):
    port: str = "COM4"
    baudrate: int = 921600
    wifi: bool = False
    model_path: str = "resource/model/model_torchscript.pt"
    model_param_path: str = "resource/model/parameters.json"

    class Config:
        protected_namespaces = ()


class FilterConfig(BaseModel):
    update_freq: float = 20.0
    sigma_na: float = 0.03162277660168379
    sigma_ng: float = 0.01
    ita_ba: float = 1e-4
    ita_bg: float = 1e-6
    init_attitude_sigma: float = 0.017453292519943295
    init_yaw_sigma: float = 0.0017453292519943295
    init_vel_sigma: float = 1.0
    init_pos_sigma: float = 0.001
    init_bg_sigma: float = 0.0001
    init_ba_sigma: float = 0.02
    g_norm: float = 9.80
    meascov_scale: float = 10.0
    initialize_with_vio: bool = False
    initialize_with_offline_calib: bool = False
    calib: bool = False
    mahalanobis_fail_scale: float = 0


class DebugConfig(BaseModel):
    use_const_cov: bool = False
    const_cov_val_x: float = 0.01
    const_cov_val_y: float = 0.01
    const_cov_val_z: float = 0.01
    add_sim_meas_noise: bool = False
    sim_meas_cov_val: float = 0.0001
    sim_meas_cov_val_z: float = 0.0001
    force_cpu: bool = False


class Config(BaseModel):
    io: IOConfig
    filter: FilterConfig
    debug: DebugConfig

    def __getattr__(self, name):
        for component in [self.io, self.filter, self.debug]:
            if hasattr(component, name):
                return getattr(component, name)
        return None


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


if __name__ == '__main__':
    config = load_config("config.yaml")
    print("Port:", config.port)
    print("Update Frequency:", config.update_freq)
    print("Use Const Cov:", config.use_const_cov)
