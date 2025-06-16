from network.llio.model_twolayer import TwoLayerModel
from network.model_resnet import BasicBlock1D, ResNet1D
from network.model_resnet_seq import ResNetSeq1D
from network.model_tcn import TlioTcn

from utils.logging import logging


def get_model(arch, net_config, input_dim=6, output_dim=3):
    if arch == "resnet":
        network = ResNet1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "resnet_seq":
        network = ResNetSeq1D(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    elif arch == "tcn":
        network = TlioTcn(
            input_dim,
            output_dim,
            [64, 64, 64, 64, 128, 128, 128],
            kernel_size=2,
            dropout=0.2,
            activation="GELU",
        )
    elif arch == "llio512":
        model_para = {
            "input_len": 200,
            "input_channel": 6,
            "patch_len": 25,
            "feature_dim": 512,
            "out_dim": 3,
            "active_func": "GELU",
            "extractor": {  # include: Feature Convert & ResMLP Module in the paper Fig. 3.
                "name": "ResMLP",
                "layer_num": 6,
                "expansion": 2,
                "dropout": 0.2,
            },
            "reg": {  # Regression in the paper Fig.3
                "name": "MeanMLP",
                # "name": "MaxMLP",
                "layer_num": 3,
            }
        }
        network = TwoLayerModel(model_para)
    elif arch == "llio256":
        model_para = {
            "input_len": 200,
            "input_channel": 6,
            "patch_len": 25,
            "feature_dim": 256,
            "out_dim": 3,
            "active_func": "GELU",
            "extractor": {  # include: Feature Convert & ResMLP Module in the paper Fig. 3.
                "name": "ResMLP",
                "layer_num": 6,
                "expansion": 2,
                "dropout": 0.2,
            },
            "reg": {  # Regression in the paper Fig.3
                "name": "MeanMLP",
                # "name": "MaxMLP",
                "layer_num": 3,
            }
        }
        network = TwoLayerModel(model_para)
    elif arch == "llio128":
        model_para = {
            "input_len": 200,
            "input_channel": 6,
            "patch_len": 25,
            "feature_dim": 128,
            "out_dim": 3,
            "active_func": "GELU",
            "extractor": {  # include: Feature Convert & ResMLP Module in the paper Fig. 3.
                "name": "ResMLP",
                "layer_num": 6,
                "expansion": 2,
                "dropout": 0.2,
            },
            "reg": {  # Regression in the paper Fig.3
                "name": "MeanMLP",
                # "name": "MaxMLP",
                "layer_num": 3,
            }
        }
        network = TwoLayerModel(model_para)
    else:
        raise ValueError("Invalid architecture: ", arch)

    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    logging.info(f"Number of params for {arch} model is {num_params}")

    return network
