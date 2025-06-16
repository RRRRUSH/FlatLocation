import os
import yaml

from model_library.imunet import IMUNet
from model_library.ronin import RONINResNet1D
from model_library.model_resnet1d import *

def get_arch(args):
    model_params = get_model_params(args)
    if args.arch == 'RONIN':
        model = RONINResNet1D(model_params)
    elif args.arch == 'IMUNET':
        model = IMUNet(model_params)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return model

def get_model_params(args):
    if args.arch == 'RONIN':
        with open(os.path.join(args.config, "model/ronin.yaml"), 'r') as file:
            model_params = yaml.safe_load(file)['model_params']
    elif args.arch == 'IMUNET':
        with open(os.path.join(args.config, "model/imunet.yaml"), 'r') as file:
            model_params = yaml.safe_load(file)['model_params']
    return model_params

def mseloss(pred, targ):
    return (pred - targ).pow(2)

def get_loss(pred, targ, arch):
    if arch == 'RONIN':
        loss = mseloss(pred, targ)
    elif arch == "IMUNET":
        loss = mseloss(pred, targ)
    else:
        raise ValueError('Invalid loss: ', arch)
    return loss

def get_model(model_path, dim, ronin_instance):
    input_channel  = 6
    output_channel = dim
    fc_config = {'fc_dim': 512, 'in_dim': 200// 32 + 1, 'dropout': 0.5, 'trans_planes': 128}
    if ronin_instance == 'resnet18':
        model = ResNet1D(input_channel, output_channel, BasicBlock1D, [2, 2, 2, 2],
                               base_plane=64, output_block=FCOutputModule, kernel_size=3, **fc_config)
    elif ronin_instance == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        fc_config['fc_dim'] = 1024
        model = ResNet1D(input_channel, output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fc_config)
    elif ronin_instance == 'resnet101':
        fc_config['fc_dim'] = 1024
        model = ResNet1D(input_channel, output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **fc_config)
    elif ronin_instance == 'IMUNet':
        fc_config['num_inputs'] = 6
        fc_config['num_outputs'] = 2
        model = IMUNet(fc_config)

    else:
        raise ValueError('Invalid architecture: ', ronin_instance)
    model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
    return model