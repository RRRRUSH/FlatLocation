本文件说明主要使用的操作

# 模型训练

## 参数说明

```
`--mode`: 运行方式，`train`是训练，`test`是测试 
`--root_dir`: 数据集根路径，训练时文件夹内至少需要需要有train_list.txt和test_list.txt，测试时文件夹内至少需要有test_list.txt
`--out_dir`: 模型保存路径，模型参数保存路径
`--dataset_style`: 数据集风格，`ram`是内存中读取，一般不需要更改
`--arch` : 模型架构，`llio512`是LLIO512，`llio256`是LLIO256，`llio128`是LLIO128，`resnet`是TLIO
`--seed`: 随机种子
```
    
## TLIO train
```shell script
CUDA_VISIBLE_DEVICES=1 python src/main_net.py --mode train --root_dir /data/for_llio --out_dir /home/user/HaozhanLi/tlio/models/tlio/open_body --dataset_style ram --arch resnet --seed 1
```
## LLIO256 train
```shell script
CUDA_VISIBLE_DEVICES=1 python src/main_net.py --mode train --root_dir /data/for_llio --out_dir /home/user/HaozhanLi/tlio/models/tlio/open_body --dataset_style ram --arch llio256 --seed 1
```
## LLIO512 train
```shell script
CUDA_VISIBLE_DEVICES=2 python src/main_net.py --mode train --root_dir /data/for_llio --out_dir /home/user/HaozhanLi/tlio/models/tlio/open_body --dataset_style ram --arch llio512 --seed 1
```

```
python src/main_net.py --mode train --root_dir E:\HaozhanLi\Project\FlatLoc\tlio\data\train\raw\golden-new-format-cc-by-nc-with-imus --out_dir E:\HaozhanLi\Project\FlatLoc\tlio\models\test\open_body --dataset_style ram --arch llio512 --seed 1
```

# 模型转换

## 参数说明

```
`--model_path`: 模型路径
`--model_param_path`: 模型参数路径
`--out_dir`: torchscript模型保存路径，使用EKF时需要
```

## LLIO512
```shell script
python src/convert_model_to_torchscript.py --model_path E:\HaozhanLi\Project\FlatLoc\tlio\models\TLIO\TLIOv2\checkpoints\checkpoint_1.pt --model_param_path E:\HaozhanLi\Project\FlatLoc\tlio\models\TLIO\tlio_with_seed\parameters.json --out_dir E:\HaozhanLi\Project\FlatLoc\tlio\models\TLIO\TLIOv2
```

# EKF推理

## 参数说明

```
`--root_dir`: 推理数据集根路径，文件夹内至少需要有test_list.txt和数据文件夹，数据文件夹名称需填写在test_list.txt中，可以写很多个，数据文件夹中需要有imu_samples_0.csv
结构大致如下： test_list.txt中写入1010，则预测1010文件夹中的数据
---root_dir
    ----test_list.txt
    ----1010
        ----imu_samples_0.csv
    ----1011
        ----imu_samples_0.csv
`--model_path`: 模型路径
`--model_param_path`: 模型参数路径
`--out_dir`: 输出路径
`--erase_old_log`: 是否删除之前的日志
`--save_as_npy`: 是否保存为npy格式，必须，用于后续渲染
`--no-calib`: 是否使用标定的数据
```

### TLIO

```shell script
python3 src/main_filter.py \
--root_dir /home/a/Desktop/git/TLIO/data/our_data \
--model_path /home/a/Desktop/git/TLIO/models/stepdata/model_torchscript.pt \
--model_param_path /home/a/Desktop/git/TLIO/models/stepdata/parameters.json \
--out_dir /home/a/Desktop/git/TLIO/output/model_ronin_ridi_imunet_tlio/output_ours \
--erase_old_log \
--save_as_npy \
--no-calib
```

### LLIO512

```shell script
python src/main_filter.py --root_dir E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\predict\input\model_mag --model_path E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\model_torchscript.pt --model_param_path E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\parameters.json --out_dir E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\predict\output\test --erase_old_log --save_as_npy --no-calib
```

### LLIO256

```shell script
python3 src/main_filter.py \
--root_dir /home/a/Desktop/git/TLIO/data/our_data \
--model_path /home/a/Desktop/git/TLIO/outputs/llio256/model_torchscript.pt \
--model_param_path /home/a/Desktop/git/TLIO/outputs/llio256/parameters.json \
--out_dir /home/a/Desktop/git/TLIO/outputs/llio256/output_ours \
--erase_old_log \
--save_as_npy \
--no-calib
```

# 渲染

## 参数说明

```
`--state_path`: 文件夹路径下需要存有not_vio_state.txt.npy文件，这是滤波生成的，如果没有请查看上一步
```

not_vio_state.txt.npy这个文件的格式请查找[imu_tracker_runner.py](src/tracker/imu_tracker_runner.py)
中add_data_to_be_logged函数

```shell
python E:\HaozhanLi\Project\FlatLoc\tlio\src\draw_ekf_traj.py --state_path E:\HaozhanLi\Project\FlatLoc\IMUCalibration\data\output\tlio\nvidia_sv2file_test
```