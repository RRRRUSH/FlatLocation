# 本项目可以训练类RONIN模型，IMUNET模型

# 修改说明

## 数据集添加

1. 首先在[Sequence](Sequence)文件夹中添加类似[ronin_sequence.py](Sequence/ronin_sequence.py)的数据加载序列文件，主要是load方法
2. 然后再[config.json](GetData/config.json)文件中添加数据集配置，以及在[Datasets](Datasets)添加数据列表
3. 最后可以运行[get_dataset.py](GetData/get_dataset.py)文件查看是否可以读取数据
4. 补充说明：load方法中，最后构造的特征和标签的单位是固定的，特征中gyro单位是rad/s，acc单位是m/s^2,标签的单位是m/s

## 模型添加

1. 首先在[model_library](model_library)
   中添加模型架构，输入参数是固定的，具体可以参考[model_library/RONIN.py](model_library/RONIN.py)
   以及[model_factory.py](model_factory.py)
2. 然后在[model](config/model)
   中添加相应的配置文件，模型配置文件格式参考[ronin.yaml](config/model/ronin.yaml)[config/model/RONIN.json](config/model/RONIN.json)
3. 最后在[model_factory.py](model_factory.py)中的各个函数中添加逻辑

# 模型训练

## 参数说明

```
--dataset_config ： 数据集配置文件路径
--config ： 模型配置文件路径
--out_dir ： 模型输出路径
--arch ： 模型架构
--seed ： 随机种子，可控制随机性
```

## IMUNET train
```shell script
python train.py --dataset_config E:\HaozhanLi\Project\FlatLoc\ronin\GetData\config.json --config E:\HaozhanLi\Project\FlatLoc\ronin\config --out_dir E:\HaozhanLi\Project\FlatLoc\ronin\output\IMUNET\test --arch IMUNET --seed 1
```
## RONIN train
```shell script
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_config /home/user/HaozhanLi/train_model_use_ronin_method/GetData/config.json --config /home/user/HaozhanLi/train_model_use_ronin_method/config --out_dir /home/user/HaozhanLi/tlio/models/ronin/open_body --arch RONIN --seed 1
```