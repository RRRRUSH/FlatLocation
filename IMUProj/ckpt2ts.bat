@echo off
setlocal EnableDelayedExpansion


set "MODEL_ARCH=TLIO"
set "DATASET=resnet(tlioraw)_tlio"

set "PYTHON_CMD=D:\Anaconda\envs\tlio\python.exe -u"

echo Converting...
%PYTHON_CMD% "E:\HaozhanLi\Project\FlatLoc\tlio\src\convert_model_to_torchscript.py" ^
--model_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\%MODEL_ARCH%\%DATASET%\checkpoint_best.pt" ^
--model_param_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\%MODEL_ARCH%\%DATASET%\parameters.json" ^
--out_dir "E:\HaozhanLi\Project\FlatLoc\tlio\models\%MODEL_ARCH%\%DATASET%"
