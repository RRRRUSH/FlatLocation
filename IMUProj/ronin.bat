@echo off
setlocal EnableDelayedExpansion


set "DIR_NAME=test\0417_0"

set "INPUT_DIR=E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\%DIR_NAME%"
set "OUTPUT_DIR=E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\output\ronin\%DIR_NAME%"

:: 输出参数以确认
echo Input directory: %INPUT_DIR%
echo Output directory: %OUTPUT_DIR%

set "PYTHON_CMD=D:\Anaconda\envs\RoNIN\python.exe -u"

:: RoNIN 预测
:: ronin_ridi_imunet_tlio_oxiod_RONIN
@REM --model_path "E:\HaozhanLi\Project\FlatLoc\ronin\models\RONIN\ronin_ridi_imunet_tlio\2d_base.pt" ^

echo Predicting...
%PYTHON_CMD% "E:\HaozhanLi\Project\FlatLoc\ronin\predict.py" ^
--model_path "E:\HaozhanLi\Project\FlatLoc\ronin\models\RONIN\ronin_ridi_imunet_tlio\2d_base.pt" ^
--data_dir "%INPUT_DIR%" ^
--out_dir "%OUTPUT_DIR%" ^
--arch "resnet18"
