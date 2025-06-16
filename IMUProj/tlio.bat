@echo off
setlocal EnableDelayedExpansion

:: 子目录名
set "DIR_NAME=test\0425_2"

:: 输入输出根目录
set "INPUT_DIR=E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\input\%DIR_NAME%"
set "OUTPUT_DIR=E:\HaozhanLi\Project\FlatLoc\IMUProj\data\predict\output\tlio\%DIR_NAME%"

:: 输出参数以确认
echo Input directory: %INPUT_DIR%
echo Output directory: %OUTPUT_DIR%

set "PYTHON_CMD=D:\Anaconda\envs\tlio\python.exe -u"

:: TLIO 预测
:: ronin_ridi_imunet_tlio_LLIO512
@REM --model_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\model_torchscript.pt" ^
@REM --model_param_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\parameters.json" ^

:: 开源数据集+VRBody
@REM --model_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\open_body\model_torchscript.pt" ^
@REM --model_param_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\open_body\parameters.json" ^

@REM echo Predicting...
@REM %PYTHON_CMD% "E:\HaozhanLi\Project\FlatLoc\tlio\src\main_filter.py" ^
@REM --root_dir "%INPUT_DIR%" ^
@REM --model_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\model_torchscript.pt" ^
@REM --model_param_path "E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\parameters.json" ^
@REM --out_dir "%OUTPUT_DIR%" ^
@REM --erase_old_log --save_as_npy --no-calib

:: 绘图
echo Plotting...
%PYTHON_CMD% "E:\HaozhanLi\Project\FlatLoc\tlio\src\draw_ekf_traj.py" ^
--state_path "%OUTPUT_DIR%" ^