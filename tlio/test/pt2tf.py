import os
import argparse
import numpy as np
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from pathlib import Path
import time
import json


def load_torchscript_model(model_path):
    """加载TorchScript模型"""
    print(f"正在加载TorchScript模型: {model_path}")
    model = torch.jit.load(model_path)
    model.eval()
    return model


def load_model_parameters(param_path):
    """加载模型参数配置文件"""
    print(f"正在加载模型参数配置: {param_path}")
    with open(param_path, 'r') as f:
        params = json.load(f)
    return params


def get_input_shape_from_params(params):
    """从参数配置中获取输入形状"""
    # 根据TLIO项目的参数格式提取输入形状
    # 注意：这里的逻辑可能需要根据实际的parameters.json格式进行调整
    batch_size = 1  # 默认批次大小
    
    # 通常TLIO模型使用IMU数据，有6个通道（3轴加速度+3轴陀螺仪）
    channels = 6
    
    # 计算序列长度
    if "imu_freq" in params and "past_time" in params and "window_time" in params:
        imu_freq = params["imu_freq"]
        past_time = params["past_time"]
        window_time = params["window_time"]
        
        # 计算输入序列长度
        past_data_size = int(past_time * imu_freq)
        disp_window_size = int(window_time * imu_freq)
        sequence_length = past_data_size + disp_window_size
        
        print(f"从参数中计算得到输入序列长度: {sequence_length} (past_data_size={past_data_size}, disp_window_size={disp_window_size})")
    else:
        # 如果参数中没有相关信息，使用默认值
        sequence_length = 1000
        print(f"未找到序列长度相关参数，使用默认值: {sequence_length}")
    
    return (batch_size, channels, sequence_length)


def convert_to_onnx(model, input_shape, onnx_path):
    """将TorchScript模型转换为ONNX格式"""
    print(f"正在转换为ONNX格式，输出路径: {onnx_path}")
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")
    return onnx_path


def convert_onnx_to_tf(onnx_path, tf_path):
    """将ONNX模型转换为TensorFlow SavedModel格式"""
    print(f"正在转换ONNX到TensorFlow，输出路径: {tf_path}")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)
    return tf_path


def convert_tf_to_tflite(tf_path, tflite_path, quantize=False):
    """将TensorFlow模型转换为TFLite格式"""
    print(f"正在转换为TFLite格式，输出路径: {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    if quantize:
        print("启用量化优化...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 可以添加更多量化配置，如:
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.representative_dataset = representative_dataset_gen
    
    # 尝试启用实验性转换器以提高精度
    converter.experimental_new_converter = True
    
    # 设置浮点精度
    converter.target_spec.supported_types = [tf.float32]
    
    # 转换模型
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_path


def generate_sample_data(input_shape, num_samples=10):
    """生成样本数据用于精度比较"""
    print(f"生成{num_samples}个样本数据进行精度比较")
    samples = []
    for _ in range(num_samples):
        # 使用正态分布生成更真实的IMU数据
        sample = torch.randn(*input_shape)
        samples.append(sample)
    return samples


def run_torchscript_inference(model, samples):
    """使用TorchScript模型进行推理"""
    print("运行TorchScript模型推理...")
    results = []
    start_time = time.time()
    
    for sample in samples:
        with torch.no_grad():
            output = model(sample)
            results.append(output.cpu().numpy())
    
    end_time = time.time()
    print(f"TorchScript推理耗时: {end_time - start_time:.4f}秒")
    return results


def run_tflite_inference(tflite_path, samples):
    """使用TFLite模型进行推理"""
    print("运行TFLite模型推理...")
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 获取输入和输出张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    results = []
    start_time = time.time()
    
    for sample in samples:
        # 准备输入数据
        input_data = sample.numpy()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # 运行推理
        interpreter.invoke()
        
        # 获取输出
        output = interpreter.get_tensor(output_details[0]['index'])
        results.append(output)
    
    end_time = time.time()
    print(f"TFLite推理耗时: {end_time - start_time:.4f}秒")
    return results


def compare_results(torch_results, tflite_results):
    """比较两种模型的输出结果"""
    print("\n精度比较结果:")
    print("-" * 50)
    
    total_samples = len(torch_results)
    total_mse = 0
    total_mae = 0
    max_diff = 0
    
    for i, (torch_out, tflite_out) in enumerate(zip(torch_results, tflite_results)):
        # 计算均方误差(MSE)
        mse = np.mean((torch_out - tflite_out) ** 2)
        # 计算平均绝对误差(MAE)
        mae = np.mean(np.abs(torch_out - tflite_out))
        # 计算最大误差
        max_error = np.max(np.abs(torch_out - tflite_out))
        
        print(f"样本 {i+1}:")
        print(f"  均方误差 (MSE): {mse:.8f}")
        print(f"  平均绝对误差 (MAE): {mae:.8f}")
        print(f"  最大误差: {max_error:.8f}")
        
        total_mse += mse
        total_mae += mae
        max_diff = max(max_diff, max_error)
    
    print("-" * 50)
    print(f"平均 MSE: {total_mse/total_samples:.8f}")
    print(f"平均 MAE: {total_mae/total_samples:.8f}")
    print(f"所有样本最大误差: {max_diff:.8f}")
    
    # 判断精度是否可接受
    if total_mae/total_samples < 1e-4:
        print("\n✅ 精度非常好: 平均误差小于 1e-4")
    elif total_mae/total_samples < 1e-3:
        print("\n✅ 精度良好: 平均误差小于 1e-3")
    elif total_mae/total_samples < 1e-2:
        print("\n⚠️ 精度一般: 平均误差小于 1e-2，可能需要调整转换参数")
    else:
        print("\n❌ 精度较差: 平均误差大于 1e-2，建议重新检查转换过程")


def main():
    parser = argparse.ArgumentParser(description="将TorchScript模型转换为TFLite格式并比较精度")
    parser.add_argument("--model_path", type=str, required=True, 
        default=r"E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\model_torchscript.pt", 
        help="TorchScript模型路径"
    )
    parser.add_argument("--param_path", type=str, 
        default=r"E:\HaozhanLi\Project\FlatLoc\tlio\models\LLIO512\ronin_ridi_imunet_tlio\parameters.json", 
        help="模型参数配置文件路径(parameters.json)"
    )
    parser.add_argument("--output_dir", type=str, default="./converted_models", help="输出目录")
    parser.add_argument("--input_shape", type=str, default=None, help="输入张量形状，格式为逗号分隔的数字")
    parser.add_argument("--quantize", action="store_true", help="是否启用量化")
    parser.add_argument("--num_samples", type=int, default=10, help="用于精度比较的样本数量")
    
    args = parser.parse_args()
    
    # 自动推断parameters.json路径（如果未指定）
    if args.param_path is None:
        model_dir = os.path.dirname(args.model_path)
        possible_param_path = os.path.join(model_dir, "parameters.json")
        if os.path.exists(possible_param_path):
            args.param_path = possible_param_path
            print(f"自动找到参数文件: {args.param_path}")
    
    # 确定输入形状
    input_shape = None
    
    # 如果提供了参数文件，从中读取输入形状
    if args.param_path and os.path.exists(args.param_path):
        params = load_model_parameters(args.param_path)
        input_shape = get_input_shape_from_params(params)
    
    # 如果命令行指定了输入形状，优先使用命令行指定的
    if args.input_shape:
        input_shape = tuple(map(int, args.input_shape.split(',')))
        print(f"使用命令行指定的输入形状: {input_shape}")
    
    # 如果仍然没有输入形状，使用默认值
    if input_shape is None:
        input_shape = (1, 6, 1000)
        print(f"未找到输入形状信息，使用默认值: {input_shape}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置输出文件路径
    onnx_path = output_dir / "model.onnx"
    tf_path = output_dir / "tf_model"
    tflite_path = output_dir / "model.tflite"
    
    # 加载TorchScript模型
    torch_model = load_torchscript_model(args.model_path)
    
    # 转换为ONNX
    convert_to_onnx(torch_model, input_shape, onnx_path)
    
    # 转换为TensorFlow
    convert_onnx_to_tf(onnx_path, tf_path)
    
    # 转换为TFLite
    convert_tf_to_tflite(tf_path, tflite_path, args.quantize)
    
    # 生成样本数据
    samples = generate_sample_data(input_shape, args.num_samples)
    
    # 运行TorchScript模型推理
    torch_results = run_torchscript_inference(torch_model, samples)
    
    # 运行TFLite模型推理
    tflite_results = run_tflite_inference(tflite_path, samples)
    
    # 比较结果
    compare_results(torch_results, tflite_results)
    
    print(f"\n转换完成！TFLite模型已保存至: {tflite_path}")


if __name__ == "__main__":
    main()