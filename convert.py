#!/bin/bash

import torch
import torchvision.models as models

# 加载预训练的 ResNet50 模型
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)

# 加载自定义的 7 分类权重文件
weight_path = 'resnet50-112.pt'
try:
    model.load_state_dict(torch.load(weight_path))
    print(f"成功加载权重文件: {weight_path}")
except FileNotFoundError:
    print(f"未找到权重文件: {weight_path}，请检查文件路径。")
except Exception as e:
    print(f"加载权重文件时出错: {e}")

model.eval()

# 定义输入示例
# 输入形状为 [批量大小, 通道数, 高度, 宽度]
input_tensor = torch.randn(1, 3, 112, 112)

# 导出模型为 ONNX 格式
onnx_path = "resnet50-112.onnx"
torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    export_params=True,  # 保存模型的参数
    opset_version=11,  # ONNX 操作集版本
    do_constant_folding=True,  # 执行常量折叠优化
    input_names=['input'],  # 输入张量的名称
    output_names=['output'],  # 输出张量的名称
)

print(f"ResNet50 模型已成功转换为 ONNX 格式，保存路径为: {onnx_path}")

