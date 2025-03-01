import json
import cv2
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import math
from torch.nn import Parameter  

# from pytorchtools import EarlyStopping
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

import shutil
import time
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch.optim import *
from torchvision.transforms import *
from torchsummary import summary


############

class MSMambaBlock(nn.Module):
    def __init__(self, dim=256, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(dim, d_state))
        self.B = nn.Parameter(torch.randn(dim, d_state))
        self.C = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.zeros(dim))
        
        # Forward path
        self.conv_forward = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Backward path
        self.conv_backward = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Projections
        self.forward_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        self.backward_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def selective_scan(self, x, direction="forward"):
        """Simplified selective scan operation"""
        if direction == "forward":
            x = torch.einsum('bcd,ce->bed', x, self.B @ self.C.T)
        else:
            x = torch.einsum('bcd,ce->bed', x.flip(-1), self.B @ self.C.T).flip(-1)
        return x
    
    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        # Reshape for processing while preserving spatial information
        x_reshaped = x.view(B, C, H * W)
        
        # Forward path - 关注局部上下文和前向依赖
        x_f = self.conv_forward(x_reshaped)
        x_f = F.gelu(x_f)  # 非线性激活
        x_f = self.selective_scan(x_f, "forward")
        x_f = x_f.view(B, self.dim, H, W)
        # 添加局部注意力
        x_f = x_f + F.avg_pool2d(x_f, 3, stride=1, padding=1)
        x_f = x_f.permute(0, 2, 3, 1)
        x_f = self.forward_proj(x_f)
        
        # Backward path - 关注全局上下文和反向依赖
        x_b = self.conv_backward(x_reshaped)
        x_b = F.gelu(x_b)  # 非线性激活
        x_b = self.selective_scan(x_b, "backward")
        x_b = x_b.view(B, self.dim, H, W)
        # 添加全局注意力
        x_b = x_b + F.adaptive_avg_pool2d(x_b, (H, W))
        x_b = x_b.permute(0, 2, 3, 1)
        x_b = self.backward_proj(x_b)
        
        # Combine paths
        x = x_f + x_b
        
        # 确保维度匹配
        d_state_ones = torch.ones(self.d_state, device=x.device)  # [d_state]
        dim_ones = torch.ones(self.dim, device=x.device)  # [dim]
        
        ssm_weight = torch.sigmoid(
            self.D[None, None, None, :] + 
            (self.A @ d_state_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0) +
            (self.B @ self.C.T @ dim_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
        
        # ssm_weight = torch.sigmoid(
        #     # 1. 基础偏置项
        #     self.D[None, None, None, :] + 
            
        #     # 2. A 矩阵的影响
        #     (self.A @ d_state_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0) +
            
        #     # 3. B和C矩阵的组合影响
        #     (self.B @ self.C.T @ dim_ones).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # )
        
        x = x * ssm_weight
        
        # Final processing
        x = self.norm(x)
        x = self.out_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return identity + 0.1 * x


def create_conv_sequence(in_channels=48, out_channels=512):
    return nn.Sequential(
        # 第一层：降维
        nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # 第二层
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        # 第三层
        nn.Conv2d(256, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class MultiTaskFusionGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super(MultiTaskFusionGate, self).__init__()
        self.channels = channels
        self.reduced_dim = channels // reduction
        
        # QKV transformations
        self.query = nn.Sequential(
            nn.Conv2d(channels*3, self.reduced_dim, 1, bias=False),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(inplace=True)
        )
        
        self.key = nn.Sequential(
            nn.Conv2d(channels*3, self.reduced_dim, 1, bias=False),
            nn.BatchNorm2d(self.reduced_dim)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Task-specific gates
        self.gate1_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate1_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate1_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gate2_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate2_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate2_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gate3_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate3_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate3_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gate4_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate4_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.gate4_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, h1, h2, h3):
        # Concatenate inputs
        concat = torch.cat([h1, h2, h3], dim=1)
        
        # Generate Q, K, V
        q = self.query(concat)
        k = self.key(concat)
        v = self.value(concat)
        
        # Attention computation
        B, C, H, W = q.shape
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)
        
        attn = torch.bmm(q.transpose(1, 2), k)
        attn = F.softmax(attn / math.sqrt(self.reduced_dim), dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, -1, H, W)
        
        # Apply task-specific gates
        emotion_feat = h1 * self.gate1_1(out) + h2 * self.gate1_2(out) + h3 * self.gate1_3(out)
        behavior_feat = h1 * self.gate2_1(out) + h2 * self.gate2_2(out) + h3 * self.gate2_3(out)
        scene_feat = h1 * self.gate3_1(out) + h2 * self.gate3_2(out) + h3 * self.gate3_3(out)
        vehicle_feat = h1 * self.gate4_1(out) + h2 * self.gate4_2(out) + h3 * self.gate4_3(out)
        
        return emotion_feat, behavior_feat, scene_feat, vehicle_feat

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        
        # 降低初始通道数和特征维度
        self.initial_conv1 = create_conv_sequence(in_channels=48, out_channels=128)
        self.initial_conv2 = create_conv_sequence(in_channels=48, out_channels=128)
        self.initial_conv3 = create_conv_sequence(in_channels=48, out_channels=128)
        self.initial_conv4 = create_conv_sequence(in_channels=48, out_channels=128)
        self.face_conv = create_conv_sequence(in_channels=48, out_channels=128)
        self.body_conv = create_conv_sequence(in_channels=48, out_channels=128)
        
        # 归一化层
        self.bn = nn.BatchNorm2d(128)
        self.norm = nn.LayerNorm([128, 64, 64])
        
        # Mamba块
        self.mamba_block1 = MSMambaBlock(dim=128, d_state=4)
        self.mamba_block2 = MSMambaBlock(dim=128, d_state=4)
        
        # 3D卷积
        self.conv3d_gesture = ConvNet3D(num_keypoints=26, out_dim=128)
        self.conv3d_posture = ConvNet3D(num_keypoints=42, out_dim=128)
        
        # Replace RDF gates with attention fusion gates
        self.fusion_gate = MultiTaskFusionGate(channels=128)
        
        # 为每个任务创建独立的池化层
        self.avg_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(4)
        ])
        
        # 为每个任务创建独立的分类器
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(4)
        ])
        
        self.fc1 = nn.Linear(128, 5)
        self.fc2 = nn.Linear(128, 7)
        self.fc3 = nn.Linear(128, 3)
        self.fc4 = nn.Linear(128, 5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用更保守的初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 更保守的线性层初始化
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img1, img2, img3, img4, face, body, gesture, posture):
        # 特征提取
        def process_feature(x, conv):
            x = conv(x)
            x = F.adaptive_avg_pool2d(x, (64, 64))
            return x * 0.1  # 缩放因子
        
        # 处理所有输入
        img1 = process_feature(img1, self.initial_conv1)
        img2 = process_feature(img2, self.initial_conv2)
        img3 = process_feature(img3, self.initial_conv3)
        img4 = process_feature(img4, self.initial_conv4)
        face = process_feature(face, self.face_conv)
        body = process_feature(body, self.body_conv)
        
        # 特征融合
        group1 = self.bn(img1 + face + body)
        group2 = self.bn(img2 + img3 + img4)
        
        # Mamba处理
        h1 = self.mamba_block1(group1)
        h2 = self.mamba_block2(group2)
        
        # 处理姿态和手势
        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)
        
        # 扩展维度
        h_gesture = h_gesture.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64) * 0.1
        h_posture = h_posture.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64) * 0.1
        h3 = self.bn(h_gesture + h_posture)
        
        # 使用新的多任务融合门控
        emotion_feat, behavior_feat, scene_feat, vehicle_feat = self.fusion_gate(h1, h2, h3)
        
        # 为每个任务独立处理特征
        emotion_out = self.classifiers[0](self.avg_pools[0](emotion_feat).flatten(1))
        behavior_out = self.classifiers[1](self.avg_pools[1](behavior_feat).flatten(1))
        scene_out = self.classifiers[2](self.avg_pools[2](scene_feat).flatten(1))
        vehicle_out = self.classifiers[3](self.avg_pools[3](vehicle_feat).flatten(1))
        
        # 返回每个任务的预测结果
        return (
            self.fc1(emotion_out),
            self.fc2(behavior_out),
            self.fc3(scene_out),
            self.fc4(vehicle_out)
        )



class ConvNet3D(nn.Module):
    def __init__(self, num_keypoints=26, out_dim=256):
        super(ConvNet3D, self).__init__()
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        # 添加投影层，将特征映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(128 * num_keypoints, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: [batch, channels, frames, keypoints, 1]
        B, C, F, K, _ = x.shape
        
        # 3D卷积处理
        x = self.conv3d(x)  # [B, 128, F, K, 1]
        
        # 重塑并投影到指定维度
        x = x.mean(dim=2)  # 在时间维度上平均池化 [B, 128, K, 1]
        x = x.view(B, -1)  # 展平 [B, 128*K]
        x = self.projection(x)  # 投影到指定维度 [B, out_dim]
        
        return x


###########
import torch
import time
from fvcore.nn import FlopCountAnalysis
from torchsummary import summary
import sys

cuda = torch.cuda.is_available()


class WrapperNet(torch.nn.Module):
    """
    Wrapper model to handle multi-input TotalNet for compatibility with torchsummary.
    """
    def __init__(self, model):
        super(WrapperNet, self).__init__()
        self.model = model

    def forward(self, img1):
        # 从单个输入生成所有需要的输入张量
        batch_size = img1.shape[0]
        H, W = img1.shape[2], img1.shape[3]
        
        # 生成相同大小的图像输入
        img2 = torch.randn_like(img1)
        img3 = torch.randn_like(img1)
        img4 = torch.randn_like(img1)
        
        # 生成face和body输入
        face = torch.randn(batch_size, 48, 64, 64).to(img1.device)
        body = torch.randn(batch_size, 48, 112, 112).to(img1.device)
        
        # 生成gesture和posture输入
        gesture = torch.randn(batch_size, 3, 16, 26, 1).to(img1.device)
        posture = torch.randn(batch_size, 3, 16, 42, 1).to(img1.device)

        return self.model(img1, img2, img3, img4, face, body, gesture, posture)

def calculate_flops_manually(model, input_shapes):
    """手动计算模型的 FLOPs"""
    def count_conv2d(layer, input_shape):
        out_h = input_shape[2] // layer.stride[0]
        out_w = input_shape[3] // layer.stride[1]
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
        flops = kernel_ops * layer.out_channels * out_h * out_w
        if layer.bias is not None:
            flops += layer.out_channels * out_h * out_w
        return flops

    def count_linear(layer, input_shape):
        flops = layer.in_features * layer.out_features
        if layer.bias is not None:
            flops += layer.out_features
        return flops

    total_flops = 0
    
    # 遍历模型的所有层
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取该层的输入形状
            if 'encoder' in name:
                input_shape = (1, module.in_channels, 224, 224)  # 编码器部分
            elif 'decoder' in name:
                input_shape = (1, module.in_channels, 112, 112)  # 解码器部分
            else:
                input_shape = (1, module.in_channels, 64, 64)    # 其他部分
            
            layer_flops = count_conv2d(module, input_shape)
            print(f"{name}: {layer_flops:,} FLOPs")
            total_flops += layer_flops
            
        elif isinstance(module, nn.Linear):
            layer_flops = count_linear(module, None)
            print(f"{name}: {layer_flops:,} FLOPs")
            total_flops += layer_flops

    return total_flops

def test_model_performance(model):
    """测试模型性能"""
    output_file = "./comparison/model_performance_MGv4MG+.txt"
    with open(output_file, "w") as f:
        # 重定向输出到文件和终端
        class Logger:
            def __init__(self, file):
                self.terminal = sys.stdout
                self.file = file

            def write(self, message):
                self.terminal.write(message)
                self.file.write(message)

            def flush(self):
                # 确保在关闭文件之前进行刷新
                if not self.file.closed:
                    self.terminal.flush()
                    self.file.flush()

        sys.stdout = Logger(f)

        print("="*50)
        print("Model Performance Analysis")
        print("="*50 + "\n")
        
        # 1. 模型结构和参数量
        print("Model Structure and Parameters:")
        print("-"*50)
        wrapped_model = WrapperNet(model)
        if torch.cuda.is_available():
            wrapped_model = wrapped_model.cuda()
        summary(wrapped_model, input_size=(48, 224, 224))
        
        # 2. 手动计算FLOPs
        print("\nComputing FLOPs manually:")
        print("-"*50)
        input_shapes = {
            'main': (1, 48, 224, 224),
            'aux1': (1, 48, 64, 64),
            'aux2': (1, 48, 112, 112)
        }
        total_flops = calculate_flops_manually(model, input_shapes)
        print(f"\nTotal FLOPs: {total_flops:,}")
        print(f"FLOPs (G): {total_flops/1e9:.2f}G")
        
        # 3. 测试FPS
        print("\nMeasuring Inference Speed:")
        print("-"*50)
        batch_size = 1
        
        # 准备输入数据
        input_data = [
            torch.randn(batch_size, 48, 224, 224),
            torch.randn(batch_size, 48, 224, 224),
            torch.randn(batch_size, 48, 224, 224),
            torch.randn(batch_size, 48, 224, 224),
            torch.randn(batch_size, 48, 64, 64),
            torch.randn(batch_size, 48, 112, 112),
            torch.randn(batch_size, 3, 16, 26, 1),
            torch.randn(batch_size, 3, 16, 42, 1)
        ]
        
        if torch.cuda.is_available():
            input_data = [x.cuda() for x in input_data]
            model = model.cuda()
        
        model.eval()
        
        # 预热
        print("Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(*input_data)
        
        # 测量推理时间
        times = []
        num_iters = 100
        print(f"Running {num_iters} iterations...")
        
        with torch.no_grad():
            for i in range(num_iters):
                start_time = time.time()
                _ = model(*input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{num_iters}")
        
        # 计算统计数据
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / mean_time
        
        print(f"\nResults:")
        print(f"Average inference time: {mean_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
        print(f"FPS: {fps:.2f}")

        # 恢复标准输出
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    model = TotalNet()
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    
    test_model_performance(model)
