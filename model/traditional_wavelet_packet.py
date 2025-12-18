#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统离散小波包变换实现
用于生成时频图，支持多种小波基的消融实验
"""

import numpy as np
import torch
import torch.nn as nn
import pywt
from typing import List, Tuple, Optional

class TraditionalWaveletPacketTransform:
    """
    传统离散小波包变换类
    支持多种小波基函数
    """
    
    def __init__(self, wavelet='db4', levels=3, mode='symmetric'):
        """
        初始化小波包变换
        
        Args:
            wavelet: 小波基函数名称
            levels: 分解层数
            mode: 边界处理模式
        """
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        
        # 验证小波基是否可用
        if wavelet not in pywt.wavelist():
            raise ValueError(f"不支持的小波基: {wavelet}")
    
    def wavelet_packet_decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        执行小波包分解
        
        Args:
            signal: 输入信号 (length,)
            
        Returns:
            分解后的频带列表
        """
        # 创建小波包对象
        wp = pywt.WaveletPacket(signal, wavelet=self.wavelet, mode=self.mode)
        
        # 获取最后一层的所有节点
        nodes = [node.path for node in wp.get_level(self.levels, 'natural')]
        
        # 提取各频带的系数
        bands = []
        for node_path in nodes:
            coeffs = wp[node_path].data
            bands.append(coeffs)
        
        return bands
    
    def create_time_frequency_map(self, signal: np.ndarray) -> np.ndarray:
        """
        创建时频图
        
        Args:
            signal: 输入信号 (length,)
            
        Returns:
            时频图 (num_bands, time_steps)
        """
        # 执行小波包分解
        bands = self.wavelet_packet_decompose(signal)
        
        # 找到最小长度以对齐所有频带
        min_length = min(len(band) for band in bands)
        
        # 截断或填充到相同长度
        aligned_bands = []
        for band in bands:
            if len(band) >= min_length:
                aligned_bands.append(band[:min_length])
            else:
                # 零填充
                padded = np.zeros(min_length)
                padded[:len(band)] = band
                aligned_bands.append(padded)
        
        # 堆叠成时频图
        time_freq_map = np.stack(aligned_bands, axis=0)
        
        return time_freq_map

class TraditionalWaveletPacketCNN(nn.Module):
    """
    基于传统小波包变换的CNN分类器
    """
    
    def __init__(self, in_channels=3, num_classes=6, input_length=64, 
                 wavelet='db4', levels=3, classifier_type='ultra_lightweight'):
        """
        初始化模型
        
        Args:
            in_channels: 输入通道数
            num_classes: 分类数
            input_length: 输入序列长度
            wavelet: 小波基函数
            levels: 分解层数
            classifier_type: 分类器类型
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.wavelet = wavelet
        self.levels = levels
        
        # 创建小波包变换器
        self.wp_transform = TraditionalWaveletPacketTransform(
            wavelet=wavelet, levels=levels
        )
        
        # 计算频带数量
        self.num_bands = 2 ** levels
        
        # 估算时频图尺寸
        dummy_signal = np.random.randn(input_length)
        dummy_tf_map = self.wp_transform.create_time_frequency_map(dummy_signal)
        self.tf_height, self.tf_width = dummy_tf_map.shape
        
        # 创建分类器
        if classifier_type == 'ultra_lightweight':
            self.classifier = self._create_ultra_lightweight_classifier()
        elif classifier_type == 'lightweight':
            self.classifier = self._create_lightweight_classifier()
        elif classifier_type == 'standard':
            self.classifier = self._create_standard_classifier()
        else:
            raise ValueError(f"不支持的分类器类型: {classifier_type}")
    
    def _create_ultra_lightweight_classifier(self):
        """
        创建超轻量级分类器，使用WPDN的UltraLightweightConv2DClassifier结构
        """
        from .model_wpdn import UltraLightweightConv2DClassifier
        
        # 时频图形状
        time_freq_shape = (self.tf_height, self.tf_width)
        
        return UltraLightweightConv2DClassifier(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            time_freq_shape=time_freq_shape,
            use_parallel=False,
            num_parallel_groups=1
        )
    
    def _create_lightweight_classifier(self):
        """
        创建轻量级分类器
        """
        return nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二个卷积块
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # 分类层
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes)
        )
    
    def _create_standard_classifier(self):
        """
        创建标准分类器
        """
        return nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # 分类层
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, in_channels, length)
            
        Returns:
            分类logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 为每个样本和通道生成时频图
        time_freq_maps = []
        
        for b in range(batch_size):
            channel_maps = []
            for c in range(self.in_channels):
                # 提取单通道信号
                signal = x[b, c, :].detach().cpu().numpy()
                
                # 生成时频图
                tf_map = self.wp_transform.create_time_frequency_map(signal)
                channel_maps.append(tf_map)
            
            # 堆叠通道
            sample_tf_map = np.stack(channel_maps, axis=0)  # (in_channels, tf_height, tf_width)
            time_freq_maps.append(sample_tf_map)
        
        # 转换为张量
        tf_tensor = torch.tensor(np.stack(time_freq_maps, axis=0), 
                                dtype=torch.float32, device=x.device)
        
        # 通过分类器
        logits = self.classifier(tf_tensor)
        
        return logits
    
    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'TraditionalWaveletPacketCNN',
            'wavelet': self.wavelet,
            'levels': self.levels,
            'num_bands': self.num_bands,
            'tf_shape': (self.tf_height, self.tf_width),
            'total_params': total_params,
            'trainable_params': trainable_params
        }

# 支持的小波基列表
SUPPORTED_WAVELETS = {
    # Daubechies小波
    'db1': 'Daubechies 1 (Haar)',
    'db2': 'Daubechies 2',
    'db4': 'Daubechies 4',
    'db8': 'Daubechies 8',
    'db16': 'Daubechies 16',
    
    # Biorthogonal小波
    'bior2.2': 'Biorthogonal 2.2',
    'bior4.4': 'Biorthogonal 4.4',
    'bior6.8': 'Biorthogonal 6.8',
    
    # Coiflets小波
    'coif2': 'Coiflets 2',
    'coif4': 'Coiflets 4',
    'coif6': 'Coiflets 6',
    
    # Symlets小波
    'sym4': 'Symlets 4',
    'sym8': 'Symlets 8',
    'sym16': 'Symlets 16',
    
    # Morlet小波
    'morl': 'Morlet',
    
    # Mexican Hat小波
    'mexh': 'Mexican Hat'
}

def get_available_wavelets():
    """
    获取可用的小波基列表
    """
    available = {}
    for wavelet, description in SUPPORTED_WAVELETS.items():
        if wavelet in pywt.wavelist():
            available[wavelet] = description
    return available

def create_traditional_wavelet_model(wavelet='db4', **kwargs):
    """
    创建传统小波包模型的工厂函数
    
    Args:
        wavelet: 小波基名称
        **kwargs: 其他模型参数
        
    Returns:
        TraditionalWaveletPacketCNN模型实例
    """
    return TraditionalWaveletPacketCNN(wavelet=wavelet, **kwargs)