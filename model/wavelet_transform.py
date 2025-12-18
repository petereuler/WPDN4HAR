#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的小波变换模块
提供传统小波包变换的时频图生成功能
"""

import torch
import torch.nn as nn
import numpy as np
import pywt
from typing import List, Tuple, Optional

class TraditionalWaveletTransform(nn.Module):
    """
    传统小波包变换
    使用PyWavelets库实现标准的离散小波包变换
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 3, mode: str = 'symmetric'):
        """
        初始化传统小波变换
        
        Args:
            wavelet: 小波基类型 ('db4', 'db8', 'bior4.4', 'coif4', 'haar'等)
            levels: 分解层数
            mode: 边界处理模式
        """
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        
        # 验证小波基是否可用
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' is not supported. Available wavelets: {pywt.wavelist()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入信号进行小波包变换
        
        Args:
            x: 输入张量 (batch_size, channels, length)
            
        Returns:
            时频图张量 (batch_size, channels, freq_bins, time_bins)
        """
        batch_size, channels, length = x.shape
        device = x.device
        
        # 存储所有通道的小波系数
        all_coeffs = []
        
        for b in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                # 转换为numpy进行小波变换
                signal = x[b, c].cpu().numpy()
                
                # 执行小波包变换
                wp = pywt.WaveletPacket(signal, wavelet=self.wavelet, mode=self.mode)
                
                # 获取指定层数的所有节点
                coeffs = []
                nodes = [node.path for node in wp.get_level(self.levels, 'natural')]
                
                for node_path in nodes:
                    node_coeffs = wp[node_path].data
                    coeffs.append(node_coeffs)
                
                # 将系数组织成时频图
                # 每个节点的系数作为一个频率带
                max_len = max(len(coeff) for coeff in coeffs)
                
                # 填充到相同长度
                padded_coeffs = []
                for coeff in coeffs:
                    if len(coeff) < max_len:
                        # 零填充
                        padded = np.pad(coeff, (0, max_len - len(coeff)), mode='constant')
                    else:
                        padded = coeff[:max_len]
                    padded_coeffs.append(padded)
                
                # 堆叠成时频图 (freq_bins, time_bins)
                time_freq_map = np.stack(padded_coeffs, axis=0)
                batch_coeffs.append(time_freq_map)
            
            # 堆叠所有通道
            all_coeffs.append(np.stack(batch_coeffs, axis=0))
        
        # 转换回torch张量
        result = torch.from_numpy(np.stack(all_coeffs, axis=0)).float().to(device)
        
        return result
    
    def get_output_shape(self, input_length: int) -> Tuple[int, int]:
        """
        获取输出时频图的形状
        
        Args:
            input_length: 输入信号长度
            
        Returns:
            (freq_bins, time_bins)
        """
        # 频率bins数量 = 2^levels (小波包分解的节点数)
        freq_bins = 2 ** self.levels
        
        # 时间bins数量约等于 input_length / (2^levels)
        time_bins = input_length // (2 ** self.levels)
        
        return freq_bins, time_bins

def get_available_wavelets() -> dict:
    """
    获取可用的小波基及其描述
    
    Returns:
        字典，键为小波基名称，值为描述
    """
    wavelets = {
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
        
        # Haar小波
        'haar': 'Haar wavelet',
        
        # Morlet小波
        'morl': 'Morlet wavelet',
        'mexh': 'Mexican hat wavelet'
    }
    
    # 只返回PyWavelets支持的小波基
    available = {}
    for name, desc in wavelets.items():
        if name in pywt.wavelist():
            available[name] = desc
    
    return available

def test_wavelet_transform():
    """
    测试小波变换功能
    """
    print("测试传统小波变换...")
    
    # 创建测试数据
    batch_size, channels, length = 2, 6, 128
    x = torch.randn(batch_size, channels, length)
    
    # 测试不同小波基
    wavelets_to_test = ['db4', 'db8', 'bior4.4', 'coif4']
    
    for wavelet in wavelets_to_test:
        if wavelet in pywt.wavelist():
            print(f"\n测试小波基: {wavelet}")
            transform = TraditionalWaveletTransform(wavelet=wavelet, levels=3)
            
            try:
                output = transform(x)
                print(f"输入形状: {x.shape}")
                print(f"输出形状: {output.shape}")
                
                freq_bins, time_bins = transform.get_output_shape(length)
                print(f"预期输出形状: (batch_size={batch_size}, channels={channels}, freq_bins={freq_bins}, time_bins={time_bins})")
                
            except Exception as e:
                print(f"错误: {e}")
        else:
            print(f"跳过不支持的小波基: {wavelet}")
    
    print("\n可用的小波基:")
    available = get_available_wavelets()
    for name, desc in available.items():
        print(f"  {name}: {desc}")

if __name__ == '__main__':
    test_wavelet_transform()