"""
模型工厂模块
统一管理模型实例化逻辑，消除重复代码
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Dict, Any

# 导入模型
from model.model_wpdn import LightweightWaveletPacketCNN
from model.wavelet_transform import get_available_wavelets
from model.traditional_wavelet_packet import TraditionalWaveletPacketCNN

# 导入基线模型 - 更新为新的标准和轻量化版本
from model.baselines import (
    StandardLSTM, LightweightLSTM, 
    StandardGRU, LightweightGRU,
    StandardTransformer, LightweightTransformer,
    StandardCNN, LightweightCNN
)
from model.baselines.resnet_models import StandardResNet, LightweightResNet


from .config import DatasetConfig, ModelConfig


class ModelFactory:
    """模型工厂类，负责创建各种模型实例"""
    
    @staticmethod
    def create_model(mode: str, dataset_config: DatasetConfig, model_config: ModelConfig, device: torch.device) -> torch.nn.Module:
        """
        根据模式创建模型实例
        
        Args:
            mode: 模型模式
            dataset_config: 数据集配置
            model_config: 模型配置
            device: 设备
            
        Returns:
            模型实例
        """
        in_channels = dataset_config.in_channels
        num_classes = dataset_config.num_classes
        input_length = dataset_config.input_length
        kernel_size = dataset_config.kernel_size
        
        # 小波模型
        if mode == "wavelet_traditional":
            from model.traditional_wavelet_packet import TraditionalWaveletPacketCNN
            model = TraditionalWaveletPacketCNN(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length,
                wavelet=model_config.wavelet_type,
                levels=model_config.wavelet_levels,
                classifier_type="lightweight"
            ).to(device)
            ModelFactory._print_wavelet_traditional_info(model_config)
            
        elif mode == "wavelet_lite":
            model = LightweightWaveletPacketCNN(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length,
                kernel_size=kernel_size,
                use_parallel=False,
                num_parallel_groups=model_config.num_parallel_groups,
                decompose_levels=model_config.decompose_levels,
                verbose=False
            ).to(device)
            ModelFactory._print_wavelet_lite_info(model_config)
            
        # 基线模型 - 标准版本
        elif mode == "lstm":
            model = StandardLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_standard_lstm_info()
            
        elif mode == "gru":
            model = StandardGRU(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_standard_gru_info()
            
        elif mode == "transformer":
            model = StandardTransformer(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_standard_transformer_info()
            
        elif mode == "cnn":
            model = StandardCNN(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_standard_cnn_info()
            
        elif mode == "resnet":
            model = StandardResNet(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_standard_resnet_info()
            
        # 基线模型 - 轻量化版本
        elif mode == "lstm_lite":
            model = LightweightLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_lightweight_lstm_info()
            
        elif mode == "gru_lite":
            model = LightweightGRU(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_lightweight_gru_info()
            
        elif mode == "transformer_lite":
            model = LightweightTransformer(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_lightweight_transformer_info()
            
        elif mode == "cnn_lite":
            model = LightweightCNN(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_lightweight_cnn_info()
            
        elif mode == "resnet_lite":
            model = LightweightResNet(
                in_channels=in_channels,
                num_classes=num_classes,
                input_length=input_length
            ).to(device)
            ModelFactory._print_lightweight_resnet_info()
            
        else:
            raise ValueError(f"Unsupported model mode: {mode}")
        
        return model
    
    @staticmethod
    def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> bool:
        """
        加载模型权重
        
        Args:
            model: 模型实例
            checkpoint_path: 检查点路径
            device: 设备
            
        Returns:
            是否成功加载
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 模型权重文件不存在: {checkpoint_path}")
            print("将使用随机初始化的模型进行测试")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_dict = model.state_dict()
            
            # 检查参数形状是否匹配
            mismatched_params = []
            for k, v in checkpoint.items():
                if k in model_dict:
                    if model_dict[k].shape != v.shape:
                        mismatched_params.append(f"{k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
            
            if mismatched_params:
                print(f"❌ 模型结构不匹配，无法加载权重:")
                for param in mismatched_params:
                    print(f"   - {param}")
                print(f"请确保训练和测试时使用相同的模型配置")
                return False
            
            model.load_state_dict(checkpoint)
            print(f"✅ 模型权重加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型权重时出错: {e}")
            return False
    
    # 模型信息打印方法
    @staticmethod
    def _print_wavelet_traditional_info(model_config: ModelConfig):
        print(f"🏗️ 传统小波包CNN模型架构:")
        print(f"   - 小波类型: {model_config.wavelet_type}")
        print(f"   - 小波分解层数: {model_config.wavelet_levels}")
        print(f"   - 分类器: 标准CNN")
    
    @staticmethod
    def _print_wavelet_lite_info(model_config: ModelConfig):
        print(f"🏗️ 轻量化小波包CNN模型架构:")
        print(f"   - 并行组数: {model_config.num_parallel_groups}")
        print(f"   - 分解层数: {model_config.decompose_levels}")
        print(f"   - 分类器: 轻量化CNN")
    
    # 标准模型打印信息方法
    @staticmethod
    def _print_standard_lstm_info():
        print(f"🏗️ 标准LSTM模型架构:")
        print(f"   - 架构: 标准长短期记忆网络")
        print(f"   - 特征提取: 双向LSTM + 多层结构")
        print(f"   - 参数量: 较大，性能优先")
    
    @staticmethod
    def _print_standard_gru_info():
        print(f"🏗️ 标准GRU模型架构:")
        print(f"   - 架构: 标准门控循环单元网络")
        print(f"   - 特征提取: 双向GRU + 多层结构")
        print(f"   - 参数量: 较大，性能优先")
    
    @staticmethod
    def _print_standard_transformer_info():
        print(f"🏗️ 标准Transformer模型架构:")
        print(f"   - 架构: 标准Transformer网络")
        print(f"   - 特征提取: 多头自注意力机制 + 深层结构")
        print(f"   - 参数量: 较大，性能优先")
    
    @staticmethod
    def _print_standard_cnn_info():
        print(f"🏗️ 标准CNN模型架构:")
        print(f"   - 架构: 标准卷积神经网络")
        print(f"   - 特征提取: 深层卷积 + 复杂分类器")
        print(f"   - 参数量: 较大，性能优先")
    
    @staticmethod
    def _print_standard_resnet_info():
        print(f"🏗️ 标准ResNet模型架构:")
        print(f"   - 架构: 标准ResNet残差网络")
        print(f"   - 特征提取: 深层残差连接")
        print(f"   - 参数量: 较大，性能优先")
    
    # 轻量化模型打印信息方法
    @staticmethod
    def _print_lightweight_lstm_info():
        print(f"🏗️ 轻量化LSTM模型架构:")
        print(f"   - 架构: 轻量化长短期记忆网络")
        print(f"   - 特征提取: 单向LSTM + 简化结构")
        print(f"   - 参数量: 较小，效率优先")
    
    @staticmethod
    def _print_lightweight_gru_info():
        print(f"🏗️ 轻量化GRU模型架构:")
        print(f"   - 架构: 轻量化门控循环单元网络")
        print(f"   - 特征提取: 单向GRU + 简化结构")
        print(f"   - 参数量: 较小，效率优先")
    
    @staticmethod
    def _print_lightweight_transformer_info():
        print(f"🏗️ 轻量化Transformer模型架构:")
        print(f"   - 架构: 轻量化Transformer网络")
        print(f"   - 特征提取: 简化自注意力机制")
        print(f"   - 参数量: 较小，效率优先")
    
    @staticmethod
    def _print_lightweight_cnn_info():
        print(f"🏗️ 轻量化CNN模型架构:")
        print(f"   - 架构: 轻量化卷积神经网络")
        print(f"   - 特征提取: 浅层卷积 + 简化分类器")
        print(f"   - 参数量: 较小，效率优先")
    
    @staticmethod
    def _print_lightweight_resnet_info():
        print("模型架构: 轻量化ResNet")
        print("特征提取: 轻量化残差连接")