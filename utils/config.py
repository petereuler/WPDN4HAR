"""
配置管理模块
统一管理所有配置参数，包括数据集配置、模型配置、训练配置等
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str  # "UCIHAR", "WISDM", "PAMAP2", "MHEALTH"
    data_dir: str
    in_channels: int
    num_classes: int
    input_length: int
    kernel_size: int
    batch_size: int = 32

    # 数据划分类型配置
    # "stratified": 分层分割（保持类别平衡）
    # "subject_dependent": 依赖于受试者（按用户/受试者分割）
    # "subject_independent": 独立于受试者（用户间混合分割）
    split_type: str = "stratified"

    # MHEALTH特定配置
    step_size: Optional[int] = None
    exclude_null: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    mode: str  # 模型类型
    
    # 小波相关配置
    wavelet_type: str = "db4"
    wavelet_levels: int = 3
    decompose_levels: int = 3
    num_parallel_groups: int = 4
    
    # 设备配置
    device: str = "auto"


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.001
    
    # 优化器配置
    optimizer_type: str = "adam"
    scheduler_type: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # 正交损失配置
    orth_weight: float = 0.1  # 正交损失权重（仅用于可学习小波）
    
    # 保存配置
    save_best_only: bool = True
    save_frequency: int = 10


@dataclass
class TestConfig:
    """测试配置"""
    batch_size: int = 32
    num_inference_tests: int = 100
    tsne_perplexity: int = 30


class Config:
    """统一配置管理类"""
    
    # 数据集配置映射
    DATASET_CONFIGS = {
        "UCIHAR": DatasetConfig(
            name="UCIHAR",
            data_dir="dataset/UCIHAR",
            in_channels=6,
            num_classes=6,
            input_length=128,  # 50Hz * 2.56s = 128
            kernel_size=6,
            split_type="stratified"  # UCI-HAR使用分层分割
        ),
        "WISDM": DatasetConfig(
            name="WISDM",
            data_dir="dataset/WISDM",
            in_channels=3,
            num_classes=6,  # Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
            input_length=128,  # 滑动窗口的固定长度（50%重叠）
            kernel_size=6,
            split_type="stratified"  # 默认分层分割
        ),
        "PAMAP2": DatasetConfig(
            name="PAMAP2",
            data_dir="dataset/PAMAP2",
            in_channels=18,  # 3个IMU × 3轴加速度 + 3个IMU × 3轴角速度
            num_classes=12,  # 12个活动类别
            input_length=256,  # 滑动窗口的固定长度（50%重叠）
            kernel_size=6,
            split_type="stratified"  # 默认分层分割
        ),
        "MHEALTH": DatasetConfig(
            name="MHEALTH",
            data_dir="dataset/mhealth_raw_data.csv",
            in_channels=12,  # 胸部6通道 + 右手腕6通道
            num_classes=12,  # 过滤后的活动类别数
            input_length=128,
            kernel_size=6,
            split_type="subject_independent",  # MHEALTH默认按受试者分割（subject-independent）
            step_size=64,
            exclude_null=True
        )
    }
    
    # 模型模式映射
    MODEL_MODES = {
        # 小波模型
        "wavelet_traditional": "传统小波包CNN",
        "wavelet_lite": "轻量化小波包CNN",
        
        # 基线模型 - 标准版本
        "lstm": "标准LSTM",
        "gru": "标准GRU", 
        "transformer": "标准Transformer",
        "cnn": "标准CNN",
        "resnet": "标准ResNet",
        
        # 基线模型 - 轻量化版本
        "lstm_lite": "轻量化LSTM",
        "gru_lite": "轻量化GRU",
        "transformer_lite": "轻量化Transformer", 
        "cnn_lite": "轻量化CNN",
        "resnet_lite": "轻量化ResNet"
    }
    
    # 模式详细描述映射
    MODE_DESCRIPTIONS = {
        "wavelet_traditional": "传统小波包CNN",
        "wavelet_lite": "轻量化小波包CNN",
        
        # 基线模型描述
        "lstm": "标准LSTM",
        "gru": "标准GRU",
        "transformer": "标准Transformer",
        "cnn": "标准CNN", 
        "resnet": "标准ResNet",
        
        # 基线模型 - 轻量化版本
        "lstm_lite": "轻量化LSTM",
        "gru_lite": "轻量化GRU",
        "transformer_lite": "轻量化Transformer",
        "cnn_lite": "轻量化CNN",
        "resnet_lite": "轻量化ResNet"
    }
    
    @classmethod
    def get_dataset_config(cls, dataset_name: str) -> DatasetConfig:
        """获取数据集配置"""
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return cls.DATASET_CONFIGS[dataset_name]
    
    @classmethod
    def get_model_checkpoint_path(cls, mode: str, dataset_name: str) -> str:
        """获取模型检查点路径"""
        return f"checkpoints/best_{mode}_{dataset_name}.pth"
    
    @classmethod
    def get_results_dir(cls, mode: str, dataset_name: str, wavelet_type: str = None) -> str:
        """获取结果保存目录"""
        if mode == "wavelet_traditional" and wavelet_type:
            return f"test_results_{mode}_{wavelet_type}_{dataset_name}"
        else:
            return f"test_results_{mode}_{dataset_name}"
    
    @classmethod
    def get_log_name(cls, mode: str) -> str:
        """获取日志名称"""
        log_names = {
            "wavelet_traditional": "传统小波包CNN",
            "wavelet_lite": "轻量化小波包CNN",
            
            # 标准模型
            "lstm": "标准LSTM",
            "gru": "标准GRU",
            "transformer": "标准Transformer",
            "cnn": "标准CNN",
            "resnet": "标准ResNet",
            
            # 轻量化模型
            "lstm_lite": "轻量化LSTM",
            "gru_lite": "轻量化GRU", 
            "transformer_lite": "轻量化Transformer",
            "cnn_lite": "轻量化CNN",
            "resnet_lite": "轻量化ResNet"
        }
        return log_names.get(mode, mode)
    
    @classmethod
    def get_architecture_description(cls, mode: str) -> str:
        """获取架构描述"""
        arch_descriptions = {
            "wavelet_traditional": "传统小波包分解 + 标准CNN分类器",
            "wavelet_lite": "可学习小波包分解 + 轻量化CNN分类器",
            
            # 标准模型
            "lstm": "标准长短期记忆网络",
            "gru": "标准门控循环单元网络",
            "transformer": "标准Transformer网络",
            "cnn": "标准卷积神经网络",
            "resnet": "标准残差神经网络",
            
            # 轻量化模型
            "lstm_lite": "轻量化长短期记忆网络",
            "gru_lite": "轻量化门控循环单元网络",
            "transformer_lite": "轻量化Transformer网络",
            "cnn_lite": "轻量化卷积神经网络",
            "resnet_lite": "轻量化残差神经网络"
        }
        return arch_descriptions.get(mode, mode)
    
    @classmethod
    def get_feature_extraction_description(cls, mode: str) -> str:
        """获取特征提取描述"""
        feature_descriptions = {
            "wavelet_traditional": "传统小波包变换",
            "wavelet_lite": "可学习小波包变换 + 深度可分离卷积",
            
            # 标准模型
            "lstm": "标准LSTM循环单元",
            "gru": "标准GRU循环单元",
            "transformer": "标准多头自注意力机制",
            "cnn": "标准卷积",
            "resnet": "标准残差连接",
            
            # 轻量化模型
            "lstm_lite": "轻量化LSTM循环单元",
            "gru_lite": "轻量化GRU循环单元",
            "transformer_lite": "轻量化多头自注意力机制",
            "cnn_lite": "轻量化卷积",
            "resnet_lite": "轻量化残差连接"
        }
        return feature_descriptions.get(mode, "标准卷积")
    
    @classmethod
    def setup_device(cls, device_str: str = "auto") -> torch.device:
        """设置设备"""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device_str)
    
    @classmethod
    def print_config_summary(cls, dataset_config: DatasetConfig, model_config: ModelConfig, 
                           training_config: TrainingConfig = None, test_config: TestConfig = None):
        """打印配置摘要"""
        print(f"\n{'='*60}")
        print(f"配置摘要")
        print(f"{'='*60}")
        
        # 数据集配置
        print(f"📊 数据集配置:")
        print(f"   - 数据集: {dataset_config.name}")
        print(f"   - 输入通道: {dataset_config.in_channels}")
        print(f"   - 类别数: {dataset_config.num_classes}")
        print(f"   - 时间窗口: {dataset_config.input_length}")
        print(f"   - 批次大小: {dataset_config.batch_size}")
        print(f"   - 划分类型: {dataset_config.split_type}")

        # 解释划分类型
        split_explanation = {
            "stratified": "分层分割（保持类别平衡）",
            "subject_dependent": "依赖于受试者（按用户分割）",
            "subject_independent": "独立于受试者（用户间混合）"
        }.get(dataset_config.split_type, dataset_config.split_type)
        print(f"     └─ {split_explanation}")
        
        # 模型配置
        print(f"🏗️ 模型配置:")
        print(f"   - 模式: {model_config.mode}")
        print(f"   - 设备: {model_config.device}")
        if model_config.mode == "wavelet_traditional":
            print(f"   - 小波类型: {model_config.wavelet_type}")
            print(f"   - 分解层数: {model_config.wavelet_levels}")
        elif model_config.mode in ["wavelet_learnable", "wavelet_lite"]:
            print(f"   - 分解级数: {model_config.decompose_levels}")
            print(f"   - 并行组数: {model_config.num_parallel_groups}")
        
        # 训练配置
        if training_config:
            print(f"🎯 训练配置:")
            print(f"   - 训练轮数: {training_config.epochs}")
            print(f"   - 学习率: {training_config.learning_rate}")
            print(f"   - 权重衰减: {training_config.weight_decay}")
            print(f"   - 早停耐心: {training_config.patience}")
            print(f"   - 正交损失权重: {training_config.orth_weight}")
            print(f"   - 优化器: {training_config.optimizer_type}")
        
        # 测试配置
        if test_config:
            print(f"🔬 测试配置:")
            print(f"   - 推理测试次数: {test_config.num_inference_tests}")
            print(f"   - t-SNE困惑度: {test_config.tsne_perplexity}")
        
        print(f"{'='*60}\n")