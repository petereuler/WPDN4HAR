"""
模型工厂模块。

职责：
1. 根据 mode 实例化模型
2. 打印模型结构摘要
3. 统一加载 checkpoint 并给出可读错误
"""

import os
from typing import Dict, List, Tuple, Type

import torch

from model.baselines import (
    LightweightCNN,
    LightweightGRU,
    LightweightLSTM,
    LightweightTransformer,
    StandardCNN,
    StandardGRU,
    StandardLSTM,
    StandardTransformer,
)
from model.baselines.resnet_models import LightweightResNet, StandardResNet
from model.model_wpdn import LightweightWaveletPacketCNN
from model.traditional_wavelet_packet import TraditionalWaveletPacketCNN

from .config import DatasetConfig, ModelConfig


BaselineBuilder = Type[torch.nn.Module]


class ModelFactory:
    """模型工厂类，负责创建各种模型实例。"""

    _BASELINE_BUILDERS: Dict[str, BaselineBuilder] = {
        "lstm": StandardLSTM,
        "gru": StandardGRU,
        "transformer": StandardTransformer,
        "cnn": StandardCNN,
        "resnet": StandardResNet,
        "lstm_lite": LightweightLSTM,
        "gru_lite": LightweightGRU,
        "transformer_lite": LightweightTransformer,
        "cnn_lite": LightweightCNN,
        "resnet_lite": LightweightResNet,
    }

    @staticmethod
    def create_model(
        mode: str,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        device: torch.device,
    ) -> torch.nn.Module:
        """
        根据模式创建模型实例。
        """
        if mode == "wavelet_traditional":
            model = ModelFactory._create_wavelet_traditional(dataset_config, model_config)
            ModelFactory._print_wavelet_traditional_info(model_config)
        elif mode == "wavelet_lite":
            model = ModelFactory._create_wavelet_lite(dataset_config, model_config)
            ModelFactory._print_wavelet_lite_info(model_config)
        elif mode in ModelFactory._BASELINE_BUILDERS:
            model = ModelFactory._create_baseline(mode, dataset_config)
            ModelFactory._print_baseline_info(mode)
        else:
            raise ValueError(f"Unsupported model mode: {mode}")

        return model.to(device)

    @staticmethod
    def _common_kwargs(dataset_config: DatasetConfig) -> Dict[str, int]:
        return {
            "in_channels": dataset_config.in_channels,
            "num_classes": dataset_config.num_classes,
            "input_length": dataset_config.input_length,
        }

    @staticmethod
    def _create_wavelet_traditional(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> torch.nn.Module:
        return TraditionalWaveletPacketCNN(
            **ModelFactory._common_kwargs(dataset_config),
            wavelet=model_config.wavelet_type,
            levels=model_config.wavelet_levels,
            classifier_type="lightweight",
        )

    @staticmethod
    def _create_wavelet_lite(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> torch.nn.Module:
        return LightweightWaveletPacketCNN(
            **ModelFactory._common_kwargs(dataset_config),
            kernel_size=dataset_config.kernel_size,
            use_parallel=model_config.use_parallel_wavelet_kernels,
            num_parallel_groups=model_config.num_parallel_groups,
            decompose_levels=model_config.decompose_levels,
            classifier_rank_max=model_config.classifier_factor_rank,
            classifier_out_feature_groups=model_config.classifier_feature_groups,
            verbose=False,
        )

    @classmethod
    def _create_baseline(cls, mode: str, dataset_config: DatasetConfig) -> torch.nn.Module:
        builder = cls._BASELINE_BUILDERS[mode]
        return builder(**cls._common_kwargs(dataset_config))

    @staticmethod
    def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> bool:
        """
        加载模型权重。只要结构不兼容，就给出明确说明并返回 False。
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠️ 模型权重文件不存在: {checkpoint_path}")
            print("将使用随机初始化的模型进行测试")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_state = model.state_dict()

            missing_keys = [key for key in model_state.keys() if key not in checkpoint]
            unexpected_keys = [key for key in checkpoint.keys() if key not in model_state]
            mismatched_shapes = ModelFactory._find_mismatched_shapes(model_state, checkpoint)

            if missing_keys or unexpected_keys or mismatched_shapes:
                print("❌ 模型结构不匹配，无法加载权重:")
                ModelFactory._print_load_issues(missing_keys, unexpected_keys, mismatched_shapes)
                print("请确保训练和测试时使用相同的模型配置")
                return False

            model.load_state_dict(checkpoint)
            print(f"✅ 模型权重加载成功: {checkpoint_path}")
            return True
        except Exception as exc:
            print(f"❌ 加载模型权重时出错: {exc}")
            return False

    @staticmethod
    def _find_mismatched_shapes(
        model_state: Dict[str, torch.Tensor],
        checkpoint_state: Dict[str, torch.Tensor],
    ) -> List[Tuple[str, torch.Size, torch.Size]]:
        mismatched = []
        for key, value in checkpoint_state.items():
            if key in model_state and model_state[key].shape != value.shape:
                mismatched.append((key, value.shape, model_state[key].shape))
        return mismatched

    @staticmethod
    def _print_load_issues(
        missing_keys: List[str],
        unexpected_keys: List[str],
        mismatched_shapes: List[Tuple[str, torch.Size, torch.Size]],
    ) -> None:
        if missing_keys:
            print("   - checkpoint 缺少参数:")
            for key in missing_keys[:10]:
                print(f"     {key}")
            if len(missing_keys) > 10:
                print(f"     ... 还有 {len(missing_keys) - 10} 个")

        if unexpected_keys:
            print("   - checkpoint 存在多余参数:")
            for key in unexpected_keys[:10]:
                print(f"     {key}")
            if len(unexpected_keys) > 10:
                print(f"     ... 还有 {len(unexpected_keys) - 10} 个")

        if mismatched_shapes:
            print("   - 参数形状不匹配:")
            for key, checkpoint_shape, model_shape in mismatched_shapes[:10]:
                print(f"     {key}: checkpoint {checkpoint_shape} vs model {model_shape}")
            if len(mismatched_shapes) > 10:
                print(f"     ... 还有 {len(mismatched_shapes) - 10} 个")

    @staticmethod
    def _print_wavelet_traditional_info(model_config: ModelConfig) -> None:
        print("🏗️ 传统小波包CNN模型架构:")
        print(f"   - 小波类型: {model_config.wavelet_type}")
        print(f"   - 小波分解层数: {model_config.wavelet_levels}")
        print("   - 分类器: 标准CNN")

    @staticmethod
    def _print_wavelet_lite_info(model_config: ModelConfig) -> None:
        print("🏗️ 轻量化小波包CNN模型架构:")
        print(f"   - 分解层数: {model_config.decompose_levels}")
        print(f"   - 启用并行核: {model_config.use_parallel_wavelet_kernels}")
        if model_config.use_parallel_wavelet_kernels:
            print(f"   - 并行组数: {model_config.num_parallel_groups}")
        print(f"   - 分类器因子化秩: {model_config.classifier_factor_rank}")
        feature_groups_desc = (
            f"{model_config.classifier_feature_groups}"
            if model_config.classifier_feature_groups is not None
            else "auto"
        )
        print(f"   - 分类器输出特征组: {feature_groups_desc}")
        print("     └─ 每个频带在分类头中的输出通道数")
        print("   - 分类器: 低秩时频Conv1d分类头")

    @staticmethod
    def _print_baseline_info(mode: str) -> None:
        descriptions = {
            "lstm": ("标准LSTM模型架构", "标准长短期记忆网络", "双向LSTM + 多层结构", "较大，性能优先"),
            "gru": ("标准GRU模型架构", "标准门控循环单元网络", "双向GRU + 多层结构", "较大，性能优先"),
            "transformer": ("标准Transformer模型架构", "标准Transformer网络", "多头自注意力机制 + 深层结构", "较大，性能优先"),
            "cnn": ("标准CNN模型架构", "标准卷积神经网络", "深层卷积 + 复杂分类器", "较大，性能优先"),
            "resnet": ("标准ResNet模型架构", "标准残差网络", "残差块 + 深层特征提取", "较大，性能优先"),
            "lstm_lite": ("轻量化LSTM模型架构", "轻量化长短期记忆网络", "浅层LSTM + 简化分类器", "较小，效率优先"),
            "gru_lite": ("轻量化GRU模型架构", "轻量化门控循环单元网络", "浅层GRU + 简化分类器", "较小，效率优先"),
            "transformer_lite": ("轻量化Transformer模型架构", "轻量化Transformer网络", "少头注意力 + 简化FFN", "较小，效率优先"),
            "cnn_lite": ("轻量化CNN模型架构", "轻量化卷积神经网络", "浅层卷积 + 简化分类器", "较小，效率优先"),
            "resnet_lite": ("轻量化ResNet模型架构", "轻量化残差网络", "少量残差块 + 简化分类器", "较小，效率优先"),
        }
        title, architecture, feature_extraction, parameter_scale = descriptions[mode]
        print(f"🏗️ {title}:")
        print(f"   - 架构: {architecture}")
        print(f"   - 特征提取: {feature_extraction}")
        print(f"   - 参数量: {parameter_scale}")
