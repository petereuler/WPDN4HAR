"""
数据集工具模块。

职责：
1. 统一封装各数据集 train/val/test loader 的创建
2. 隔离各数据集特有参数，例如 MHEALTH 的受试者划分
3. 提供统一的类别名和数据集摘要接口
"""

from typing import Callable, Dict, Tuple

import torch.utils.data

from .config import DatasetConfig


LoaderGetter = Callable[[DatasetConfig], torch.utils.data.DataLoader]


class DatasetLoader:
    """数据集加载器类。"""

    _CLASS_NAME_GETTERS: Dict[str, Callable[[DatasetConfig], list]] = {}

    @staticmethod
    def create_train_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        return DatasetLoader._dispatch_loader(dataset_config, split="train")

    @staticmethod
    def create_val_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        return DatasetLoader._dispatch_loader(dataset_config, split="val")

    @staticmethod
    def create_test_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        return DatasetLoader._dispatch_loader(dataset_config, split="test")

    @staticmethod
    def _dispatch_loader(dataset_config: DatasetConfig, split: str) -> torch.utils.data.DataLoader:
        handlers: Dict[str, Dict[str, LoaderGetter]] = {
            "UCIHAR": {
                "train": DatasetLoader._create_ucihar_train_loader,
                "val": DatasetLoader._create_ucihar_val_loader,
                "test": DatasetLoader._create_ucihar_test_loader,
            },
            "WISDM": {
                "train": DatasetLoader._create_wisdm_train_loader,
                "val": DatasetLoader._create_wisdm_val_loader,
                "test": DatasetLoader._create_wisdm_test_loader,
            },
            "PAMAP2": {
                "train": DatasetLoader._create_pamap2_train_loader,
                "val": DatasetLoader._create_pamap2_val_loader,
                "test": DatasetLoader._create_pamap2_test_loader,
            },
            "MHEALTH": {
                "train": DatasetLoader._create_mhealth_train_loader,
                "val": DatasetLoader._create_mhealth_val_loader,
                "test": DatasetLoader._create_mhealth_test_loader,
            },
        }

        if dataset_config.name not in handlers:
            raise ValueError(f"Unsupported dataset: {dataset_config.name}")

        return handlers[dataset_config.name][split](dataset_config)

    @staticmethod
    def _create_ucihar_train_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_UCIHAR import create_train_val_loaders

        train_loader, _ = create_train_val_loaders(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            val_split=0.1,
        )
        return train_loader

    @staticmethod
    def _create_ucihar_val_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_UCIHAR import create_train_val_loaders

        _, val_loader = create_train_val_loaders(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            val_split=0.1,
        )
        return val_loader

    @staticmethod
    def _create_ucihar_test_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_UCIHAR import create_test_loader

        test_loader, _ = create_test_loader(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
        )
        return test_loader

    @staticmethod
    def _create_wisdm_train_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_WISDM import create_train_val_loaders

        train_loader, _, _ = create_train_val_loaders(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            val_split=0.2,
        )
        return train_loader

    @staticmethod
    def _create_wisdm_val_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_WISDM import create_train_val_loaders

        _, val_loader, _ = create_train_val_loaders(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            val_split=0.2,
        )
        return val_loader

    @staticmethod
    def _create_wisdm_test_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_WISDM import create_test_loader

        test_loader, _ = create_test_loader(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
        )
        return test_loader

    @staticmethod
    def _create_pamap2_train_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_PAMAP2 import create_train_loader

        return create_train_loader(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
        )

    @staticmethod
    def _create_pamap2_val_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_PAMAP2 import create_val_loader

        return create_val_loader(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
        )

    @staticmethod
    def _create_pamap2_test_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_PAMAP2 import create_test_loader

        test_loader, _ = create_test_loader(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
        )
        return test_loader

    @staticmethod
    def _mhealth_test_subjects(dataset_config: DatasetConfig):
        if dataset_config.split_type == "subject_independent":
            return ["subject9", "subject10"]
        return None

    @staticmethod
    def _create_mhealth_train_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_mhealth import create_train_loader

        return create_train_loader(
            data_file=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            step_size=dataset_config.step_size,
            exclude_null=dataset_config.exclude_null,
            test_subjects=DatasetLoader._mhealth_test_subjects(dataset_config),
        )

    @staticmethod
    def _create_mhealth_val_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_mhealth import create_val_loader

        return create_val_loader(
            data_file=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            step_size=dataset_config.step_size,
            exclude_null=dataset_config.exclude_null,
            test_subjects=DatasetLoader._mhealth_test_subjects(dataset_config),
        )

    @staticmethod
    def _create_mhealth_test_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        from dataset_process.dataset_mhealth import create_test_loader

        test_loader, _ = create_test_loader(
            data_path=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            step_size=dataset_config.step_size,
            exclude_null=dataset_config.exclude_null,
            test_subjects=DatasetLoader._mhealth_test_subjects(dataset_config),
        )
        return test_loader

    @staticmethod
    def get_class_names(dataset_config: DatasetConfig) -> list:
        if dataset_config.name == "PAMAP2":
            from dataset_process.dataset_PAMAP2 import ACTIVITY_NAMES

            return ACTIVITY_NAMES
        if dataset_config.name == "MHEALTH":
            from dataset_process.dataset_mhealth import MHEALTH_ACTIVITY_LABELS

            return [MHEALTH_ACTIVITY_LABELS[i + 1] for i in range(dataset_config.num_classes)]
        return [f"Class {i}" for i in range(dataset_config.num_classes)]

    @staticmethod
    def print_dataset_info(dataset_config: DatasetConfig) -> None:
        print(f"🎯 使用数据集: {dataset_config.name}")
        print(f"   - 输入通道数: {dataset_config.in_channels}")
        print(f"   - 类别数: {dataset_config.num_classes}")
        print(f"   - 时间窗口: {dataset_config.input_length} 时间步")

        if dataset_config.name == "UCIHAR":
            print("   - 采样频率: 50Hz")
            print(f"   - 时间长度: {dataset_config.input_length / 50:.2f}s")
        elif dataset_config.name == "WISDM":
            print("   - 采样频率: ~20Hz")
            print("   - 窗口大小: 128 时间步（50%重叠）")
            print("   - 活动类型: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing")
            print("   - 用户数量: 36个用户")
        elif dataset_config.name == "PAMAP2":
            print("   - 采样频率: 100Hz")
            print("   - 窗口大小: 256 时间步（50%重叠）")
            print(f"   - 时间长度: {dataset_config.input_length / 100:.2f}s")
            print("   - 传感器: 3个IMU (加速度计+陀螺仪)")
        elif dataset_config.name == "MHEALTH":
            print("   - 传感器: 胸部+右手腕 (加速度计+陀螺仪)")
            print("   - 过滤: 排除空值/过渡状态 (类别0)")
            if dataset_config.step_size:
                print(f"   - 步长: {dataset_config.step_size}")
