"""
数据集工具模块
统一管理数据集加载逻辑
"""

from typing import Tuple, Any
import torch.utils.data

from .config import DatasetConfig


class DatasetLoader:
    """数据集加载器类"""
    
    @staticmethod
    def create_train_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        """
        创建训练数据加载器
        
        Args:
            dataset_config: 数据集配置
            
        Returns:
            训练数据加载器
        """
        if dataset_config.name == "UCIHAR":
            from dataset_process.dataset_UCIHAR import create_train_val_loaders
            train_loader, _ = create_train_val_loaders(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                val_split=0.1
            )
            return train_loader
        elif dataset_config.name == "WISDM":
            from dataset_process.dataset_WISDM import create_train_val_loaders
            # WISDM只支持分层分割，不支持按用户分割
            train_loader, _, _ = create_train_val_loaders(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                val_split=0.2
                # WISDM不支持user_split参数，移除此参数
            )
            return train_loader
        elif dataset_config.name == "PAMAP2":
            from dataset_process.dataset_PAMAP2 import create_train_loader
            # PAMAP2只支持分层分割，不支持按受试者分割
            return create_train_loader(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size
                # PAMAP2不支持subject_split参数，移除此参数
            )
        elif dataset_config.name == "MHEALTH":
            from dataset_process.dataset_mhealth import create_train_loader
            # MHEALTH默认按受试者分割，如果要改成分层分割则需要修改test_subjects
            test_subjects = ['subject9', 'subject10'] if dataset_config.split_type == "subject_independent" else None
            return create_train_loader(
                data_file=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                step_size=dataset_config.step_size,
                exclude_null=dataset_config.exclude_null,
                test_subjects=test_subjects
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_config.name}")
    
    @staticmethod
    def create_val_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        """
        创建验证数据加载器
        
        Args:
            dataset_config: 数据集配置
            
        Returns:
            验证数据加载器
        """
        if dataset_config.name == "UCIHAR":
            from dataset_process.dataset_UCIHAR import create_train_val_loaders
            _, val_loader = create_train_val_loaders(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                val_split=0.1
            )
            return val_loader
        elif dataset_config.name == "WISDM":
            from dataset_process.dataset_WISDM import create_train_val_loaders
            # WISDM只支持分层分割，不支持按用户分割
            _, val_loader, _ = create_train_val_loaders(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                val_split=0.2
                # WISDM不支持user_split参数，移除此参数
            )
            return val_loader
        elif dataset_config.name == "PAMAP2":
            from dataset_process.dataset_PAMAP2 import create_val_loader
            # PAMAP2只支持分层分割，不支持按受试者分割
            return create_val_loader(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size
                # PAMAP2不支持subject_split参数，移除此参数
            )
        elif dataset_config.name == "MHEALTH":
            from dataset_process.dataset_mhealth import create_val_loader
            # MHEALTH默认按受试者分割，如果要改成分层分割则需要修改test_subjects
            test_subjects = ['subject9', 'subject10'] if dataset_config.split_type == "subject_independent" else None
            return create_val_loader(
                data_file=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                step_size=dataset_config.step_size,
                exclude_null=dataset_config.exclude_null,
                test_subjects=test_subjects
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_config.name}")
    
    @staticmethod
    def create_test_loader(dataset_config: DatasetConfig) -> torch.utils.data.DataLoader:
        """
        创建测试数据加载器
        
        Args:
            dataset_config: 数据集配置
            
        Returns:
            测试数据加载器
        """
        if dataset_config.name == "UCIHAR":
            from dataset_process.dataset_UCIHAR import create_test_loader
            test_loader, _ = create_test_loader(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size
            )
            return test_loader
        elif dataset_config.name == "WISDM":
            from dataset_process.dataset_WISDM import create_test_loader
            # WISDM只支持分层分割，不支持按用户分割
            test_loader, _ = create_test_loader(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size
                # WISDM不支持user_split参数，移除此参数
            )
            return test_loader
        elif dataset_config.name == "PAMAP2":
            from dataset_process.dataset_PAMAP2 import create_test_loader
            # PAMAP2只支持分层分割，不支持按受试者分割
            test_loader, _ = create_test_loader(
                data_dir=dataset_config.data_dir,
                batch_size=dataset_config.batch_size
                # PAMAP2不支持subject_split参数，移除此参数
            )
            return test_loader
        elif dataset_config.name == "MHEALTH":
            from dataset_process.dataset_mhealth import create_test_loader
            # MHEALTH默认按受试者分割，如果要改成分层分割则需要修改test_subjects
            test_subjects = ['subject9', 'subject10'] if dataset_config.split_type == "subject_independent" else None
            test_loader, _ = create_test_loader(
                data_path=dataset_config.data_dir,
                batch_size=dataset_config.batch_size,
                step_size=dataset_config.step_size,
                exclude_null=dataset_config.exclude_null,
                test_subjects=test_subjects
            )
            return test_loader
        else:
            raise ValueError(f"Unsupported dataset: {dataset_config.name}")
    
    @staticmethod
    def get_class_names(dataset_config: DatasetConfig) -> list:
        """
        获取数据集类别名称
        
        Args:
            dataset_config: 数据集配置
            
        Returns:
            类别名称列表
        """
        if dataset_config.name == "PAMAP2":
            from dataset_process.dataset_PAMAP2 import ACTIVITY_NAMES
            return ACTIVITY_NAMES
        elif dataset_config.name == "MHEALTH":
            from dataset_process.dataset_mhealth import MHEALTH_ACTIVITY_LABELS
            return [MHEALTH_ACTIVITY_LABELS[i+1] for i in range(dataset_config.num_classes)]
        else:
            return [f"Class {i}" for i in range(dataset_config.num_classes)]
    
    @staticmethod
    def print_dataset_info(dataset_config: DatasetConfig):
        """
        打印数据集信息
        
        Args:
            dataset_config: 数据集配置
        """
        print(f"🎯 使用数据集: {dataset_config.name}")
        print(f"   - 输入通道数: {dataset_config.in_channels}")
        print(f"   - 类别数: {dataset_config.num_classes}")
        print(f"   - 时间窗口: {dataset_config.input_length} 时间步")
        
        if dataset_config.name == "UCIHAR":
            print(f"   - 采样频率: 50Hz")
            print(f"   - 时间长度: {dataset_config.input_length/50:.2f}s")
        elif dataset_config.name == "WISDM":
            print(f"   - 采样频率: ~20Hz")
            print(f"   - 窗口大小: 128 时间步（50%重叠）")
            print(f"   - 活动类型: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing")
            print(f"   - 用户数量: 36个用户")
        elif dataset_config.name == "PAMAP2":
            print(f"   - 采样频率: 100Hz")
            print(f"   - 窗口大小: 256 时间步（50%重叠）")
            print(f"   - 时间长度: {dataset_config.input_length/100:.2f}s")
            print(f"   - 传感器: 3个IMU (加速度计+陀螺仪)")
        elif dataset_config.name == "MHEALTH":
            print(f"   - 传感器: 胸部+右手腕 (加速度计+陀螺仪)")
            print(f"   - 过滤: 排除空值/过渡状态 (类别0)")
            if dataset_config.step_size:
                print(f"   - 步长: {dataset_config.step_size}")