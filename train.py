"""
重构后的训练主函数
简洁清晰，只保留主要流程控制
"""

import torch
import sys
import os

# 导入重构后的模块
from utils.config import Config, TrainingConfig, DatasetConfig, ModelConfig
from utils.model_factory import ModelFactory
from utils.dataset_utils import DatasetLoader
from training_utils import Trainer


def main():
    """主训练函数"""
    
    # ==================== 配置参数 ====================
    # 训练模式配置 - 支持以下模式
    MODE = 'wavelet_lite'  # Auto-modified by quick_test_all_models.py
                       # 'wavelet_traditional': 传统小波包变换 
                       # 'wavelet_lite': 轻量化小波包变换 
                       # 标准模型: 'lstm', 'gru', 'transformer', 'cnn', 'resnet'
                       # 轻量化模型: 'lstm_lite', 'gru_lite', 'transformer_lite', 'cnn_lite', 'resnet_lite'
    
    # 小波参数（仅在小波模式下使用）
    WAVELET_TYPE = "db4"  # 小波基类型
    WAVELET_LEVELS = 3    # 传统小波分解层数
    DECOMPOSE_LEVELS = 3  # 可学习小波分解级数
    NUM_PARALLEL_GROUPS = 4  # 并行分解组数
    
    # 数据集配置
    DATASET_TYPE = 'UCIHAR'  # Auto-modified for quick testing

    # 数据集划分类型配置
    # "stratified": 分层分割（保持类别平衡）
    # "subject_dependent": 依赖于受试者（按用户分割）
    # "subject_independent": 独立于受试者（用户间混合分割）
    SPLIT_TYPE = 'stratified'  # 可以改为 'subject_dependent' 或 'subject_independent'

    # 训练超参数
    EPOCHS = 50  # Auto-modified for quick testing
    BATCH_SIZE = 32  # Auto-modified for quick testing
    LEARNING_RATE = 0.001  # Auto-modified for quick testing
    WEIGHT_DECAY = 1e-4
    OPTIMIZER_TYPE = "Adam"  # "Adam", "SGD"
    ORTH_WEIGHT = 0.01  # 正交损失权重（仅用于可学习小波）
    
    # 设备配置
    DEVICE = "auto"  # "auto", "cuda", "cpu"
    
    # MHEALTH数据集特定参数
    MHEALTH_STEP_SIZE = 64
    MHEALTH_EXCLUDE_NULL = True
    
    # ==================== 初始化配置 ====================
    print(f"\n{'='*60}")
    print(f"小波包神经网络训练系统")
    print(f"{'='*60}")
    
    # 设备配置
    device = Config.setup_device(DEVICE)
    
    # 数据集配置
    dataset_config = Config.get_dataset_config(DATASET_TYPE)
    if DATASET_TYPE == "MHEALTH":
        dataset_config.step_size = MHEALTH_STEP_SIZE
        dataset_config.exclude_null = MHEALTH_EXCLUDE_NULL

    # 设置数据集划分类型
    dataset_config.split_type = SPLIT_TYPE
    
    # 模型配置
    model_config = ModelConfig(
        mode=MODE,
        wavelet_type=WAVELET_TYPE,
        wavelet_levels=WAVELET_LEVELS,
        decompose_levels=DECOMPOSE_LEVELS,
        num_parallel_groups=NUM_PARALLEL_GROUPS
    )
    
    # 训练配置
    training_config = TrainingConfig(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optimizer_type=OPTIMIZER_TYPE,
        orth_weight=ORTH_WEIGHT
    )
    
    # 打印配置信息
    Config.print_config_summary(dataset_config, model_config, training_config)
    
    # ==================== 数据加载 ====================
    print(f"\n📊 Loading dataset...")
    
    # 创建数据加载器
    train_loader = DatasetLoader.create_train_loader(dataset_config)
    val_loader = DatasetLoader.create_val_loader(dataset_config)
    
    # 打印数据集信息
    DatasetLoader.print_dataset_info(dataset_config)
    
    # ==================== 模型创建 ====================
    print(f"\n🏗️ Creating model...")
    
    # 创建模型
    model = ModelFactory.create_model(model_config.mode, dataset_config, model_config, device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数数量: {total_params:,}")
    print(f"📊 架构描述: {Config.get_architecture_description(model_config.mode)}")
    print(f"📊 特征提取: {Config.get_feature_extraction_description(model_config.mode)}")
    
    # ==================== 训练执行 ====================
    print(f"\n🚀 Starting training...")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        dataset_config=dataset_config,
        model_config=model_config,
        device=device
    )
    
    # 执行训练
    try:
        training_results = trainer.train()
        
        # 打印训练结果
        print(f"\n🎉 训练完成!")
        print(f"📊 全局最佳验证精度: {training_results['best_acc']:.2f}%")
        print(f"📊 最后5轮最佳验证精度: {training_results['best_acc_last_epochs']:.2f}%")
        
        # 保存训练历史
        checkpoint_path = Config.get_model_checkpoint_path(model_config.mode, dataset_config.name)
        print(f"💾 最佳模型已保存至: {checkpoint_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {str(e)}")
        raise e
    
    print(f"\n✅ 训练流程完成!")


if __name__ == "__main__":
    main()