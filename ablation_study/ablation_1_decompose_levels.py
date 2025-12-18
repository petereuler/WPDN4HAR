#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验1：验证层级分解结构的有效性
测试分解级数为0,1,2,3,4的模型性能，固定并行分解组数为1
分解级数为0时，相当于直接对输入数据做分类
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from utils.config import Config, DatasetConfig, ModelConfig, TrainingConfig
from utils.dataset_utils import DatasetLoader
from ablation_utils import train_model, evaluate_model
from model.model_wpdn import LightweightWaveletPacketCNN
from model.baselines.cnn_models import LightweightCNN


class DirectClassifier(nn.Module):
    """
    直接分类器（分解级数为0）
    使用WPDN的ultra_lightweight分类器结构，直接对输入数据进行分类
    """
    def __init__(self, in_channels=6, num_classes=6, input_length=128):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        
        # 将1D输入转换为2D时频图格式以适配ultra_lightweight分类器
        # 假设时频图形状为 (in_channels, input_length)，相当于频率维度为in_channels，时间维度为input_length
        time_freq_shape = (in_channels, input_length)
        
        # 使用WPDN的UltraLightweightConv2DClassifier
        from model.model_wpdn import UltraLightweightConv2DClassifier
        self.classifier = UltraLightweightConv2DClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            time_freq_shape=time_freq_shape,
            use_parallel=False,
            num_parallel_groups=1
        )
    
    def forward(self, x):
        # x: [B, C, T] -> [B, C, C, T] (将输入重塑为2D时频图格式)
        # 这里我们将输入数据重塑为适合2D卷积的格式
        B, C, T = x.shape
        # 将1D信号转换为2D时频图：[B, C, T] -> [B, C, 1, T] -> [B, C, C, T]
        x = x.unsqueeze(2).expand(B, C, C, T)  # [B, C, C, T]
        return self.classifier(x)
    
    def get_orthogonality_loss(self):
        """返回0，保持接口一致"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


def create_model_for_decompose_level(decompose_levels, in_channels, num_classes, input_length):
    """
    根据分解级数创建对应的模型
    """
    if decompose_levels == 0:
        # 分解级数为0，使用直接分类器
        return DirectClassifier(in_channels, num_classes, input_length)
    else:
        # 根据分解级数调整卷积核大小和并行组数
        if decompose_levels == 1:
            kernel_size = 2  # 分解级数1时使用卷积核大小2
            num_parallel_groups = 1  # 分解级数1时不使用并行分解
            use_parallel = False
        elif decompose_levels == 2:
            kernel_size = 3
            num_parallel_groups = 2
            use_parallel = True
        elif decompose_levels in [3, 4]:
            kernel_size = 3
            num_parallel_groups = 3
            use_parallel = True
        else:
            kernel_size = 4
            num_parallel_groups = 3
            use_parallel = True
            
        # 使用WPDN模型
        return LightweightWaveletPacketCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            input_length=input_length,
            kernel_size=kernel_size,
            classifier_type="ultra_lightweight",
            use_parallel=use_parallel,
            num_parallel_groups=num_parallel_groups,
            use_traditional_wavelet=False,  # 使用可学习小波
            decompose_levels=decompose_levels,
            verbose=False
        )


def run_ablation_experiment():
    """
    运行层级分解结构有效性消融实验
    """
    print("=" * 80)
    print("消融实验1：验证层级分解结构的有效性")
    print("测试分解级数：0, 1, 2, 3, 4")
    print("固定并行分解组数：3")
    print("=" * 80)
    
    # 数据集配置
    dataset_name = "MHEALTH"
    dataset_config = Config.get_dataset_config(dataset_name)
    
    # 训练配置
    training_config = TrainingConfig(
        epochs=50,  # 训练轮数
        learning_rate=0.001,  # 固定学习率
        weight_decay=1e-4,
        orth_weight=0.1  # 正交损失权重
    )
    
    # 设备配置
    device = Config.setup_device()
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"\n加载数据集: {dataset_name}")
    train_loader = DatasetLoader.create_train_loader(dataset_config)
    val_loader = DatasetLoader.create_val_loader(dataset_config)
    test_loader = DatasetLoader.create_test_loader(dataset_config)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 测试不同的分解级数
    decompose_levels_list = [0, 1, 2, 3, 4]
    results = {}
    
    for decompose_levels in decompose_levels_list:
        print(f"\n{'='*60}")
        print(f"测试分解级数: {decompose_levels}")
        print(f"{'='*60}")
        
        # 创建模型
        model = create_model_for_decompose_level(
            decompose_levels, 
            dataset_config.in_channels, 
            dataset_config.num_classes, 
            dataset_config.input_length
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型类型: {'直接分类器' if decompose_levels == 0 else 'WPDN'}")
        print(f"分解级数: {decompose_levels}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 训练模型
        model_config = ModelConfig(
            mode=f"ablation_decompose_levels_{decompose_levels}",
            decompose_levels=decompose_levels,
            num_parallel_groups=1,
            device=str(device)
        )
        
        try:
            # 训练
            train_history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                device=device,
                save_path=None,  # 不保存模型
                use_orthogonality_loss=(decompose_levels > 0)  # 只有使用小波分解时才使用正交损失
            )
            
            # 评估
            test_metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device
            )
            
            # 记录结果
            results[decompose_levels] = {
                "model_type": "直接分类器" if decompose_levels == 0 else "WPDN",
                "decompose_levels": decompose_levels,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "best_val_acc": max(train_history["val_acc"]),
                "best_val_loss": min(train_history["val_loss"]),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "final_train_loss": train_history["train_loss"][-1],
                "final_val_loss": train_history["val_loss"][-1],
                "epochs_trained": len(train_history["train_loss"])
            }
            
            print(f"\n训练完成!")
            print(f"最佳验证准确率: {results[decompose_levels]['best_val_acc']:.4f}")
            print(f"测试准确率: {results[decompose_levels]['test_accuracy']:.4f}")
            print(f"测试F1分数: {results[decompose_levels]['test_f1']:.4f}")
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            results[decompose_levels] = {
                "error": str(e),
                "model_type": "直接分类器" if decompose_levels == 0 else "WPDN",
                "decompose_levels": decompose_levels,
                "total_params": total_params,
                "trainable_params": trainable_params
            }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_1_decompose_levels_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("消融实验1结果汇总")
    print(f"{'='*80}")
    
    # 打印结果表格
    print(f"{'分解级数':<8} {'模型类型':<12} {'参数量':<10} {'验证准确率':<12} {'测试准确率':<12} {'测试F1':<10}")
    print("-" * 80)
    
    for level in decompose_levels_list:
        if "error" not in results[level]:
            result = results[level]
            print(f"{level:<8} {result['model_type']:<12} {result['total_params']:<10,} "
                  f"{result['best_val_acc']:<12.4f} {result['test_accuracy']:<12.4f} {result['test_f1']:<10.4f}")
        else:
            print(f"{level:<8} {'错误':<12} {'-':<10} {'-':<12} {'-':<12} {'-':<10}")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 分析结果
    print(f"\n{'='*80}")
    print("结果分析")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) > 1:
        # 找到最佳性能
        best_level = max(valid_results.keys(), key=lambda x: valid_results[x]["test_accuracy"])
        best_acc = valid_results[best_level]["test_accuracy"]
        
        print(f"最佳分解级数: {best_level}")
        print(f"最佳测试准确率: {best_acc:.4f}")
        
        # 与基线（分解级数0）对比
        if 0 in valid_results:
            baseline_acc = valid_results[0]["test_accuracy"]
            improvement = best_acc - baseline_acc
            print(f"相比直接分类器提升: {improvement:.4f} ({improvement/baseline_acc*100:.2f}%)")
    
    return results


if __name__ == "__main__":
    results = run_ablation_experiment()