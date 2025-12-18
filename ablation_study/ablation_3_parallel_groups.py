#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验3：验证并行分解结构的有效性
测试不同并行分解组数（1,2,3,4）对WPDN性能的影响
固定分解级数为3
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from config import Config, DatasetConfig, ModelConfig, TrainingConfig
from dataset_utils import DatasetLoader
from ablation_utils import train_model, evaluate_model
from model.model_wpdn import LightweightWaveletPacketCNN


def create_model_for_parallel_groups(num_parallel_groups, in_channels, num_classes, input_length):
    """
    根据并行分解组数创建对应的WPDN模型
    
    Args:
        num_parallel_groups: 并行分解组数
        in_channels: 输入通道数
        num_classes: 类别数
        input_length: 输入长度
    """
    use_parallel = (num_parallel_groups > 1)
    
    return LightweightWaveletPacketCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_length=input_length,
        kernel_size=6,  # 固定小波卷积核长度为6
        classifier_type="ultra_lightweight",
        use_parallel=use_parallel,
        num_parallel_groups=num_parallel_groups,
        use_traditional_wavelet=False,  # 使用可学习小波
        decompose_levels=3,  # 固定分解级数为3
        verbose=False
    )


def run_ablation_experiment():
    """
    运行并行分解结构有效性消融实验
    """
    print("=" * 80)
    print("消融实验3：验证并行分解结构的有效性")
    print("测试并行分解组数：1, 2, 3, 4")
    print("固定分解级数：3")
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
    
    # 测试不同的并行分解组数
    parallel_groups_list = [1, 2, 3, 4]
    results = {}
    
    for num_parallel_groups in parallel_groups_list:
        print(f"\n{'='*60}")
        print(f"测试并行分解组数: {num_parallel_groups}")
        print(f"{'='*60}")
        
        # 创建模型
        model = create_model_for_parallel_groups(
            num_parallel_groups, 
            dataset_config.in_channels, 
            dataset_config.num_classes, 
            dataset_config.input_length
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算小波分解器参数量
        if hasattr(model, 'decomposer'):
            wavelet_params = sum(p.numel() for p in model.decomposer.parameters())
        elif hasattr(model, 'parallel_decomposer'):
            wavelet_params = sum(p.numel() for p in model.parallel_decomposer.parameters())
        else:
            wavelet_params = 0
        
        # 计算分类器参数量
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        
        print(f"并行分解组数: {num_parallel_groups}")
        print(f"使用并行结构: {'是' if num_parallel_groups > 1 else '否'}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"小波分解器参数量: {wavelet_params:,}")
        print(f"分类器参数量: {classifier_params:,}")
        
        # 训练模型
        model_config = ModelConfig(
            mode=f"ablation_parallel_groups_{num_parallel_groups}",
            decompose_levels=3,
            num_parallel_groups=num_parallel_groups,
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
                use_orthogonality_loss=True  # 使用正交损失
            )
            
            # 评估
            test_metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device
            )
            
            # 记录结果
            results[num_parallel_groups] = {
                "num_parallel_groups": num_parallel_groups,
                "use_parallel": num_parallel_groups > 1,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "wavelet_params": wavelet_params,
                "classifier_params": classifier_params,
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
            print(f"最佳验证准确率: {results[num_parallel_groups]['best_val_acc']:.4f}")
            print(f"测试准确率: {results[num_parallel_groups]['test_accuracy']:.4f}")
            print(f"测试F1分数: {results[num_parallel_groups]['test_f1']:.4f}")
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            results[num_parallel_groups] = {
                "error": str(e),
                "num_parallel_groups": num_parallel_groups,
                "use_parallel": num_parallel_groups > 1,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "wavelet_params": wavelet_params,
                "classifier_params": classifier_params
            }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_3_parallel_groups_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("消融实验3结果汇总")
    print(f"{'='*80}")
    
    # 打印结果表格
    print(f"{'并行组数':<8} {'并行结构':<8} {'总参数量':<10} {'小波参数':<10} {'验证准确率':<12} {'测试准确率':<12} {'测试F1':<10}")
    print("-" * 90)
    
    for num_groups in parallel_groups_list:
        if "error" not in results[num_groups]:
            result = results[num_groups]
            parallel_str = "是" if result['use_parallel'] else "否"
            print(f"{num_groups:<8} {parallel_str:<8} {result['total_params']:<10,} {result['wavelet_params']:<10,} "
                  f"{result['best_val_acc']:<12.4f} {result['test_accuracy']:<12.4f} {result['test_f1']:<10.4f}")
        else:
            print(f"{num_groups:<8} {'错误':<8} {'-':<10} {'-':<10} {'-':<12} {'-':<12} {'-':<10}")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 分析结果
    print(f"\n{'='*80}")
    print("结果分析")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) > 1:
        # 找到最佳性能
        best_groups = max(valid_results.keys(), key=lambda x: valid_results[x]["test_accuracy"])
        best_acc = valid_results[best_groups]["test_accuracy"]
        
        print(f"最佳并行分解组数: {best_groups}")
        print(f"最佳测试准确率: {best_acc:.4f}")
        
        # 与基线（组数为1）对比
        if 1 in valid_results:
            baseline_acc = valid_results[1]["test_accuracy"]
            baseline_params = valid_results[1]["total_params"]
            
            print(f"\n基线性能（组数=1）:")
            print(f"测试准确率: {baseline_acc:.4f}")
            print(f"总参数量: {baseline_params:,}")
            
            if best_groups != 1:
                improvement = best_acc - baseline_acc
                best_params = valid_results[best_groups]["total_params"]
                param_increase = best_params - baseline_params
                
                print(f"\n最佳配置相比基线:")
                print(f"准确率提升: {improvement:.4f} ({improvement/baseline_acc*100:.2f}%)")
                print(f"参数量增加: {param_increase:,} ({param_increase/baseline_params*100:.2f}%)")
                print(f"参数效率: {improvement/(param_increase/1000):.6f} (准确率提升/千参数增加)")
        
        # 性能趋势分析
        print(f"\n性能趋势分析:")
        sorted_groups = sorted(valid_results.keys())
        for i, groups in enumerate(sorted_groups):
            result = valid_results[groups]
            if i > 0:
                prev_result = valid_results[sorted_groups[i-1]]
                acc_change = result["test_accuracy"] - prev_result["test_accuracy"]
                param_change = result["total_params"] - prev_result["total_params"]
                print(f"组数{sorted_groups[i-1]}→{groups}: 准确率变化{acc_change:+.4f}, 参数变化{param_change:+,}")
            else:
                print(f"组数{groups}: 准确率{result['test_accuracy']:.4f}, 参数量{result['total_params']:,}")
        
        # 参数效率分析
        print(f"\n参数效率分析:")
        for groups in sorted(valid_results.keys()):
            result = valid_results[groups]
            efficiency = result["test_accuracy"] / (result["total_params"] / 1000)  # 准确率/千参数
            wavelet_efficiency = result["test_accuracy"] / (result["wavelet_params"] / 1000) if result["wavelet_params"] > 0 else 0
            print(f"组数{groups}: 总体效率{efficiency:.6f}, 小波效率{wavelet_efficiency:.6f}")
        
        # 小波参数量分析
        print(f"\n小波参数量分析:")
        for groups in sorted(valid_results.keys()):
            result = valid_results[groups]
            wavelet_ratio = result["wavelet_params"] / result["total_params"] * 100
            print(f"组数{groups}: 小波参数{result['wavelet_params']:,} ({wavelet_ratio:.1f}%)")
        
        # 收益递减分析
        print(f"\n收益递减分析:")
        if len(valid_results) >= 2:
            sorted_results = [(k, v) for k, v in sorted(valid_results.items())]
            for i in range(1, len(sorted_results)):
                curr_groups, curr_result = sorted_results[i]
                prev_groups, prev_result = sorted_results[i-1]
                
                acc_gain = curr_result["test_accuracy"] - prev_result["test_accuracy"]
                param_cost = curr_result["total_params"] - prev_result["total_params"]
                
                if param_cost > 0:
                    cost_benefit = acc_gain / (param_cost / 1000)  # 准确率提升/千参数成本
                    print(f"组数{prev_groups}→{curr_groups}: 成本效益比{cost_benefit:.6f}")
    
    return results


if __name__ == "__main__":
    results = run_ablation_experiment()