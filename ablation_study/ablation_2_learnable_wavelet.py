#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验2：验证可学习小波的有效性
对比WPDN（可学习小波）与传统小波包CNN的性能
固定分解级数为3，并行分解组数为4
测试多种传统小波基函数
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
from model.traditional_wavelet_packet import TraditionalWaveletPacketCNN, SUPPORTED_WAVELETS


def create_model_for_wavelet_type(wavelet_type, in_channels, num_classes, input_length, use_learnable=True, kernel_size=4):
    """
    根据小波类型创建对应的模型
    
    Args:
        wavelet_type: 小波类型，如果use_learnable=True则忽略此参数
        in_channels: 输入通道数
        num_classes: 类别数
        input_length: 输入长度
        use_learnable: 是否使用可学习小波
        kernel_size: 卷积核大小（仅用于可学习小波）
    """
    if use_learnable:
        # 使用WPDN（可学习小波）
        return LightweightWaveletPacketCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            input_length=input_length,
            kernel_size=kernel_size,
            classifier_type="ultra_lightweight",
            use_parallel=True,  # 使用并行分解
            num_parallel_groups=4,  # 改为4组
            use_traditional_wavelet=False,  # 使用可学习小波
            decompose_levels=3,  # 3级分解
            verbose=False
        )
    else:
        # 使用传统小波包CNN
        return TraditionalWaveletPacketCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            input_length=input_length,
            wavelet=wavelet_type,
            levels=3,  # 3级分解
            classifier_type='ultra_lightweight'  # 使用ultra_lightweight分类器
        )


def run_ablation_experiment():
    """
    运行可学习小波有效性消融实验
    """
    print("=" * 80)
    print("消融实验2：验证可学习小波的有效性")
    print("对比WPDN（可学习小波）与传统小波包CNN")
    print("固定分解级数：3，并行分解组数：4")
    print("=" * 80)
    
    # 数据集配置
    dataset_name = "MHEALTH"
    dataset_config = Config.get_dataset_config(dataset_name)
    
    # 训练配置
    training_config = TrainingConfig(
        epochs=25,  # 训练轮数
        learning_rate=0.001,  # 固定学习率
        weight_decay=1e-4,
        orth_weight=0.01  # 正交损失权重改为0.01
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
    
    # 测试的小波类型
    test_wavelets = [
        ('learnable_4', 'WPDN可学习小波(窗长4)'),
        ('learnable_6', 'WPDN可学习小波(窗长6)'),
        ('learnable_8', 'WPDN可学习小波(窗长8)'),
        ('db1', 'Daubechies 1 (Haar)'),
        ('db4', 'Daubechies 4'),
        ('bior2.2', 'Biorthogonal 2.2'),
        ('coif2', 'Coiflets 2'),
        ('sym4', 'Symlets 4')
    ]
    
    results = {}
    
    for wavelet_type, wavelet_name in test_wavelets:
        print(f"\n{'='*60}")
        print(f"测试小波类型: {wavelet_name} ({wavelet_type})")
        print(f"{'='*60}")
        
        # 创建模型
        use_learnable = wavelet_type.startswith('learnable')
        if use_learnable:
            # 提取卷积核大小
            kernel_size = int(wavelet_type.split('_')[1])
        else:
            kernel_size = 4  # 传统小波使用默认值
            
        model = create_model_for_wavelet_type(
            wavelet_type, 
            dataset_config.in_channels, 
            dataset_config.num_classes, 
            dataset_config.input_length,
            use_learnable=use_learnable,
            kernel_size=kernel_size
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型类型: {'WPDN (可学习小波)' if use_learnable else '传统小波包CNN'}")
        print(f"小波类型: {wavelet_name}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 训练模型
        model_config = ModelConfig(
            mode=f"ablation_wavelet_{wavelet_type}",
            wavelet_type=wavelet_type if not use_learnable else "learnable",
            decompose_levels=3,
            num_parallel_groups=4 if use_learnable else 1,  # 可学习小波使用4，传统小波使用1
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
                use_orthogonality_loss=use_learnable  # 只有可学习小波才使用正交损失
            )
            
            # 评估
            test_metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device
            )
            
            # 记录结果
            results[wavelet_type] = {
                "wavelet_type": wavelet_type,
                "wavelet_name": wavelet_name,
                "model_type": "WPDN (可学习小波)" if use_learnable else "传统小波包CNN",
                "use_learnable": use_learnable,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "best_val_acc": max(train_history["val_acc"]),
                "best_val_loss": min(train_history["val_loss"]),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "avg_inference_time_ms": test_metrics["avg_inference_time_ms"],
                "final_train_loss": train_history["train_loss"][-1],
                "final_val_loss": train_history["val_loss"][-1],
                "epochs_trained": len(train_history["train_loss"])
            }
            
            print(f"\n训练完成!")
            print(f"最佳验证准确率: {results[wavelet_type]['best_val_acc']:.4f}")
            print(f"测试准确率: {results[wavelet_type]['test_accuracy']:.4f}")
            print(f"测试F1分数: {results[wavelet_type]['test_f1']:.4f}")
            print(f"平均推理时间: {results[wavelet_type]['avg_inference_time_ms']:.2f} ms")
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            results[wavelet_type] = {
                "error": str(e),
                "wavelet_type": wavelet_type,
                "wavelet_name": wavelet_name,
                "model_type": "WPDN (可学习小波)" if use_learnable else "传统小波包CNN",
                "use_learnable": use_learnable,
                "total_params": total_params,
                "trainable_params": trainable_params
            }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_2_learnable_wavelet_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("消融实验2结果汇总")
    print(f"{'='*80}")
    
    # 打印结果表格
    print(f"{'小波类型':<15} {'模型类型':<20} {'参数量':<10} {'验证准确率':<12} {'测试准确率':<12} {'测试F1':<10} {'推理时间(ms)':<12}")
    print("-" * 120)
    
    for wavelet_type, wavelet_name in test_wavelets:
        if wavelet_type in results and "error" not in results[wavelet_type]:
            result = results[wavelet_type]
            print(f"{wavelet_type:<15} {result['model_type']:<20} {result['total_params']:<10,} "
                  f"{result['best_val_acc']:<12.4f} {result['test_accuracy']:<12.4f} {result['test_f1']:<10.4f} "
                  f"{result['avg_inference_time_ms']:<12.2f}")
        else:
            print(f"{wavelet_type:<15} {'错误':<20} {'-':<10} {'-':<12} {'-':<12} {'-':<10} {'-':<12}")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 分析结果
    print(f"\n{'='*80}")
    print("结果分析")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) > 1:
        # 找到最佳性能
        best_wavelet = max(valid_results.keys(), key=lambda x: valid_results[x]["test_accuracy"])
        best_acc = valid_results[best_wavelet]["test_accuracy"]
        
        print(f"最佳小波类型: {valid_results[best_wavelet]['wavelet_name']} ({best_wavelet})")
        print(f"最佳测试准确率: {best_acc:.4f}")
        
        # 可学习小波与传统小波对比
        learnable_results = {k: v for k, v in valid_results.items() if k.startswith('learnable')}
        traditional_results = {k: v for k, v in valid_results.items() if not k.startswith('learnable')}
        
        if learnable_results and traditional_results:
            # 找到最佳可学习小波
            best_learnable = max(learnable_results.keys(), key=lambda x: learnable_results[x]["test_accuracy"])
            best_learnable_acc = learnable_results[best_learnable]["test_accuracy"]
            best_learnable_time = learnable_results[best_learnable]["avg_inference_time_ms"]
            
            print(f"\n最佳可学习小波: {learnable_results[best_learnable]['wavelet_name']} ({best_learnable})")
            print(f"最佳可学习小波准确率: {best_learnable_acc:.4f}")
            print(f"最佳可学习小波推理时间: {best_learnable_time:.2f} ms")
            
            # 与最佳传统小波对比
            best_traditional = max(traditional_results.keys(), key=lambda x: traditional_results[x]["test_accuracy"])
            best_traditional_acc = traditional_results[best_traditional]["test_accuracy"]
            best_traditional_time = traditional_results[best_traditional]["avg_inference_time_ms"]
            
            print(f"最佳传统小波: {traditional_results[best_traditional]['wavelet_name']} ({best_traditional})")
            print(f"最佳传统小波准确率: {best_traditional_acc:.4f}")
            print(f"最佳传统小波推理时间: {best_traditional_time:.2f} ms")
            
            improvement = best_learnable_acc - best_traditional_acc
            time_diff = best_learnable_time - best_traditional_time
            print(f"可学习小波相比最佳传统小波提升: {improvement:.4f} ({improvement/best_traditional_acc*100:.2f}%)")
            print(f"推理时间差异: {time_diff:.2f} ms ({time_diff/best_traditional_time*100:.2f}%)")
            
            # 平均性能对比
            learnable_avg_acc = np.mean([v["test_accuracy"] for v in learnable_results.values()])
            traditional_avg_acc = np.mean([v["test_accuracy"] for v in traditional_results.values()])
            learnable_avg_time = np.mean([v["avg_inference_time_ms"] for v in learnable_results.values()])
            traditional_avg_time = np.mean([v["avg_inference_time_ms"] for v in traditional_results.values()])
            
            avg_improvement = learnable_avg_acc - traditional_avg_acc
            avg_time_diff = learnable_avg_time - traditional_avg_time
            print(f"可学习小波平均准确率: {learnable_avg_acc:.4f}")
            print(f"传统小波平均准确率: {traditional_avg_acc:.4f}")
            print(f"可学习小波相比传统小波平均提升: {avg_improvement:.4f} ({avg_improvement/traditional_avg_acc*100:.2f}%)")
            print(f"可学习小波平均推理时间: {learnable_avg_time:.2f} ms")
            print(f"传统小波平均推理时间: {traditional_avg_time:.2f} ms")
            print(f"平均推理时间差异: {avg_time_diff:.2f} ms ({avg_time_diff/traditional_avg_time*100:.2f}%)")
            
            # 不同窗长的可学习小波对比
            print(f"\n可学习小波窗长对比:")
            for wavelet_type in sorted(learnable_results.keys()):
                result = learnable_results[wavelet_type]
                kernel_size = wavelet_type.split('_')[1]
                print(f"窗长{kernel_size}: 准确率{result['test_accuracy']:.4f}, 推理时间{result['avg_inference_time_ms']:.2f}ms")
        
        # 传统小波性能排序
        if traditional_results:
            print(f"\n传统小波性能排序:")
            traditional_sorted = sorted(
                [(k, v) for k, v in traditional_results.items()],
                key=lambda x: x[1]["test_accuracy"],
                reverse=True
            )
            for i, (wavelet_type, result) in enumerate(traditional_sorted, 1):
                print(f"{i}. {result['wavelet_name']}: 准确率{result['test_accuracy']:.4f}, 推理时间{result['avg_inference_time_ms']:.2f}ms")
    
    return results


if __name__ == "__main__":
    results = run_ablation_experiment()