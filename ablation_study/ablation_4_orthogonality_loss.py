#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验4：验证完美重构（正交损失）的有效性
测试不同正交损失权重（0, 0.01, 0.1, 1）对WPDN性能的影响
固定分解级数为3，并行分解组数为4
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import csv
from config import Config, DatasetConfig, ModelConfig, TrainingConfig
from dataset_utils import DatasetLoader
from ablation_utils import train_model, evaluate_model
from model.model_wpdn import LightweightWaveletPacketCNN
import glob


def create_wpdn_model(in_channels, num_classes, input_length):
    """
    创建WPDN模型，固定配置
    """
    return LightweightWaveletPacketCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_length=input_length,
        kernel_size=6,  # 滤波器核长度设置为6
        classifier_type="ultra_lightweight",
        use_parallel=True,  # 使用并行分解
        num_parallel_groups=3,  # 改为3个并行分解结构
        use_traditional_wavelet=False,  # 使用可学习小波
        decompose_levels=3,  # 3级分解
        verbose=False
    )


def run_ablation_experiment():
    """
    运行完美重构（正交损失）有效性消融实验
    """
    print("=" * 80)
    print("消融实验4：验证完美重构（正交损失）的有效性")
    print("测试正交损失权重：0, 0.01, 0.1, 1，并扩展多个中间取值")
    print("固定分解级数：3，并行分解组数：3，滤波器核长度：6")
    print("=" * 80)
    
    # 数据集配置
    dataset_name = "MHEALTH"
    dataset_config = Config.get_dataset_config(dataset_name)
    
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
    
    # 测试不同的正交损失权重
    # 扩展权重扫描：在三个区间内取多点
    orth_weights = [
        # 0-0.01
        0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01,
        # 0.01-0.1
        0.02, 0.05, 0.08, 0.1,
        # 0.1-1
        0.2, 0.5, 0.8, 1.0
    ]
    results = {}
    
    for orth_weight in orth_weights:
        print(f"\n{'='*60}")
        print(f"测试正交损失权重: {orth_weight}")
        print(f"{'='*60}")
        
        # 创建模型
        model = create_wpdn_model(
            dataset_config.in_channels, 
            dataset_config.num_classes, 
            dataset_config.input_length
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算小波分解器参数量
        if hasattr(model, 'parallel_decomposer'):
            wavelet_params = sum(p.numel() for p in model.parallel_decomposer.parameters())
        else:
            wavelet_params = 0
        
        # 计算分类器参数量
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        
        print(f"正交损失权重: {orth_weight}")
        print(f"使用正交损失: {'是' if orth_weight > 0 else '否'}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"小波分解器参数量: {wavelet_params:,}")
        print(f"分类器参数量: {classifier_params:,}")
        
        # 训练配置
        training_config = TrainingConfig(
            epochs=50,  # 训练轮数
            learning_rate=0.001,  # 固定学习率
            weight_decay=1e-4,
            orth_weight=orth_weight  # 设置正交损失权重
        )
        
        # 训练模型
        model_config = ModelConfig(
            mode=f"ablation_orth_weight_{orth_weight}",
            decompose_levels=3,
            num_parallel_groups=3,  # 改为3个并行分解结构
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
                use_orthogonality_loss=(orth_weight > 0)  # 根据权重决定是否使用正交损失
            )
            
            # 评估
            test_metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device
            )
            
            # 计算最终的正交损失值（用于分析）
            model.eval()
            with torch.no_grad():
                final_orth_loss = model.get_orthogonality_loss().item()
            
            # 记录结果
            results[orth_weight] = {
                "orth_weight": orth_weight,
                "use_orthogonality_loss": orth_weight > 0,
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
                "final_orthogonality_loss": final_orth_loss,
                "epochs_trained": len(train_history["train_loss"])
            }
            
            # 如果训练历史中包含正交损失，记录其变化
            if train_history.get("orth_loss") is not None:
                results[orth_weight]["orth_loss_history"] = train_history["orth_loss"]
                results[orth_weight]["final_orth_loss_from_history"] = train_history["orth_loss"][-1]
            
            print(f"\n训练完成!")
            print(f"最佳验证准确率: {results[orth_weight]['best_val_acc']:.4f}")
            print(f"测试准确率: {results[orth_weight]['test_accuracy']:.4f}")
            print(f"测试F1分数: {results[orth_weight]['test_f1']:.4f}")
            print(f"最终正交损失: {final_orth_loss:.6f}")
            
        except Exception as e:
            print(f"训练过程中出现错误: {str(e)}")
            results[orth_weight] = {
                "error": str(e),
                "orth_weight": orth_weight,
                "use_orthogonality_loss": orth_weight > 0,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "wavelet_params": wavelet_params,
                "classifier_params": classifier_params
            }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_4_orthogonality_loss_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("消融实验4结果汇总")
    print(f"{'='*80}")
    
    # 打印结果表格
    print(f"{'正交权重':<10} {'使用正交':<8} {'验证准确率':<12} {'测试准确率':<12} {'测试F1':<10} {'正交损失':<12}")
    print("-" * 80)
    
    for orth_weight in orth_weights:
        if "error" not in results[orth_weight]:
            result = results[orth_weight]
            use_orth_str = "是" if result['use_orthogonality_loss'] else "否"
            print(f"{orth_weight:<10} {use_orth_str:<8} {result['best_val_acc']:<12.4f} "
                  f"{result['test_accuracy']:<12.4f} {result['test_f1']:<10.4f} "
                  f"{result['final_orthogonality_loss']:<12.6f}")
        else:
            print(f"{orth_weight:<10} {'错误':<8} {'-':<12} {'-':<12} {'-':<10} {'-':<12}")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 另存CSV，便于后续绘图与分析
    csv_file = f"ablation_4_orthogonality_loss_results_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["orth_weight", "use_orth", "best_val_acc", "test_accuracy", "test_f1", "final_orth_loss"])
        for w in sorted(results.keys()):
            r = results[w]
            if "error" in r:
                writer.writerow([w, r.get("use_orthogonality_loss", w > 0), None, None, None, None])
            else:
                writer.writerow([
                    w,
                    r["use_orthogonality_loss"],
                    r["best_val_acc"],
                    r["test_accuracy"],
                    r["test_f1"],
                    r["final_orthogonality_loss"]
                ])
    print(f"CSV已保存到: {csv_file}")
    
    # 生成趋势图（权重 vs 测试准确率 & 正交损失）
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) > 0:
        sorted_weights = sorted(valid_results.keys())
        accs = [valid_results[w]["test_accuracy"] for w in sorted_weights]
        orths = [valid_results[w]["final_orthogonality_loss"] for w in sorted_weights]
        
        plt.figure(figsize=(8,5), dpi=150)
        ax1 = plt.gca()
        ax1.plot(sorted_weights, accs, marker='o', color='#1f77b4', label='Test Accuracy')
        ax1.set_xlabel('Orthogonality Loss Weight (lambda)', fontsize=14)
        ax1.set_ylabel('Test Accuracy', fontsize=14, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.6, linestyle='--', linewidth=0.8)
        
        ax2 = ax1.twinx()
        ax2.plot(sorted_weights, orths, marker='s', color='#ff7f0e', label='Final Orth Loss')
        ax2.set_ylabel('Final Orthogonality Loss', fontsize=14, color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        plt.title('Ablation4: Effect of lambda on performance', fontsize=16)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best', fontsize=12)
        plt.tight_layout()
        trend_png = f"ablation4_lambda_trend_{timestamp}.png"
        trend_pdf = f"ablation4_lambda_trend_{timestamp}.pdf"
        plt.savefig(trend_png)
        plt.savefig(trend_pdf)
        plt.close()
        print(f"趋势图已保存：{trend_png}, {trend_pdf}")
    
    # 分析结果
    print(f"\n{'='*80}")
    print("结果分析")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) > 1:
        # 找到最佳性能
        best_weight = max(valid_results.keys(), key=lambda x: valid_results[x]["test_accuracy"])
        best_acc = valid_results[best_weight]["test_accuracy"]
        
        print(f"最佳正交损失权重: {best_weight}")
        print(f"最佳测试准确率: {best_acc:.4f}")
        
        # 与基线（权重为0）对比
        if 0.0 in valid_results:
            baseline_acc = valid_results[0.0]["test_accuracy"]
            baseline_orth_loss = valid_results[0.0]["final_orthogonality_loss"]
            
            print(f"\n基线性能（无正交损失）:")
            print(f"测试准确率: {baseline_acc:.4f}")
            print(f"正交损失值: {baseline_orth_loss:.6f}")
            
            if best_weight != 0.0:
                improvement = best_acc - baseline_acc
                best_orth_loss = valid_results[best_weight]["final_orthogonality_loss"]
                
                print(f"\n最佳配置相比基线:")
                print(f"准确率提升: {improvement:.4f} ({improvement/baseline_acc*100:.2f}%)")
                print(f"正交损失变化: {best_orth_loss - baseline_orth_loss:.6f}")
                print(f"正交损失改善: {(baseline_orth_loss - best_orth_loss)/baseline_orth_loss*100:.2f}%")
        
        # 正交损失权重效果分析
        print(f"\n正交损失权重效果分析:")
        sorted_weights = sorted(valid_results.keys())
        for weight in sorted_weights:
            result = valid_results[weight]
            print(f"权重{weight}: 准确率{result['test_accuracy']:.4f}, "
                  f"正交损失{result['final_orthogonality_loss']:.6f}")
        
        # 正交损失与性能的关系分析
        print(f"\n正交损失与性能关系:")
        orth_losses = [valid_results[w]["final_orthogonality_loss"] for w in sorted_weights]
        accuracies = [valid_results[w]["test_accuracy"] for w in sorted_weights]
        
        # 计算相关性（简单的皮尔逊相关系数）
        if len(orth_losses) > 2:
            orth_mean = np.mean(orth_losses)
            acc_mean = np.mean(accuracies)
            
            numerator = sum((orth_losses[i] - orth_mean) * (accuracies[i] - acc_mean) 
                          for i in range(len(orth_losses)))
            denominator = np.sqrt(sum((o - orth_mean)**2 for o in orth_losses) * 
                                sum((a - acc_mean)**2 for a in accuracies))
            
            if denominator != 0:
                correlation = numerator / denominator
                print(f"正交损失与准确率的相关系数: {correlation:.4f}")
                if correlation < -0.5:
                    print("强负相关：正交损失越小，准确率越高")
                elif correlation < -0.3:
                    print("中等负相关：正交损失减小有助于提高准确率")
                elif correlation > 0.3:
                    print("正相关：可能存在过拟合或其他因素")
                else:
                    print("弱相关：正交损失对准确率影响不明显")
        
        # 权重敏感性分析
        print(f"\n权重敏感性分析:")
        if len(valid_results) >= 2:
            sorted_results = [(k, v) for k, v in sorted(valid_results.items())]
            for i in range(1, len(sorted_results)):
                curr_weight, curr_result = sorted_results[i]
                prev_weight, prev_result = sorted_results[i-1]
                
                acc_change = curr_result["test_accuracy"] - prev_result["test_accuracy"]
                orth_change = curr_result["final_orthogonality_loss"] - prev_result["final_orthogonality_loss"]
                weight_change = curr_weight - prev_weight
                
                sensitivity = acc_change / weight_change if weight_change != 0 else 0
                print(f"权重{prev_weight}→{curr_weight}: 准确率变化{acc_change:+.4f}, "
                      f"敏感性{sensitivity:.4f}")
        
        # 最优权重推荐
        print(f"\n最优权重推荐:")
        
        # 基于准确率的推荐
        acc_ranking = sorted(valid_results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True)
        print(f"基于准确率排序:")
        for i, (weight, result) in enumerate(acc_ranking[:3], 1):
            print(f"  {i}. 权重{weight}: 准确率{result['test_accuracy']:.4f}")
        
        # 基于正交损失改善的推荐
        if 0.0 in valid_results:
            baseline_orth = valid_results[0.0]["final_orthogonality_loss"]
            orth_improvement = {}
            for weight, result in valid_results.items():
                if weight > 0:
                    improvement = (baseline_orth - result["final_orthogonality_loss"]) / baseline_orth
                    orth_improvement[weight] = improvement
            
            if orth_improvement:
                orth_ranking = sorted(orth_improvement.items(), key=lambda x: x[1], reverse=True)
                print(f"\n基于正交损失改善排序:")
                for i, (weight, improvement) in enumerate(orth_ranking[:3], 1):
                    acc = valid_results[weight]["test_accuracy"]
                    print(f"  {i}. 权重{weight}: 正交损失改善{improvement*100:.2f}%, 准确率{acc:.4f}")
        
        # 综合推荐
        print(f"\n综合推荐:")
        if best_weight == 0.0:
            print("建议不使用正交损失，因为它没有带来性能提升")
        else:
            print(f"建议使用正交损失权重{best_weight}，可获得最佳性能")
            
            # 检查是否存在性价比更高的权重
            if 0.0 in valid_results:
                baseline_acc = valid_results[0.0]["test_accuracy"]
                for weight in [0.01, 0.1]:
                    if weight in valid_results:
                        weight_acc = valid_results[weight]["test_accuracy"]
                        if weight_acc > baseline_acc and weight < best_weight:
                            improvement = weight_acc - baseline_acc
                            print(f"权重{weight}也是不错的选择，相比无正交损失提升{improvement:.4f}")
    
    return results


if __name__ == "__main__":
    results = run_ablation_experiment()