"""
多数据集重复实验脚本
按 experiment_config.py 中定义的 sweep 配置运行多数据集实验，并绘制箱式图。
"""

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import seaborn as sns
import pandas as pd

# 设置matplotlib支持中文显示，但用户偏好英文标签
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体
plt.rcParams['axes.unicode_minus'] = False

from experiment_config import MULTI_DATASET_MODE, MULTI_DATASET_REPEAT, MULTI_DATASET_SWEEP, ExperimentSelection
from experiment_runtime import (
    build_experiment_bundle,
    create_test_setup,
    create_training_setup,
    get_checkpoint_path,
    prepare_model_for_inference,
)
from training_utils import Trainer


def run_single_experiment(dataset_name: str, experiment_id: int) -> Dict[str, Any]:
    """
    运行单个实验

    Args:
        dataset_name: 数据集名称
        experiment_id: 实验ID

    Returns:
        实验结果字典
    """
    selection = ExperimentSelection(mode=MULTI_DATASET_MODE, dataset=dataset_name)
    print(f"\n{'='*70}")
    print(f"运行实验 {experiment_id+1}/{MULTI_DATASET_REPEAT} 在数据集 {dataset_name} 上")
    print(f"{'='*70}")

    train_bundle = build_experiment_bundle(selection, stage="train")
    test_bundle = build_experiment_bundle(selection, stage="test")
    print(f"数据集: {dataset_name}, 训练设备: {train_bundle.device}, 测试设备: {test_bundle.device}")

    # ==================== 数据加载 ====================
    model, train_loader, val_loader = create_training_setup(train_bundle)
    _, test_loader, _ = create_test_setup(test_bundle)

    # ==================== 训练阶段 ====================
    print(f"\n🏗️ 创建并训练模型...")

    # 创建训练器并训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=train_bundle.training_config,
        dataset_config=train_bundle.dataset_config,
        model_config=train_bundle.model_config,
        device=train_bundle.device
    )

    # 执行训练
    training_results = trainer.train()
    best_val_acc = training_results['best_acc']

    # ==================== 测试阶段 ====================
    print(f"\n🧪 测试训练后的模型...")
    model = model.to(test_bundle.device)

    # 加载最佳模型权重
    model_checkpoint_path = get_checkpoint_path(test_bundle)
    if os.path.exists(model_checkpoint_path):
        from utils.model_factory import ModelFactory

        ModelFactory.load_model_weights(model, model_checkpoint_path, test_bundle.device)
        print(f"✅ 加载最佳模型权重: {model_checkpoint_path}")
    else:
        print(f"⚠️ 模型权重文件不存在: {model_checkpoint_path}")

    model = prepare_model_for_inference(model, test_bundle)

    # 简化的测试，只计算准确性，跳过t-SNE等可视化
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(test_bundle.device), yb.to(test_bundle.device)
            outputs = model(xb)
            _, predicted = torch.max(outputs.data, 1)
            test_total += yb.size(0)
            test_correct += (predicted == yb).sum().item()

    test_acc = test_correct / test_total

    print(f"✅ 实验完成! 验证精度: {best_val_acc:.2f}%, 测试精度: {test_acc*100:.2f}%")

    return {
        'experiment_id': experiment_id,
        'dataset': dataset_name,
        'mode': selection.mode,
        'val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'training_history': training_results['history']
    }


def run_all_experiments() -> Dict[str, List[Dict[str, Any]]]:
    """
    在所有数据集上运行10次实验

    Returns:
        所有实验结果
    """
    datasets = list(MULTI_DATASET_SWEEP)
    all_results = {dataset: [] for dataset in datasets}

    print(f"\n{'='*80}")
    print(f"开始多数据集重复实验 - 共{len(datasets)}个数据集，每个数据集{MULTI_DATASET_REPEAT}次实验")
    print(f"{'='*80}")

    total_experiments = len(datasets) * MULTI_DATASET_REPEAT
    current_experiment = 0

    for dataset in datasets:
        print(f"\n🎯 开始数据集 {dataset} 的实验")

        for exp_id in range(MULTI_DATASET_REPEAT):
            current_experiment += 1
            print(f"\n📊 总进度: {current_experiment}/{total_experiments}")

            try:
                result = run_single_experiment(dataset, exp_id)
                all_results[dataset].append(result)
                print(f"✅ {dataset} 实验 {exp_id+1} 完成")

            except Exception as e:
                print(f"❌ {dataset} 实验 {exp_id+1} 失败: {str(e)}")
                # 记录失败的实验
                all_results[dataset].append({
                    'experiment_id': exp_id,
                    'dataset': dataset,
                    'val_accuracy': None,
                    'test_accuracy': None,
                    'error': str(e)
                })

    return all_results


def save_results(results: Dict[str, List[Dict[str, Any]]], filename: str = "experiment_results.json"):
    """保存实验结果到文件"""
    # 转换为可序列化的格式
    serializable_results = {}
    for dataset, experiments in results.items():
        serializable_results[dataset] = []
        for exp in experiments:
            exp_copy = exp.copy()
            # 移除不可序列化的训练历史
            if 'training_history' in exp_copy:
                del exp_copy['training_history']
            serializable_results[dataset].append(exp_copy)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"💾 实验结果已保存到 {filename}")


def create_boxplot(results: Dict[str, List[Dict[str, Any]]], save_path: str = "dataset_comparison_boxplot.png"):
    """
    创建箱式图显示各数据集的测试精度分布

    Args:
        results: 实验结果
        save_path: 保存路径
    """
    # 准备数据
    plot_data = []

    print("📊 箱式图数据统计:")
    print("-" * 50)

    for dataset, experiments in results.items():
        valid_experiments = [exp for exp in experiments if exp.get('test_accuracy') is not None]
        failed_experiments = [exp for exp in experiments if exp.get('test_accuracy') is None]

        print(f"{dataset}: {len(valid_experiments)}成功, {len(failed_experiments)}失败")

        if valid_experiments:
            acc_values = [exp['test_accuracy'] * 100 for exp in valid_experiments]
            for acc in acc_values:
                plot_data.append({'Dataset': dataset, 'Accuracy': acc})

    # 创建箱式图
    if plot_data:
        plt.figure(figsize=(12, 8))

        # 创建箱式图
        ax = sns.boxplot(x='Dataset', y='Accuracy', data=pd.DataFrame(plot_data),
                        palette="Set3", width=0.6)

        # 添加散点显示每个实验的结果
        for i, dataset in enumerate(results.keys()):
            valid_experiments = [exp for exp in results[dataset] if exp.get('test_accuracy') is not None]
            if valid_experiments:
                x_positions = np.full(len(valid_experiments), i)
                y_values = [exp['test_accuracy'] * 100 for exp in valid_experiments]
                plt.scatter(x_positions, y_values, alpha=0.7, color='red', s=40, zorder=3, label='Individual Runs' if i == 0 else "")

        # 设置标签和标题
        plt.xlabel('Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        plt.title(
            f'WPDN Model Performance Across Datasets\n({MULTI_DATASET_REPEAT} runs per dataset)',
            fontsize=16,
            pad=20,
            fontweight='bold',
        )

        # 添加网格
        plt.grid(True, alpha=0.3, axis='y')

        # 添加统计信息注释
        stats_text = []
        for dataset in results.keys():
            valid_experiments = [exp for exp in results[dataset] if exp.get('test_accuracy') is not None]
            if valid_experiments:
                acc_values = [exp['test_accuracy'] * 100 for exp in valid_experiments]
                mean_acc = np.mean(acc_values)
                std_acc = np.std(acc_values)
                stats_text.append(f"{dataset}: {mean_acc:.2f} ± {std_acc:.2f}")

        # 在图的右上角添加统计信息
        stats_box = '\n'.join(stats_text)
        plt.text(0.98, 0.98, stats_box, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # 添加图例
        plt.legend(loc='lower right', fontsize=12)

        # 调整布局
        plt.tight_layout()

        # 保存高清图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 箱式图已保存到 {save_path}")
        plt.show()
    else:
        print("⚠️ 没有有效的实验结果来绘制箱式图")


def print_summary(results: Dict[str, List[Dict[str, Any]]]):
    """打印实验结果摘要"""
    print(f"\n{'='*80}")
    print("实验结果摘要")
    print(f"{'='*80}")

    for dataset, experiments in results.items():
        valid_experiments = [exp for exp in experiments if exp.get('test_accuracy') is not None]
        failed_experiments = [exp for exp in experiments if exp.get('test_accuracy') is None]

        if valid_experiments:
            test_accs = [exp['test_accuracy'] * 100 for exp in valid_experiments]
            val_accs = [exp['val_accuracy'] for exp in valid_experiments if exp.get('val_accuracy') is not None]

            print(f"\n📊 {dataset}:")
            print(f"   成功实验数: {len(valid_experiments)}/{MULTI_DATASET_REPEAT}")
            print(f"   测试精度 - 均值: {np.mean(test_accs):.2f}%, 标准差: {np.std(test_accs):.2f}%")
            print(f"   测试精度 - 范围: {np.min(test_accs):.2f}% - {np.max(test_accs):.2f}%")

            if val_accs:
                print(f"   验证精度 - 均值: {np.mean(val_accs):.2f}%, 标准差: {np.std(val_accs):.2f}%")

        if failed_experiments:
            print(f"   失败实验数: {len(failed_experiments)}/{MULTI_DATASET_REPEAT}")

    print(f"{'='*80}")


def main():
    """主函数"""
    print("🚀 WPDN多数据集重复实验脚本")
    print(f"📋 计划: 模型 {MULTI_DATASET_MODE} 在 {', '.join(MULTI_DATASET_SWEEP)} 上各运行 {MULTI_DATASET_REPEAT} 次实验")
    print("⚙️  配置: 统一从 experiment_config.py 读取")

    # 运行所有实验
    results = run_all_experiments()

    # 保存结果
    save_results(results)

    # 打印摘要
    print_summary(results)

    # 创建箱式图
    create_boxplot(results)

    print("\n🎉 所有实验完成！")


if __name__ == "__main__":
    main()
