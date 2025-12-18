import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os

def plot_tsne(features, labels, class_names=None, title="t-SNE Visualization", 
              perplexity=30, n_components=2, save_path=None, figsize=(10, 8)):
    """
    使用t-SNE可视化特征向量
    
    Args:
        features: 特征向量 [N, D]
        labels: 标签 [N]
        class_names: 类别名称列表
        title: 图表标题
        perplexity: t-SNE的perplexity参数
        n_components: 降维后的维度
        save_path: 保存路径
        figsize: 图表大小
    """
    # 转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 执行t-SNE降维
    print(f"执行t-SNE降维，特征维度: {features.shape} -> {n_components}D")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_jobs=-1)
    features_2d = tsne.fit_transform(features)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 获取唯一标签和颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # 绘制散点图
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if class_names is not None else f"Class {label}"
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=class_name, alpha=0.7, s=30)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE可视化已保存到: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", 
                         save_path=None, figsize=(10, 8), normalize=True):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        normalize: 是否归一化
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # 处理除零情况
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 使用seaborn绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()

def plot_training_curves(train_losses, train_accs, val_losses=None, val_accs=None, 
                        save_path=None, figsize=(12, 5)):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        save_path: 保存路径
        figsize: 图表大小
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    if val_accs is not None:
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()

def plot_feature_distribution(features, labels, feature_names=None, n_features=6, 
                             save_path=None, figsize=(15, 10)):
    """
    绘制特征分布图
    
    Args:
        features: 特征向量 [N, D]
        labels: 标签 [N]
        feature_names: 特征名称列表
        n_features: 要显示的特征数量
        save_path: 保存路径
        figsize: 图表大小
    """
    # 转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 选择要显示的特征
    n_features = min(n_features, features.shape[1])
    features_subset = features[:, :n_features]
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # 创建子图
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i in range(n_features):
        ax = axes[i]
        
        # 为每个类别绘制分布
        for j, label in enumerate(unique_labels):
            mask = labels == label
            ax.hist(features_subset[mask, i], bins=30, alpha=0.7, 
                   color=colors[j], label=f'Class {label}', density=True)
        
        feature_name = feature_names[i] if feature_names is not None else f'Feature {i+1}'
        ax.set_title(feature_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存到: {save_path}")
    
    plt.show()

def plot_time_series(data, labels=None, title="Time Series Data", 
                    save_path=None, figsize=(12, 8), n_samples=5):
    """
    绘制时间序列数据
    
    Args:
        data: 时间序列数据 [N, C, T] 或 [N, T]
        labels: 标签 [N]
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        n_samples: 要显示的样本数量
    """
    # 转换为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 选择要显示的样本
    n_samples = min(n_samples, data.shape[0])
    data_subset = data[:n_samples]
    
    # 确定数据维度
    if len(data_subset.shape) == 2:
        # [N, T] -> [N, 1, T]
        data_subset = data_subset[:, np.newaxis, :]
    
    n_channels = data_subset.shape[1]
    time_steps = data_subset.shape[2]
    
    # 创建子图
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize)
    if n_channels == 1:
        axes = [axes]
    
    time_axis = np.arange(time_steps)
    
    for ch in range(n_channels):
        ax = axes[ch]
        
        for i in range(n_samples):
            label_str = f"Sample {i}" if labels is None else f"Class {labels[i]}"
            ax.plot(time_axis, data_subset[i, ch, :], 
                   label=label_str, alpha=0.8, linewidth=1.5)
        
        ax.set_title(f'Channel {ch+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时间序列图已保存到: {save_path}")
    
    plt.show()

def plot_accuracy_comparison(accuracies, model_names, title="Model Accuracy Comparison", 
                           save_path=None, figsize=(10, 6)):
    """
    绘制模型准确率对比图
    
    Args:
        accuracies: 准确率列表
        model_names: 模型名称列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 创建柱状图
    bars = plt.bar(range(len(accuracies)), accuracies, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(accuracies))))
    
    # 在柱子上添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率对比图已保存到: {save_path}")
    
    plt.show()

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, 
                        save_path=None, figsize=(15, 5)):
    """
    绘制完整的学习曲线（损失和准确率）
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
        figsize: 图表大小
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Learning Curves - Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    ax2.set_title('Learning Curves - Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存到: {save_path}")
    
    plt.show()
