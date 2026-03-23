"""
测试工具模块
包含测试、评估和可视化逻辑
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from thop import profile, clever_format
from typing import Tuple, Dict, Any, List, Optional

from utils.config import Config, TestConfig, DatasetConfig, ModelConfig
from utils.visualize import plot_tsne


class ModelTester:
    """模型测试器类，负责模型测试和评估"""
    
    def __init__(self, model: torch.nn.Module, test_loader, 
                 test_config: TestConfig, dataset_config: DatasetConfig, 
                 model_config: ModelConfig, device: torch.device):
        """
        初始化测试器
        
        Args:
            model: 模型实例
            test_loader: 测试数据加载器
            test_config: 测试配置
            dataset_config: 数据集配置
            model_config: 模型配置
            device: 设备
        """
        self.model = model
        self.test_loader = test_loader
        self.test_config = test_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.device = device
        
        # 创建结果目录
        self.results_dir = Config.get_results_dir(model_config.mode, dataset_config.name)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, List, List]:
        """
        评估模型性能
        
        Returns:
            准确率、预测结果、真实标签、特征、时频图、推理时间
        """
        self.model.eval()
        all_preds, all_labels, all_feats = [], [], []
        all_time_freq_maps = []
        single_sample_times = []
        
        with torch.inference_mode():
            for xb, yb in tqdm(self.test_loader, desc="Testing"):
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                # 记录推理时间
                start_time = time.perf_counter()
                
                # 前向传播
                if hasattr(self.model, 'forward_with_features'):
                    logits, feats, time_freq_maps = self.model.forward_with_features(xb)
                    all_time_freq_maps.extend(time_freq_maps)
                else:
                    logits = self.model(xb)
                    feats = logits  # 使用输出作为特征
                    all_time_freq_maps.extend([None] * xb.size(0))
                
                end_time = time.perf_counter()
                batch_time = (end_time - start_time) * 1000  # 转换为毫秒
                single_sample_times.extend([batch_time / xb.size(0)] * xb.size(0))
                
                # 收集结果
                preds = F.softmax(logits, dim=1).argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())
                all_feats.append(feats.cpu())
        
        # 合并结果
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_feats = torch.cat(all_feats)
        
        # 计算准确率
        accuracy = (all_preds == all_labels).float().mean().item()
        
        return accuracy, all_preds, all_labels, all_feats, all_time_freq_maps, single_sample_times
    
    def test_single_sample_inference(self, input_shape: Tuple[int, int],
                                   num_tests: int = 100,
                                   batch_avg_time: Optional[float] = None,
                                   inference_batch_size: int = 1) -> Dict[str, Any]:
        """
        测试单样本推理时间
        
        Args:
            input_shape: 输入形状 (channels, length)
            num_tests: 测试次数
            batch_avg_time: 批次平均时间
            
        Returns:
            推理时间统计信息
        """
        self.model.eval()
        
        # 创建随机输入
        batch_size = max(1, int(inference_batch_size))
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # 预热
        with torch.inference_mode():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # 测试推理时间
        inference_times = []
        with torch.inference_mode():
            for _ in tqdm(range(num_tests), desc="Single Sample Inference Test"):
                start_time = time.perf_counter()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        inference_times = np.array(inference_times)
        
        # 计算统计信息
        stats = {
            'times': inference_times,
            'mean': np.mean(inference_times),
            'median': np.median(inference_times),
            'std': np.std(inference_times),
            'min': np.min(inference_times),
            'max': np.max(inference_times),
            'p95': np.percentile(inference_times, 95),
            'p99': np.percentile(inference_times, 99),
            'cv': np.std(inference_times) / np.mean(inference_times),
            'batch_size': batch_size,
            'mean_per_sample_ms': np.mean(inference_times) / batch_size,
            'median_per_sample_ms': np.median(inference_times) / batch_size,
            'p95_per_sample_ms': np.percentile(inference_times, 95) / batch_size,
            'p99_per_sample_ms': np.percentile(inference_times, 99) / batch_size,
            'peak_cpu_rss_mb': 0.0,
            'peak_cpu_rss_delta_mb': 0.0,
            'peak_gpu_allocated_mb': 0.0,
            'peak_gpu_reserved_mb': 0.0,
            'component_profile': None,
            'component_profile_path': None
        }
        
        # 打印结果
        print(f"\n⏱️ Inference Time Analysis ({num_tests} tests, batch={batch_size}):")
        print(f"   Mean: {stats['mean']:.4f} ms")
        print(f"   Mean per sample: {stats['mean_per_sample_ms']:.4f} ms")
        print(f"   Median: {stats['median']:.4f} ms")
        print(f"   Std: {stats['std']:.4f} ms")
        print(f"   Min: {stats['min']:.4f} ms")
        print(f"   Max: {stats['max']:.4f} ms")
        print(f"   95th percentile: {stats['p95']:.4f} ms")
        print(f"   99th percentile: {stats['p99']:.4f} ms")
        print(f"   Coefficient of Variation: {stats['cv']:.4f}")
        
        if batch_avg_time is not None:
            print(f"   Batch avg time per sample: {batch_avg_time:.4f} ms")
            print(f"   Single vs Batch ratio: {stats['mean'] / batch_avg_time:.2f}x")
        
        return stats
    
    def evaluate_model_complexity(self, input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        评估模型复杂度
        
        Args:
            input_shape: 输入形状 (channels, length)
            
        Returns:
            复杂度信息字典
        """
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # 计算FLOPs和参数量
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        # 计算模型大小（MB）
        param_size = 0
        buffer_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # 计算内存使用（MB）
        dummy_input_size = dummy_input.nelement() * dummy_input.element_size() / 1024 / 1024
        
        complexity_info = {
            'flops': flops,
            'flops_str': flops_str,
            'params': params,
            'params_str': params_str,
            'model_size_mb': model_size_mb,
            'input_size_mb': dummy_input_size,
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024
        }
        
        return complexity_info
    
    def print_model_complexity(self, complexity_info: Dict[str, Any], model_name: str):
        """打印模型复杂度信息"""
        print(f"\n📊 Model Complexity Analysis for {model_name}:")
        print(f"   Parameters: {complexity_info['params_str']}")
        print(f"   FLOPs: {complexity_info['flops_str']}")
        print(f"   Model Size: {complexity_info['model_size_mb']:.2f} MB")
        print(f"   Parameter Size: {complexity_info['param_size_mb']:.2f} MB")
        print(f"   Buffer Size: {complexity_info['buffer_size_mb']:.2f} MB")
        print(f"   Input Size: {complexity_info['input_size_mb']:.2f} MB")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str]) -> Tuple[np.ndarray, float]:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            
        Returns:
            混淆矩阵和准确率
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.trace(cm) / np.sum(cm)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制混淆矩阵热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {self.model_config.mode.upper()} on {self.dataset_config.name}\nOverall Accuracy: {accuracy:.3f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 调整布局并保存
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {save_path}")
        
        # 打印详细的分类报告
        print(f"\n📋 Classification Report for {self.dataset_config.name} ({self.model_config.mode.upper()}):")
        print("=" * 60)
        
        # 获取实际存在的类别
        unique_classes = np.unique(y_true)
        actual_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
        
        report = classification_report(y_true, y_pred, 
                                     labels=unique_classes,
                                     target_names=actual_class_names, 
                                     digits=4)
        print(report)
        
        # 计算每个类别的准确率
        print(f"\n📈 Per-class Accuracy:")
        print("-" * 40)
        for i, class_name in enumerate(class_names):
            if i < len(cm):
                class_accuracy = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                print(f"  {class_name}: {class_accuracy:.4f} ({cm[i, i]}/{np.sum(cm[i, :])})")
        
        return cm, accuracy
    
    def visualize_inference_times(self, inference_times: Dict[str, Any]):
        """
        可视化推理时间分布
        
        Args:
            inference_times: 推理时间统计信息
        """
        times = inference_times['times']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 推理时间直方图
        ax1.hist(times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(inference_times['mean'], color='red', linestyle='--', 
                    label=f'Mean: {inference_times["mean"]:.4f}ms')
        ax1.axvline(inference_times['median'], color='green', linestyle='--', 
                    label=f'Median: {inference_times["median"]:.4f}ms')
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 推理时间箱线图
        ax2.boxplot(times, vert=True)
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. 推理时间时序图
        ax3.plot(range(len(times)), times, 'b-', alpha=0.7, linewidth=1)
        ax3.axhline(inference_times['mean'], color='red', linestyle='--', 
                    label=f'Mean: {inference_times["mean"]:.4f}ms')
        ax3.set_xlabel('Test Sample Index')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.set_title('Inference Time Over Test Samples')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计信息表格
        ax4.axis('tight')
        ax4.axis('off')
        
        stats_data = [
            ['Metric', 'Value (ms)'],
            ['Mean', f"{inference_times['mean']:.4f}"],
            ['Median', f"{inference_times['median']:.4f}"],
            ['Std Dev', f"{inference_times['std']:.4f}"],
            ['Min', f"{inference_times['min']:.4f}"],
            ['Max', f"{inference_times['max']:.4f}"],
            ['95th %ile', f"{inference_times['p95']:.4f}"],
            ['99th %ile', f"{inference_times['p99']:.4f}"],
            ['CV', f"{inference_times['cv']:.4f}"]
        ]
        
        table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(stats_data)):
            if i == 0:  # 标题行
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#ffffff')
        
        ax4.set_title('Inference Time Statistics')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "inference_time_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Inference time analysis saved to: {save_path}")
    
    def visualize_time_frequency_maps(self, time_freq_maps: List, labels: torch.Tensor):
        """
        可视化时频图
        
        Args:
            time_freq_maps: 时频图列表
            labels: 标签
        """
        if not time_freq_maps or time_freq_maps[0] is None:
            print("⚠️ No time-frequency maps available for visualization")
            return
        
        # 选择每个类别的一个样本进行可视化
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        
        fig, axes = plt.subplots(2, min(4, num_classes), figsize=(16, 8))
        if num_classes == 1:
            axes = axes.reshape(2, 1)
        elif num_classes <= 4:
            axes = axes.reshape(2, num_classes)
        
        for i, label in enumerate(unique_labels[:8]):  # 最多显示8个类别
            # 找到该类别的第一个样本
            indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                sample_idx = indices[0].item()
                time_freq_map = time_freq_maps[sample_idx]
                
                if time_freq_map is not None:
                    row = i // 4
                    col = i % 4
                    
                    if num_classes <= 4:
                        ax = axes[row, col] if num_classes > 1 else axes[row]
                    else:
                        ax = axes[row, col]
                    
                    # 显示时频图
                    im = ax.imshow(time_freq_map, aspect='auto', cmap='viridis')
                    ax.set_title(f'Class {label.item()}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Frequency')
                    plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "time_frequency_maps.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Time-frequency maps saved to: {save_path}")
    
    def create_complexity_comparison(self, complexity_info: Dict[str, Any]):
        """
        创建复杂度对比图
        
        Args:
            complexity_info: 复杂度信息
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 参数量饼图
        param_mb = complexity_info['param_size_mb']
        buffer_mb = complexity_info['buffer_size_mb']
        
        sizes = [param_mb, buffer_mb]
        labels = ['Parameters', 'Buffers']
        colors = ['#ff9999', '#66b3ff']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Model Memory Distribution\nTotal: {complexity_info["model_size_mb"]:.2f} MB')
        
        # 2. FLOPs条形图
        flops_gflops = complexity_info['flops'] / 1e9
        ax2.bar(['FLOPs'], [flops_gflops], color='lightgreen')
        ax2.set_ylabel('GFLOPs')
        ax2.set_title(f'Computational Complexity\n{complexity_info["flops_str"]} FLOPs')
        
        # 3. 参数量条形图
        params_millions = complexity_info['params'] / 1e6
        ax3.bar(['Parameters'], [params_millions], color='lightcoral')
        ax3.set_ylabel('Millions')
        ax3.set_title(f'Model Parameters\n{complexity_info["params_str"]} Parameters')
        
        # 4. 模型信息表格
        ax4.axis('tight')
        ax4.axis('off')
        
        info_data = [
            ['Metric', 'Value'],
            ['Parameters', complexity_info['params_str']],
            ['FLOPs', complexity_info['flops_str']],
            ['Model Size', f"{complexity_info['model_size_mb']:.2f} MB"],
            ['Param Size', f"{complexity_info['param_size_mb']:.2f} MB"],
            ['Buffer Size', f"{complexity_info['buffer_size_mb']:.2f} MB"]
        ]
        
        table = ax4.table(cellText=info_data, cellLoc='center', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(info_data)):
            if i == 0:  # 标题行
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#ffffff')
        
        ax4.set_title('Model Complexity Summary')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "complexity_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Complexity comparison saved to: {save_path}")
    
    def run_complete_test(self, input_shape: Tuple[int, int], 
                         class_names: List[str]) -> Dict[str, Any]:
        """
        运行完整测试流程
        
        Args:
            input_shape: 输入形状
            class_names: 类别名称
            
        Returns:
            测试结果字典
        """
        print(f"\n🔬 开始模型测试...")
        
        # 1. 评估模型复杂度
        print(f"\n🔬 Evaluating model complexity...")
        complexity_info = self.evaluate_model_complexity(input_shape)
        self.print_model_complexity(complexity_info, f"{self.model_config.mode}_{self.dataset_config.name}")
        
        # 2. 模型评估
        print(f"\n🧪 Running model evaluation...")
        acc, preds, labels, feats, time_freq_maps, single_sample_times = self.evaluate()
        print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
        print("🔍 Label distribution (Ground Truth):", dict(Counter(labels.tolist())))
        print("🔍 Prediction distribution:", dict(Counter(preds.tolist())))
        
        # 3. 单样本推理时间测试
        print(f"\n⏱️ Testing single sample inference time...")
        batch_avg_time = np.mean(single_sample_times)
        detailed_inference_times = self.test_single_sample_inference(
            input_shape, 
            num_tests=self.test_config.num_inference_tests, 
            batch_avg_time=batch_avg_time
        )
        
        # 4. 绘制混淆矩阵
        print(f"\n📊 Generating confusion matrix...")
        cm, cm_accuracy = self.plot_confusion_matrix(
            y_true=labels.numpy(),
            y_pred=preds.numpy(),
            class_names=class_names
        )
        
        # 5. 可视化推理时间
        print(f"\n📊 Visualizing inference times...")
        self.visualize_inference_times(detailed_inference_times)
        
        # 6. 可视化时频图
        print(f"\n📊 Visualizing time-frequency maps...")
        self.visualize_time_frequency_maps(time_freq_maps, labels)
        
        # 7. t-SNE可视化
        print(f"\n📊 Generating t-SNE visualization...")
        plot_tsne(
            feats,
            labels,
            class_names=class_names,
            title=f"t-SNE of {self.model_config.mode.capitalize()} Mode ({self.dataset_config.name}) Final Classifier Outputs",
            perplexity=self.test_config.tsne_perplexity,
            save_path=os.path.join(self.results_dir, "tsne_visualization.png")
        )
        
        # 8. 复杂度对比图
        print(f"\n📊 Creating complexity comparison...")
        self.create_complexity_comparison(complexity_info)
        
        # 9. 保存结果
        results_dict = {
            "predictions": preds,
            "labels": labels,
            "accuracy": acc,
            "time_freq_maps": time_freq_maps,
            "model_type": self.model_config.mode,
            "mode": self.model_config.mode,
            "dataset_type": self.dataset_config.name,
            "single_sample_times": single_sample_times,
            "detailed_inference_times": detailed_inference_times,
            "complexity_info": complexity_info,
            "confusion_matrix": cm,
            "confusion_matrix_accuracy": cm_accuracy,
            "class_names": class_names
        }
        
        # 添加模式特定信息
        if self.model_config.mode == "wavelet_traditional":
            results_dict["wavelet_type"] = self.model_config.wavelet_type
            results_dict["wavelet_levels"] = self.model_config.wavelet_levels
        elif self.model_config.mode in ["wavelet_learnable", "wavelet_lite"]:
            results_dict["decompose_levels"] = self.model_config.decompose_levels
            results_dict["num_parallel_groups"] = self.model_config.num_parallel_groups
            results_dict["feature_extraction"] = Config.get_feature_extraction_description(self.model_config.mode)
        
        # 保存结果
        torch.save(results_dict, os.path.join(self.results_dir, "test_results.pt"))
        
        print(f"🎉 Testing completed! Results saved to {self.results_dir} directory")
        
        return results_dict
