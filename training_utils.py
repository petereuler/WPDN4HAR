"""
训练工具模块
包含训练循环、验证逻辑和保存逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
from typing import Dict, Any

from utils.config import Config, TrainingConfig, DatasetConfig, ModelConfig


class Trainer:
    """训练器类，负责模型训练和验证"""
    
    def __init__(self, model: torch.nn.Module, train_loader, val_loader, 
                 training_config: TrainingConfig, dataset_config: DatasetConfig, 
                 model_config: ModelConfig, device: torch.device):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            training_config: 训练配置
            dataset_config: 数据集配置
            model_config: 模型配置
            device: 设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.device = device
        
        # 初始化优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化TensorBoard
        self.writer = self._create_tensorboard_writer()
        
        # 训练状态
        self.best_acc = 0.0
        self.best_acc_last_epochs = 0.0
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.training_config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer_type}")
    
    def _create_tensorboard_writer(self) -> SummaryWriter:
        """创建TensorBoard写入器"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_name = f"{Config.get_log_name(self.model_config.mode)}_{self.dataset_config.name}_{timestamp}"
        
        writer = SummaryWriter(log_dir=f"runs/{log_name}")
        
        # 记录模型信息
        total_params = self._count_parameters()
        writer.add_text("Model/Parameter_Count", f"{total_params:,}")
        writer.add_text("Model/Architecture", Config.get_architecture_description(self.model_config.mode))
        writer.add_text("Training/Mode", self.model_config.mode)
        writer.add_text("Dataset/Type", self.dataset_config.name)
        writer.add_text("Dataset/Channels", str(self.dataset_config.in_channels))
        writer.add_text("Dataset/Classes", str(self.dataset_config.num_classes))
        
        # 记录模式特定信息
        if self.model_config.mode == "wavelet_traditional":
            writer.add_text("Model/Wavelet_Type", self.model_config.wavelet_type)
            writer.add_text("Model/Wavelet_Levels", str(self.model_config.wavelet_levels))
            writer.add_text("Training/Orthogonality_Method", "None (Traditional)")
        elif self.model_config.mode in ["wavelet_learnable", "wavelet_lite"]:
            writer.add_text("Training/Orthogonality_Method", "Loss-based")
            writer.add_text("Training/Filter_Normalization", "None")
            if hasattr(self.training_config, 'orth_weight'):
                writer.add_text("Training/Orth_Weight", str(self.training_config.orth_weight))
            if self.model_config.mode == "wavelet_lite":
                writer.add_text("Model/Classifier_Type", "Low-rank time-frequency Conv1d")
                writer.add_text("Model/Convolution_Type", "Factorized Conv1d")
        
        return writer
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        total_orth_loss = 0.0

        for xb, yb in tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}"):
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(xb)
            clf_loss = self.criterion(logits, yb)
            
            # 正交损失（仅对学习型小波有效）
            if self.model_config.mode in ['wavelet_lite'] and hasattr(self.model, 'get_orthogonality_loss'):
                orth_loss = self.model.get_orthogonality_loss()
                orth_weight = self.training_config.orth_weight
                total_loss_batch = clf_loss + orth_weight * orth_loss
            else:
                orth_loss = torch.tensor(0.0, device=self.device)
                total_loss_batch = clf_loss
            
            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += clf_loss.item() * xb.size(0)
            total_orth_loss += orth_loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += xb.size(0)
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_orth_loss = total_orth_loss / total_samples
        train_acc = total_correct / total_samples * 100
        
        # 记录到TensorBoard
        self.writer.add_scalar("Train/Classification_Loss", avg_loss, epoch)
        self.writer.add_scalar("Train/Orthogonality_Loss", avg_orth_loss, epoch)
        self.writer.add_scalar("Train/Accuracy", train_acc, epoch)

        return {
            "loss": avg_loss,
            "orth_loss": avg_orth_loss,
            "accuracy": train_acc,
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += xb.size(0)
        
        # 计算平均指标
        if val_total > 0:
            val_avg_loss = val_loss / val_total
            val_acc = val_correct / val_total * 100
        else:
            # 如果验证集为空，使用训练集的损失和精度作为替代
            print("⚠️ 验证集为空，跳过验证步骤")
            val_avg_loss = 0.0
            val_acc = 0.0
        
        # 记录到TensorBoard
        self.writer.add_scalar("Val/Loss", val_avg_loss, epoch)
        self.writer.add_scalar("Val/Accuracy", val_acc, epoch)
        
        # 记录学习率
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar("LR", param_group["lr"], epoch)
        
        return {
            "loss": val_avg_loss,
            "accuracy": val_acc
        }
    
    def save_best_model(self, val_acc: float, epoch: int) -> bool:
        """
        保存最佳模型
        
        Args:
            val_acc: 验证准确率
            epoch: 当前epoch数
            
        Returns:
            是否保存了模型
        """
        saved = False
        
        # 保存全局最佳模型
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print(f"📊 全局最佳验证精度: {val_acc:.2f}% (Epoch {epoch})")
        
        # 保存最后几轮的最佳模型
        last_epochs = 5
        if epoch > self.training_config.epochs - last_epochs:
            if val_acc > self.best_acc_last_epochs:
                self.best_acc_last_epochs = val_acc
                
                # 获取模型文件名
                model_filename = Config.get_model_checkpoint_path(
                    self.model_config.mode, 
                    self.dataset_config.name
                )
                
                # 确保目录存在
                os.makedirs("checkpoints", exist_ok=True)
                
                # 保存模型
                torch.save(self.model.state_dict(), model_filename)
                
                print(f"✅ 最后{last_epochs}轮最佳模型已保存! Val Acc: {val_acc:.2f}% (Epoch {epoch})")
                print(f"   模型文件: {model_filename}")
                saved = True
        
        return saved
    
    def train(self) -> Dict[str, Any]:
        """
        完整训练流程
        
        Returns:
            训练结果字典
        """
        print(f"\n🚀 开始训练...")
        print(f"📊 模型参数数量: {self._count_parameters():,}")
        
        training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        for epoch in range(1, self.training_config.epochs + 1):
            # 训练阶段
            train_metrics = self.train_epoch(epoch)
            
            # 验证阶段
            val_metrics = self.validate_epoch(epoch)
            
            # 记录历史
            training_history["train_loss"].append(train_metrics["loss"])
            training_history["train_acc"].append(train_metrics["accuracy"])
            training_history["val_loss"].append(val_metrics["loss"])
            training_history["val_acc"].append(val_metrics["accuracy"])
            
            # 打印进度
            print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                  f"Orth Loss = {train_metrics['orth_loss']:.4f}, "
                  f"Accuracy = {train_metrics['accuracy']:.2f}%")
            print(f"Epoch {epoch}: Val   Loss = {val_metrics['loss']:.4f}, "
                  f"Accuracy = {val_metrics['accuracy']:.2f}%")
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"[Info] Epoch {epoch}: Current learning rate = {current_lr:.6f} (固定)")
            
            # 保存最佳模型
            self.save_best_model(val_metrics["accuracy"], epoch)
        
        # 关闭TensorBoard
        self.writer.close()
        
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🎉 训练完成！最佳模型已保存。")
        
        return {
            "best_acc": self.best_acc,
            "best_acc_last_epochs": self.best_acc_last_epochs,
            "history": training_history
        }
