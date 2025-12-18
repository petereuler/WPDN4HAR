#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验工具模块
提供消融实验所需的训练和评估函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, List
from config import TrainingConfig


def train_model(model: torch.nn.Module, 
                train_loader, 
                val_loader, 
                config: TrainingConfig, 
                device: torch.device,
                save_path: str = None,
                use_orthogonality_loss: bool = True) -> Dict[str, List]:
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
        device: 设备
        save_path: 模型保存路径（可选）
        use_orthogonality_loss: 是否使用正交损失
        
    Returns:
        训练历史记录
    """
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'orth_loss': [] if use_orthogonality_loss else None
    }
    
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_orth_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 正交损失
            if use_orthogonality_loss and hasattr(model, 'get_orthogonality_loss'):
                orth_loss = model.get_orthogonality_loss()
                total_loss = loss + config.orth_weight * orth_loss
                train_orth_loss += orth_loss.item()
            else:
                total_loss = loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            # 更新进度条
            train_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
                
                val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(len(val_loader)):.4f}',
                    'Acc': f'{val_acc:.2f}%'
                })
        
        # 计算平均值
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100. * val_correct / val_total
        avg_orth_loss = train_orth_loss / len(train_loader) if use_orthogonality_loss else 0.0
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        if use_orthogonality_loss and history['orth_loss'] is not None:
            history['orth_loss'].append(avg_orth_loss)
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%')
        if use_orthogonality_loss and avg_orth_loss > 0:
            print(f'Orthogonality Loss: {avg_orth_loss:.6f}')
    
    return history


def evaluate_model(model: torch.nn.Module, 
                  test_loader, 
                  device: torch.device) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        device: 设备
        
    Returns:
        评估指标
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # 推理时间测量
    inference_times = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            
            # 测量推理时间
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            # 记录每个batch的推理时间（转换为每个样本的时间）
            batch_time = (end_time - start_time) / data.size(0)
            inference_times.extend([batch_time] * data.size(0))
            
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )
    
    avg_test_loss = test_loss / len(test_loader)
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_loss': avg_test_loss,
        'avg_inference_time_ms': avg_inference_time
    }