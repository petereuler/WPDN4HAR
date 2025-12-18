import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class UCIHARWindowedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_uci_har_split(data_dir, split='train', window_size=128, step=64):
    """
    载入UCI HAR训练或测试数据，做滑窗切分，并对每个通道归一化。
    split: 'train' 或 'test'
    """
    signals = []
    for signal_type in ['total_acc_x', 'total_acc_y', 'total_acc_z',
                        'body_gyro_x', 'body_gyro_y', 'body_gyro_z']:
        path = os.path.join(data_dir, split, 'Inertial Signals', f'{signal_type}_{split}.txt')
        signals.append(np.loadtxt(path))  # shape (num_samples, 128)
    X_raw = np.stack(signals, axis=-1)  # (num_samples, 128, 6)

    # [重点] 对每个通道归一化（全体样本+时间维）
    for c in range(X_raw.shape[-1]):
        mean = X_raw[:, :, c].mean()
        std = X_raw[:, :, c].std()
        X_raw[:, :, c] = (X_raw[:, :, c] - mean) / (std + 1e-8)

    y_raw = pd.read_csv(os.path.join(data_dir, split, f'y_{split}.txt'), header=None)[0].values - 1

    all_X, all_y = [], []
    for i in range(len(X_raw)):
        series = X_raw[i]  # (128, 6)
        label = y_raw[i]
        for start in range(0, 128 - window_size + 1, step):
            all_X.append(series[start:start + window_size])
            all_y.append(label)

    X = np.stack(all_X)
    y = np.array(all_y)
    return X, y

def create_train_val_loaders(data_dir, batch_size=64, val_split=0.1, window_size=128, step=64, seed=42):
    """
    从原训练集划分出一部分作为验证集，保持类别比例。
    使用分层采样确保验证集和训练集具有相同的类别分布。
    """
    X_train, y_train = load_uci_har_split(data_dir, split='train', window_size=window_size, step=step)
    
    # 计算每个类别的样本数量
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print(f"原始训练集类别分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  类别 {label}: {count} 样本")
    
    # 为每个类别计算验证集大小
    val_indices = []
    train_indices = []
    
    np.random.seed(seed)
    
    for label in unique_labels:
        # 找到该类别的所有样本索引
        label_indices = np.where(y_train == label)[0]
        
        # 计算该类别的验证集大小
        label_val_size = int(len(label_indices) * val_split)
        
        # 随机打乱该类别的索引
        np.random.shuffle(label_indices)
        
        # 分配验证集和训练集索引
        val_indices.extend(label_indices[:label_val_size])
        train_indices.extend(label_indices[label_val_size:])
    
    # 创建数据集
    full_dataset = UCIHARWindowedDataset(X_train, y_train)
    
    # 使用自定义的SubsetRandomSampler来创建训练集和验证集
    from torch.utils.data import SubsetRandomSampler, SequentialSampler
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, 
                             num_workers=16, pin_memory=True, persistent_workers=True, 
                             prefetch_factor=4)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler,
                           num_workers=16, pin_memory=True, persistent_workers=True,
                           prefetch_factor=4)
    
    # 验证类别分布
    print(f"\n划分后的数据集大小:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")
    
    # 检查验证集的类别分布
    val_labels = y_train[val_indices]
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    print(f"验证集类别分布:")
    for label, count in zip(val_unique, val_counts):
        print(f"  类别 {label}: {count} 样本")
    
    return train_loader, val_loader

def create_test_loader(data_dir, batch_size=64, window_size=128, step=64):
    X_test, y_test = load_uci_har_split(data_dir, split='test', window_size=window_size, step=step)
    test_ds = UCIHARWindowedDataset(X_test, y_test)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False), len(np.unique(y_test))

if __name__ == "__main__":
    data_dir = "/Users/shezhuoyuan/Documents/M-Research/WNN/dataset/UCIHAR"

    train_loader, val_loader = create_train_val_loaders(data_dir, batch_size=32, val_split=0.1)
    test_loader, num_classes = create_test_loader(data_dir, batch_size=32)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    # 获取训练集前2个样本
    train_iter = iter(train_loader)
    X_train_batch, y_train_batch = next(train_iter)
    for i in range(2):
        print(f"\n[Train Sample {i}]")
        print(f"  X shape: {X_train_batch[i].shape}")  # (C, T)
        print(f"  y label: {y_train_batch[i].item()}")
        print(f"  X data (first 5 values of each channel):")
        for c in range(X_train_batch.shape[1]):
            print(f"    Channel {c}: {X_train_batch[i][c][:5].tolist()}")

    # 获取验证集前2个样本
    val_iter = iter(val_loader)
    X_val_batch, y_val_batch = next(val_iter)
    for i in range(2):
        print(f"\n[Val Sample {i}]")
        print(f"  X shape: {X_val_batch[i].shape}")  # (C, T)
        print(f"  y label: {y_val_batch[i].item()}")
        print(f"  X data (first 5 values of each channel):")
        for c in range(X_val_batch.shape[1]):
            print(f"    Channel {c}: {X_val_batch[i][c][:5].tolist()}")

    # 获取测试集前2个样本
    test_iter = iter(test_loader)
    X_test_batch, y_test_batch = next(test_iter)
    for i in range(2):
        print(f"\n[Test Sample {i}]")
        print(f"  X shape: {X_test_batch[i].shape}")  # (C, T)
        print(f"  y label: {y_test_batch[i].item()}")
        print(f"  X data (first 5 values of each channel):")
        for c in range(X_test_batch.shape[1]):
            print(f"    Channel {c}: {X_test_batch[i][c][:5].tolist()}")