import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class WISDMWindowedDataset(Dataset):
    """
    WISDM数据集的PyTorch Dataset实现
    直接从原始数据文件加载，进行滑窗处理和归一化
    """
    def __init__(self, X, y):
        """
        Args:
            X: numpy array, shape (N, T, C) - N个样本，T个时间步，C个通道
            y: numpy array, shape (N,) - 标签
        """
        # 转换为 (N, C, T) 格式，符合PyTorch卷积层的输入要求
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_wisdm_raw_data(data_dir):
    """
    加载WISDM原始数据文件
    
    Args:
        data_dir: WISDM数据集目录路径
        
    Returns:
        pandas.DataFrame: 清理后的原始数据
    """
    raw_data_path = os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt")
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"WISDM原始数据文件不存在: {raw_data_path}")
    
    print(f"正在加载WISDM原始数据: {raw_data_path}")
    
    # 读取原始数据
    df = pd.read_csv(raw_data_path, header=None, 
                     names=["user", "activity", "timestamp", "x", "y", "z"], 
                     on_bad_lines='skip')
    
    # 清理z轴数据（去除分号）
    df["z"] = df["z"].astype(str).str.rstrip(";").astype(float)
    
    # 删除缺失值
    original_len = len(df)
    df.dropna(axis=0, how="any", inplace=True)
    print(f"数据清理: {original_len} -> {len(df)} 样本 (删除了 {original_len - len(df)} 个缺失值)")
    
    # 按用户和时间戳排序
    df.sort_values(by=["user", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def create_sliding_windows(df, window_size=128, step_size=64, min_samples_per_activity=100):
    """
    从原始数据创建滑动窗口，确保每个窗口内的运动类别和用户一致
    
    Args:
        df: 原始数据DataFrame
        window_size: 窗口大小（时间步数）
        step_size: 滑动步长
        min_samples_per_activity: 每个活动类型的最小样本数
        
    Returns:
        tuple: (windows, labels, window_info) - 窗口数据、对应标签和窗口信息
    """
    # 活动映射
    activity_mapping = {
        "Walking": 0,
        "Jogging": 1, 
        "Upstairs": 2,
        "Downstairs": 3,
        "Sitting": 4,
        "Standing": 5
    }
    
    print(f"创建滑动窗口: 窗口大小={window_size}, 步长={step_size}")
    
    windows = []
    labels = []
    window_info = []  # 存储每个窗口的用户和活动信息
    activity_counts = {}
    
    # 按用户分组处理
    for user_id, user_group in tqdm(df.groupby("user"), desc="处理用户数据"):
        # 按活动类型分组
        for activity, activity_group in user_group.groupby("activity"):
            if activity not in activity_mapping:
                print(f"跳过未知活动类型: {activity}")
                continue
                
            activity_label = activity_mapping[activity]
            activity_data = activity_group.sort_values(by="timestamp")
            
            # 创建滑动窗口，确保每个窗口内用户和活动类型一致
            user_activity_windows = 0
            for i in range(0, len(activity_data) - window_size + 1, step_size):
                window_data = activity_data.iloc[i:i + window_size]
                
                # 检查窗口内用户和活动类型的一致性
                window_users = window_data["user"].unique()
                window_activities = window_data["activity"].unique()
                
                if len(window_users) == 1 and len(window_activities) == 1:
                    # 确保窗口内用户和活动类型一致
                    
                    # 检查时间连续性（放宽时间容忍度，因为WISDM数据采样率不稳定）
                    time_diff = (window_data["timestamp"].iloc[-1] - window_data["timestamp"].iloc[0]) / 1e9
                    expected_duration = (window_size - 1) * (1 / 20.0)  # 20Hz采样率
                    
                    # 放宽时间容忍度到5倍，因为实际数据采样率不稳定
                    if time_diff < expected_duration * 5.0:  
                        # 提取加速度数据
                        accel_data = window_data[["x", "y", "z"]].values  # shape: (window_size, 3)
                        windows.append(accel_data)
                        labels.append(activity_label)
                        
                        # 记录窗口信息
                        window_info.append({
                            'user_id': user_id,
                            'activity': activity,
                            'activity_label': activity_label,
                            'start_timestamp': window_data["timestamp"].iloc[0],
                            'end_timestamp': window_data["timestamp"].iloc[-1]
                        })
                        
                        user_activity_windows += 1
            
            # 统计每个活动的窗口数量
            if activity not in activity_counts:
                activity_counts[activity] = 0
            activity_counts[activity] += user_activity_windows
    
    print(f"\n各活动类型的窗口数量:")
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count}")
    
    # 过滤样本数量过少的活动类型
    valid_activities = [act for act, count in activity_counts.items() if count >= min_samples_per_activity]
    print(f"\n保留的活动类型 (>={min_samples_per_activity}样本): {valid_activities}")
    
    # 过滤数据
    filtered_windows = []
    filtered_labels = []
    filtered_window_info = []
    for window, label, info in zip(windows, labels, window_info):
        if info['activity'] in valid_activities:
            filtered_windows.append(window)
            filtered_labels.append(label)
            filtered_window_info.append(info)
    
    # 重新映射标签为连续的整数
    unique_labels = sorted(list(set(filtered_labels)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = [label_mapping[label] for label in filtered_labels]
    
    print(f"\n标签重映射:")
    for old_label, new_label in label_mapping.items():
        activity_name = [k for k, v in activity_mapping.items() if v == old_label][0]
        print(f"  {activity_name}: {old_label} -> {new_label}")
    
    windows = np.array(filtered_windows)
    labels = np.array(remapped_labels)
    
    print(f"\n最终数据集: {len(windows)} 个窗口, {len(unique_labels)} 个类别")
    print(f"窗口形状: {windows.shape}")
    
    # 验证数据一致性
    print(f"\n数据一致性验证:")
    user_activity_combinations = set()
    for info in filtered_window_info:
        user_activity_combinations.add((info['user_id'], info['activity']))
    print(f"  用户-活动组合数: {len(user_activity_combinations)}")
    
    return windows, labels, filtered_window_info


def normalize_data(X_train, X_val=None, X_test=None):
    """
    对数据进行归一化处理
    
    Args:
        X_train: 训练数据
        X_val: 验证数据（可选）
        X_test: 测试数据（可选）
        
    Returns:
        tuple: 归一化后的数据
    """
    print("正在进行数据归一化...")
    
    # 检查数据是否为空
    if len(X_train) == 0:
        print("警告: 训练数据为空，无法进行归一化")
        return (X_train,) if X_val is None and X_test is None else (X_train, X_val, X_test)
    
    # 计算训练集的均值和标准差（按通道计算）
    # X_train shape: (N, T, C)
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)  # shape: (1, 1, C)
    std = np.std(X_train, axis=(0, 1), keepdims=True)    # shape: (1, 1, C)
    
    # 避免除零
    std = np.where(std == 0, 1.0, std)
    
    print(f"归一化参数:")
    for c in range(mean.shape[-1]):
        print(f"  通道{c}: mean={mean[0,0,c]:.4f}, std={std[0,0,c]:.4f}")
    
    # 归一化
    X_train_norm = (X_train - mean) / std
    
    results = [X_train_norm]
    
    if X_val is not None:
        X_val_norm = (X_val - mean) / std
        results.append(X_val_norm)
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        results.append(X_test_norm)
    
    return tuple(results) if len(results) > 1 else results[0]


def load_wisdm_split(data_dir, split='train', window_size=128, step_size=64, 
                     train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, 
                     random_state=42, user_split=False, save_split=True):
    """
    加载WISDM数据集的指定分割
    
    Args:
        data_dir: 数据目录
        split: 'train', 'val', 或 'test'
        window_size: 滑动窗口大小
        step_size: 滑动窗口步长
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        user_split: 是否按用户分割（False表示按样本分层分割）
        save_split: 是否保存分割结果
        
    Returns:
        tuple: (X, y) - 数据和标签
    """
    import os
    import pickle
    
    # 创建分割缓存文件路径
    split_cache_dir = os.path.join(data_dir, "processed_cache")
    os.makedirs(split_cache_dir, exist_ok=True)
    
    split_params = f"ws{window_size}_ss{step_size}_tr{train_ratio}_vr{val_ratio}_te{test_ratio}_rs{random_state}_us{user_split}"
    split_cache_file = os.path.join(split_cache_dir, f"split_indices_{split_params}.pkl")
    
    # 尝试加载已保存的分割结果
    if os.path.exists(split_cache_file):
        print(f"加载已保存的分割结果: {split_cache_file}")
        with open(split_cache_file, 'rb') as f:
            split_data = pickle.load(f)
        
        train_indices = split_data['train_indices']
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']
        windows = split_data['windows']
        labels = split_data['labels']
        window_info = split_data['window_info']
        
        print(f"从缓存加载的分割结果:")
        print(f"  训练集: {len(train_indices)} 样本")
        print(f"  验证集: {len(val_indices)} 样本")
        print(f"  测试集: {len(test_indices)} 样本")
        
    else:
        print(f"未找到缓存文件，重新创建分割...")
        
        # 加载原始数据
        df = load_wisdm_raw_data(data_dir)
        
        # 创建滑动窗口
        windows, labels, window_info = create_sliding_windows(df, window_size, step_size)
        
        if user_split:
            # 按用户分割数据（保留原有逻辑）
            print("按用户分割数据...")
            unique_users = df['user'].unique()
            n_users = len(unique_users)
            
            # 计算每个分割的用户数量
            n_train_users = max(1, int(n_users * train_ratio))
            n_val_users = max(1, int(n_users * val_ratio))
            n_test_users = n_users - n_train_users - n_val_users
            
            # 随机分配用户
            np.random.seed(random_state)
            shuffled_users = np.random.permutation(unique_users)
            
            train_users = shuffled_users[:n_train_users]
            val_users = shuffled_users[n_train_users:n_train_users + n_val_users]
            test_users = shuffled_users[n_train_users + n_val_users:]
            
            print(f"用户分割: 训练={len(train_users)}, 验证={len(val_users)}, 测试={len(test_users)}")
            print(f"训练用户: {train_users}")
            print(f"验证用户: {val_users}")
            print(f"测试用户: {test_users}")
            
            # 根据用户分割创建数据索引
            train_indices = []
            val_indices = []
            test_indices = []
            
            for i, info in enumerate(window_info):
                user_id = info['user_id']
                if user_id in train_users:
                    train_indices.append(i)
                elif user_id in val_users:
                    val_indices.append(i)
                elif user_id in test_users:
                    test_indices.append(i)
            
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)
            
        else:
            # 按样本分层分割数据（新的默认方式）
            print("按类别进行分层分割数据...")
            indices = np.arange(len(windows))
            
            # 分层分割以保持类别平衡
            train_indices, temp_indices, train_labels, temp_labels = train_test_split(
                indices, labels, test_size=(val_ratio + test_ratio), 
                random_state=random_state, stratify=labels
            )
            
            if val_ratio > 0 and test_ratio > 0:
                val_size = val_ratio / (val_ratio + test_ratio)
                if val_size > 0 and val_size < 1:
                    val_indices, test_indices, _, _ = train_test_split(
                        temp_indices, temp_labels, test_size=(1 - val_size),
                        random_state=random_state, stratify=temp_labels
                    )
                else:
                    # 如果val_size为0或1，直接分配
                    if val_size >= 1:
                        val_indices = temp_indices
                        test_indices = []
                    else:
                        val_indices = []
                        test_indices = temp_indices
            elif val_ratio > 0:
                val_indices = temp_indices
                test_indices = []
            elif test_ratio > 0:
                val_indices = []
                test_indices = temp_indices
            else:
                val_indices = []
                test_indices = []
            
            print(f"分层分割结果:")
            print(f"  训练集: {len(train_indices)} 样本")
            print(f"  验证集: {len(val_indices)} 样本")
            print(f"  测试集: {len(test_indices)} 样本")
            
            # 检查类别分布
            for split_name, split_indices in [("训练集", train_indices), ("验证集", val_indices), ("测试集", test_indices)]:
                if len(split_indices) > 0:
                    split_labels = labels[split_indices]
                    unique_labels, counts = np.unique(split_labels, return_counts=True)
                    print(f"  {split_name}类别分布: {dict(zip(unique_labels, counts))}")
        
        # 保存分割结果
        if save_split:
            split_data = {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'windows': windows,
                'labels': labels,
                'window_info': window_info,
                'split_params': {
                    'window_size': window_size,
                    'step_size': step_size,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'random_state': random_state,
                    'user_split': user_split
                }
            }
            
            with open(split_cache_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"分割结果已保存到: {split_cache_file}")
    
    # 根据分割类型返回对应数据
    if split == 'train':
        return windows[train_indices], labels[train_indices]
    elif split == 'val':
        return windows[val_indices], labels[val_indices]
    elif split == 'test':
        return windows[test_indices], labels[test_indices]
    else:
        raise ValueError(f"未知的分割类型: {split}")


def create_train_val_loaders(data_dir, batch_size=64, val_split=0.1, window_size=128, 
                             step_size=64, num_workers=0, random_state=42):
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        val_split: 验证集比例
        window_size: 滑动窗口大小
        step_size: 滑动窗口步长（设置为window_size//2实现50%重叠）
        num_workers: 数据加载器工作进程数
        random_state: 随机种子
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    print("=== 创建训练和验证数据加载器 ===")
    
    # 计算训练集、验证集、测试集比例 (7:1:2)
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    # 加载训练和验证数据（使用分层分割）
    X_train, y_train = load_wisdm_split(
        data_dir, 'train', window_size, step_size, 
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state, user_split=False
    )
    
    X_val, y_val = load_wisdm_split(
        data_dir, 'val', window_size, step_size,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state, user_split=False
    )
    
    # 数据归一化
    X_train_norm, X_val_norm = normalize_data(X_train, X_val)
    
    # 创建数据集
    train_dataset = WISDMWindowedDataset(X_train_norm, y_train)
    val_dataset = WISDMWindowedDataset(X_val_norm, y_val)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    num_classes = len(np.unique(y_train))
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"类别数: {num_classes}")
    
    return train_loader, val_loader, num_classes


def create_test_loader(data_dir, batch_size=64, window_size=128, step_size=64, 
                       num_workers=0, random_state=42):
    """
    创建测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        window_size: 滑动窗口大小
        step_size: 滑动窗口步长（设置为window_size//2实现50%重叠）
        num_workers: 数据加载器工作进程数
        random_state: 随机种子
        
    Returns:
        tuple: (test_loader, num_classes)
    """
    print("=== 创建测试数据加载器 ===")
    
    # 使用固定的7:1:2比例
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    # 加载测试数据（使用分层分割）
    X_test, y_test = load_wisdm_split(
        data_dir, 'test', window_size, step_size,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state, user_split=False
    )
    
    # 为了归一化，需要加载训练数据来计算统计量
    X_train, _ = load_wisdm_split(
        data_dir, 'train', window_size, step_size,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state, user_split=False
    )
    
    # 数据归一化
    _, X_test_norm = normalize_data(X_train, X_test)
    
    # 创建数据集和加载器
    test_dataset = WISDMWindowedDataset(X_test_norm, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    num_classes = len(np.unique(y_test))
    
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"类别数: {num_classes}")
    
    return test_loader, num_classes


if __name__ == "__main__":
    # 测试数据加载
    data_dir = "dataset/WISDM"
    
    print("=== WISDM数据集测试 ===")
    
    # 创建训练和验证数据加载器
    train_loader, val_loader, num_classes = create_train_val_loaders(
        data_dir, batch_size=32, val_split=0.2, window_size=128, step_size=64
    )
    
    # 创建测试数据加载器
    test_loader, _ = create_test_loader(
        data_dir, batch_size=32, window_size=128, step_size=64
    )
    
    print(f"\n数据加载器信息:")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    print(f"类别数: {num_classes}")
    
    # 检查数据形状
    print(f"\n数据形状检查:")
    for name, loader in [("训练", train_loader), ("验证", val_loader), ("测试", test_loader)]:
        for X_batch, y_batch in loader:
            print(f"{name}集 - X: {X_batch.shape}, y: {y_batch.shape}")
            print(f"{name}集 - 标签范围: {y_batch.min().item()} - {y_batch.max().item()}")
            break
    
    print("\n=== 数据集测试完成 ===")