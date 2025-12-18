import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import glob

class PAMAP2WindowedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, C, T) - 已经是正确格式
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PAMAP2活动标签映射（根据数据集文档）
ACTIVITY_MAP = {
    0: 'other',           # other (transient activities)
    1: 'lying',           # lying
    2: 'sitting',         # sitting
    3: 'standing',        # standing
    4: 'walking',         # walking
    5: 'running',         # running
    6: 'cycling',         # cycling
    7: 'Nordic_walking',  # Nordic walking
    9: 'watching_TV',     # watching TV
    10: 'computer_work',  # computer work
    11: 'car_driving',    # car driving
    12: 'ascending_stairs', # ascending stairs
    13: 'descending_stairs', # descending stairs
    16: 'vacuum_cleaning', # vacuum cleaning
    17: 'ironing',        # ironing
    18: 'folding_laundry', # folding laundry
    19: 'house_cleaning', # house cleaning
    20: 'playing_soccer', # playing soccer
    24: 'rope_jumping'    # rope jumping
}

# 重新映射到连续的类别ID，只包含DataCollectionProtocol.pdf中的12个标准类别
ACTIVITY_ID_MAP = {
    1: 0,   # lying -> 0
    2: 1,   # sitting -> 1
    3: 2,   # standing -> 2
    17: 3,  # ironing -> 3
    16: 4,  # vacuum_cleaning -> 4
    12: 5,  # ascending_stairs -> 5
    13: 6,  # descending_stairs -> 6
    4: 7,   # walking -> 7
    7: 8,   # Nordic_walking -> 8
    6: 9,   # cycling -> 9
    5: 10,  # running -> 10
    24: 11  # rope_jumping -> 11
}

ACTIVITY_NAMES = [
    'lying', 'sitting', 'standing', 'ironing', 'vacuum_cleaning',
    'ascending_stairs', 'descending_stairs', 'walking', 'Nordic_walking',
    'cycling', 'running', 'rope_jumping'
]

def load_pamap2_data(data_dir, subjects=None, window_size=256, step=128, show_subject_distribution=False):
    """
    载入PAMAP2数据，做滑窗切分，并对每个通道归一化。
    修复版本：确保滑窗内数据来自同一活动且时间连续。
    
    Args:
        data_dir: PAMAP2数据目录路径
        subjects: 要加载的受试者列表，如果为None则加载所有受试者
        window_size: 滑动窗口大小
        step: 滑动窗口步长
        show_subject_distribution: 是否显示受试者活动分布
    
    Returns:
        X: 特征数据 (N, C, T)
        y: 标签数据 (N,)
    """
    if subjects is None:
        # 加载Protocol目录下的所有受试者数据
        subject_files = glob.glob(os.path.join(data_dir, 'Protocol', 'subject*.dat'))
        subjects = [os.path.basename(f).replace('subject', '').replace('.dat', '') for f in subject_files]
    
    all_X = []
    all_y = []
    all_subject_ids = []
    
    for subject in subjects:
        file_path = os.path.join(data_dir, 'Protocol', f'subject{subject}.dat')
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        print(f"Loading subject {subject}...")
        
        # 读取数据文件，按时间序列组织
        timestamps = []
        sensor_data_list = []
        activity_labels = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    values = line.split()
                    if len(values) == 54:  # 确保数据完整性
                        try:
                            # 解析数据：时间戳, 活动ID, 心率, 温度, 然后是传感器数据
                            timestamp = float(values[0])
                            activity_id = int(values[1])
                            
                            # 只保留有效的活动ID，排除原始other类别(0)
                            if activity_id in ACTIVITY_ID_MAP and activity_id != 0:
                                # 提取IMU传感器数据（只取加速度和角速度）
                                # 使用中位数填充NaN值，避免引入0偏差
                                def safe_float(val):
                                    return float(val) if val != 'NaN' else np.nan
                                
                                # 提取加速度数据 (3个IMU × 3轴 = 9个特征)
                                acc_hand = [safe_float(values[i]) for i in [5, 6, 7]]  # hand acc
                                acc_chest = [safe_float(values[i]) for i in [22, 23, 24]]  # chest acc
                                acc_ankle = [safe_float(values[i]) for i in [39, 40, 41]]  # ankle acc
                                
                                # 提取角速度数据 (3个IMU × 3轴 = 9个特征)
                                gyro_hand = [safe_float(values[i]) for i in [11, 12, 13]]  # hand gyro
                                gyro_chest = [safe_float(values[i]) for i in [28, 29, 30]]  # chest gyro
                                gyro_ankle = [safe_float(values[i]) for i in [45, 46, 47]]  # ankle gyro
                                
                                # 合并所有IMU数据 (18个特征)
                                sensor_data = acc_hand + acc_chest + acc_ankle + gyro_hand + gyro_chest + gyro_ankle
                                
                                timestamps.append(timestamp)
                                sensor_data_list.append(sensor_data)
                                activity_labels.append(ACTIVITY_ID_MAP[activity_id])
                        except (ValueError, IndexError):
                            continue
        
        if not sensor_data_list:
            print(f"  No valid data found for subject {subject}")
            continue
            
        # 转换为numpy数组
        timestamps = np.array(timestamps)
        sensor_data_array = np.array(sensor_data_list)  # (T, 18)
        activity_labels = np.array(activity_labels)
        
        # 处理缺失值：使用前向填充和后向填充
        for i in range(sensor_data_array.shape[1]):
            channel_data = sensor_data_array[:, i]
            # 找到非NaN的值
            valid_mask = ~np.isnan(channel_data)
            if np.any(valid_mask):
                # 前向填充
                last_valid = None
                for j in range(len(channel_data)):
                    if valid_mask[j]:
                        last_valid = channel_data[j]
                    elif last_valid is not None:
                        channel_data[j] = last_valid
                
                # 后向填充（处理开头的NaN）
                first_valid = None
                for j in range(len(channel_data)-1, -1, -1):
                    if valid_mask[j]:
                        first_valid = channel_data[j]
                    elif first_valid is not None:
                        channel_data[j] = first_valid
                
                sensor_data_array[:, i] = channel_data
        
        print(f"  Loaded {len(sensor_data_array)} samples from subject {subject}")
        
        # 按受试者归一化数据
        for c in range(sensor_data_array.shape[1]):
            mean = np.nanmean(sensor_data_array[:, c])
            std = np.nanstd(sensor_data_array[:, c])
            sensor_data_array[:, c] = (sensor_data_array[:, c] - mean) / (std + 1e-8)
        
        # 滑窗切分：确保窗口内活动一致且时间连续
        subject_X, subject_y = [], []
        
        i = 0
        while i + window_size <= len(sensor_data_array):
            # 检查时间连续性（采样率约100Hz，允许一定误差）
            time_window = timestamps[i:i+window_size]
            time_diffs = np.diff(time_window)
            expected_interval = 0.01  # 100Hz = 0.01s间隔
            
            # 检查时间间隔是否合理（允许±50%误差）
            valid_time = np.all(time_diffs < expected_interval * 1.5)
            
            # 检查活动一致性
            activity_window = activity_labels[i:i+window_size]
            activity_consistent = len(np.unique(activity_window)) == 1
            
            if valid_time and activity_consistent:
                # 提取窗口数据并转置为 (C, T) 格式
                window_data = sensor_data_array[i:i+window_size].T  # (18, window_size)
                window_label = activity_window[0]
                
                subject_X.append(window_data)
                subject_y.append(window_label)
                
                i += step  # 正常步进
            else:
                # 如果时间不连续或活动不一致，跳到下一个可能的起始点
                i += 1
        
        if subject_X:
            all_X.extend(subject_X)
            all_y.extend(subject_y)
            all_subject_ids.extend([int(subject)] * len(subject_X))
            
            # 显示每个受试者的类别分布
            if show_subject_distribution:
                subject_unique, subject_counts = np.unique(subject_y, return_counts=True)
                print(f"  Subject {subject} activity distribution ({len(subject_X)} windows):")
                for label, count in zip(subject_unique, subject_counts):
                    if label < len(ACTIVITY_NAMES):
                        activity_name = ACTIVITY_NAMES[label]
                    else:
                        activity_name = f'unknown_{label}'
                    print(f"    - {activity_name}: {count} windows")
    
    if not all_X:
        raise ValueError("No valid windowed data found!")
    
    X = np.stack(all_X)  # (N, C, T)
    y = np.array(all_y)
    subject_ids = np.array(all_subject_ids)
    
    print(f"\nTotal windowed samples: {len(X)}")
    print(f"Feature shape: {X.shape} (N, C, T)")
    print(f"Subjects included: {np.unique(subject_ids)}")
    
    # 显示总体类别分布
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print(f"\nOverall activity distribution:")
    for label, count in zip(unique_labels, label_counts):
        if label < len(ACTIVITY_NAMES):
            activity_name = ACTIVITY_NAMES[label]
        else:
            activity_name = f'unknown_{label}'
        print(f"  - {activity_name}: {count} windows")
    
    return X, y, subject_ids

def validate_data_quality(X, y, subject_ids, timestamps_list=None):
    """
    验证数据质量，检查时间连续性和活动一致性
    
    Args:
        X: 特征数据 (N, C, T)
        y: 标签数据 (N,)
        subject_ids: 受试者ID (N,)
        timestamps_list: 时间戳列表（可选）
    
    Returns:
        quality_report: 数据质量报告字典
    """
    print("\n=== 数据质量检查 ===")
    
    quality_report = {
        'total_windows': len(X),
        'subjects': np.unique(subject_ids).tolist(),
        'activities': np.unique(y).tolist(),
        'data_shape': X.shape,
        'issues': []
    }
    
    # 检查数据形状一致性
    if len(X) != len(y) or len(X) != len(subject_ids):
        issue = f"数据长度不一致: X={len(X)}, y={len(y)}, subjects={len(subject_ids)}"
        quality_report['issues'].append(issue)
        print(f"❌ {issue}")
    
    # 检查是否有NaN值
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        issue = f"发现 {nan_count} 个NaN值"
        quality_report['issues'].append(issue)
        print(f"❌ {issue}")
    else:
        print(f"✅ 无NaN值")
    
    # 检查是否有无穷值
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        issue = f"发现 {inf_count} 个无穷值"
        quality_report['issues'].append(issue)
        print(f"❌ {issue}")
    else:
        print(f"✅ 无无穷值")
    
    # 检查每个受试者的数据分布
    print(f"\n受试者数据分布:")
    for subject in quality_report['subjects']:
        subject_mask = subject_ids == subject
        subject_count = np.sum(subject_mask)
        subject_activities = np.unique(y[subject_mask])
        print(f"  受试者 {subject}: {subject_count} 个窗口, {len(subject_activities)} 种活动")
    
    # 检查活动分布
    print(f"\n活动分布:")
    for activity in quality_report['activities']:
        activity_count = np.sum(y == activity)
        activity_name = ACTIVITY_NAMES[activity] if activity < len(ACTIVITY_NAMES) else f'unknown_{activity}'
        print(f"  活动 {activity} ({activity_name}): {activity_count} 个窗口")
    
    # 检查数据范围
    data_min = np.min(X)
    data_max = np.max(X)
    data_mean = np.mean(X)
    data_std = np.std(X)
    
    print(f"\n数据统计:")
    print(f"  最小值: {data_min:.4f}")
    print(f"  最大值: {data_max:.4f}")
    print(f"  均值: {data_mean:.4f}")
    print(f"  标准差: {data_std:.4f}")
    
    # 检查是否有异常值（超过3个标准差）
    outlier_threshold = 3
    outliers = np.abs(X - data_mean) > (outlier_threshold * data_std)
    outlier_count = np.sum(outliers)
    outlier_ratio = outlier_count / X.size * 100
    
    if outlier_ratio > 5:  # 如果异常值超过5%
        issue = f"异常值比例过高: {outlier_ratio:.2f}% ({outlier_count}/{X.size})"
        quality_report['issues'].append(issue)
        print(f"⚠️  {issue}")
    else:
        print(f"✅ 异常值比例正常: {outlier_ratio:.2f}%")
    
    quality_report.update({
        'data_min': data_min,
        'data_max': data_max,
        'data_mean': data_mean,
        'data_std': data_std,
        'outlier_ratio': outlier_ratio
    })
    
    # 总结
    if not quality_report['issues']:
        print(f"\n✅ 数据质量检查通过，未发现问题")
    else:
        print(f"\n❌ 发现 {len(quality_report['issues'])} 个问题:")
        for i, issue in enumerate(quality_report['issues'], 1):
            print(f"  {i}. {issue}")
    
    return quality_report

def create_train_val_loaders(data_dir, batch_size=64, window_size=256, step=128, 
                           train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42,
                           subject_split=False, save_split=True, split_file=None):
    """
    创建训练集、验证集和测试集数据加载器。
    按7:1:2比例进行分层采样，排除other类别（类别0）。
    可以选择保存分割结果到文件，避免重复加载。
    
    Args:
        data_dir: PAMAP2数据目录路径
        batch_size: 批次大小
        window_size: 滑动窗口大小
        step: 滑动窗口步长
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        subject_split: 是否按受试者分割（避免数据泄露）
        save_split: 是否保存分割结果
        split_file: 分割结果保存文件路径
    """
    import pickle
    
    # 设置默认分割文件路径
    if split_file is None:
        split_mode = "subject" if subject_split else "stratified"
        split_file = os.path.join(data_dir, f"pamap2_split_{split_mode}_seed{seed}.pkl")
    
    # 检查是否已有保存的分割结果
    if os.path.exists(split_file):
        print(f"📁 发现已保存的数据分割文件: {split_file}")
        print("🔄 加载已保存的数据分割...")
        
        try:
            with open(split_file, 'rb') as f:
                split_data = pickle.load(f)
            
            X_train, y_train = split_data['X_train'], split_data['y_train']
            X_val, y_val = split_data['X_val'], split_data['y_val']
            X_test, y_test = split_data['X_test'], split_data['y_test']
            
            print(f"✅ 成功加载已保存的数据分割")
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  验证集: {len(X_val)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            
            # 创建数据加载器
            train_dataset = PAMAP2WindowedDataset(X_train, y_train)
            val_dataset = PAMAP2WindowedDataset(X_val, y_val)
            test_dataset = PAMAP2WindowedDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"⚠️ 加载保存的分割文件失败: {e}")
            print("🔄 将重新进行数据分割...")
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 加载所有受试者的数据
    all_subjects = ['101', '102', '103', '104', '105', '106', '107', '108', '109']
    print("Loading all data...")
    X_all, y_all, subject_ids = load_pamap2_data(data_dir, subjects=all_subjects, 
                                                 window_size=window_size, step=step, 
                                                 show_subject_distribution=True)
    
    # 分析原始数据中的类别分布
    print(f"原始数据: {len(X_all)} 样本")
    print("原始数据类别分布:")
    original_unique, original_counts = np.unique(y_all, return_counts=True)
    for label, count in zip(original_unique, original_counts):
        if label < len(ACTIVITY_NAMES):
            activity_name = ACTIVITY_NAMES[label]
        else:
            activity_name = f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    # 不再需要排除类别0，因为在数据加载时已经排除了原始other类别
    # valid_mask = y_all != 0  # 排除类别0 (other)
    # X_all = X_all[valid_mask]
    # y_all = y_all[valid_mask]
    # subject_ids = subject_ids[valid_mask]  # 同时过滤subject_ids
    print(f"包含所有有效类别: {len(X_all)} 样本")
    
    if subject_split:
        # 按受试者分割，避免数据泄露
        unique_subjects = np.unique(subject_ids)
        np.random.shuffle(unique_subjects)
        
        n_subjects = len(unique_subjects)
        n_train = int(n_subjects * train_ratio)
        n_val = max(1, int(n_subjects * val_ratio))  # 确保至少有1个受试者用于验证
        
        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train:n_train+n_val]
        test_subjects = unique_subjects[n_train+n_val:]
        
        # 根据受试者分割数据
        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        
        print(f"Subject-based split:")
        print(f"  Train subjects: {sorted(train_subjects)} ({len(X_train)} samples)")
        print(f"  Val subjects: {sorted(val_subjects)} ({len(X_val)} samples)")
        print(f"  Test subjects: {sorted(test_subjects)} ({len(X_test)} samples)")
    else:
        if subject_split:
            # 按受试者分割，避免数据泄露
            unique_subjects = np.unique(subject_ids)
            np.random.shuffle(unique_subjects)
            
            n_subjects = len(unique_subjects)
            n_train = int(n_subjects * train_ratio)
            n_val = int(n_subjects * val_ratio)
            
            train_subjects = unique_subjects[:n_train]
            val_subjects = unique_subjects[n_train:n_train+n_val]
            test_subjects = unique_subjects[n_train+n_val:]
            
            # 根据受试者分割数据
            train_mask = np.isin(subject_ids, train_subjects)
            val_mask = np.isin(subject_ids, val_subjects)
            test_mask = np.isin(subject_ids, test_subjects)
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_val, y_val = X_all[val_mask], y_all[val_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            print(f"Subject-based split:")
            print(f"  Train subjects: {sorted(train_subjects)} ({len(X_train)} samples)")
            print(f"  Val subjects: {sorted(val_subjects)} ({len(X_val)} samples)")
            print(f"  Test subjects: {sorted(test_subjects)} ({len(X_test)} samples)")
        else:
            # 按类别分层采样
            unique_classes = np.unique(y_all)
            print(f"有效类别: {unique_classes}")
            
            X_train_list, y_train_list = [], []
            X_val_list, y_val_list = [], []
            X_test_list, y_test_list = [], []
            
            for class_id in unique_classes:
                # 获取当前类别的所有样本
                class_mask = y_all == class_id
                X_class = X_all[class_mask]
                y_class = y_all[class_mask]
                
                # 计算每个集合的样本数
                n_samples = len(X_class)
                n_train = int(n_samples * train_ratio)
                n_val = int(n_samples * val_ratio)
                n_test = n_samples - n_train - n_val  # 剩余的全部给测试集
                
                # 随机打乱
                indices = np.random.permutation(n_samples)
                
                # 分割数据
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]
                
                # 添加到对应列表
                X_train_list.append(X_class[train_indices])
                y_train_list.append(y_class[train_indices])
                X_val_list.append(X_class[val_indices])
                y_val_list.append(y_class[val_indices])
                X_test_list.append(X_class[test_indices])
                y_test_list.append(y_class[test_indices])
                
                print(f"类别 {class_id} ({ACTIVITY_NAMES[class_id]}): 总样本={n_samples}, "
                      f"训练={n_train}, 验证={n_val}, 测试={n_test}")
            
            # 合并所有类别的数据
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)
            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
    
    # 计算类别分布
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    val_unique, val_counts = np.unique(y_val, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    print(f"\n训练集类别分布:")
    for label, count in zip(train_unique, train_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    print(f"\n验证集类别分布:")
    for label, count in zip(val_unique, val_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    print(f"\n测试集类别分布:")
    for label, count in zip(test_unique, test_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    # 创建数据集
    train_dataset = PAMAP2WindowedDataset(X_train, y_train)
    val_dataset = PAMAP2WindowedDataset(X_val, y_val)
    test_dataset = PAMAP2WindowedDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 保存分割结果
    if save_split:
        try:
            split_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'split_params': {
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'seed': seed,
                    'subject_split': subject_split,
                    'window_size': window_size,
                    'step': step
                }
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            
            with open(split_file, 'wb') as f:
                pickle.dump(split_data, f)
            
            print(f"💾 数据分割结果已保存到: {split_file}")
            
        except Exception as e:
            print(f"⚠️ 保存分割文件失败: {e}")
    
    return train_loader, val_loader, test_loader

def create_train_val_test_loaders(data_dir, batch_size=64, window_size=256, step=128, 
                                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42,
                                 save_split=True, split_file=None, subject_split=False):
    """
    创建训练集、验证集和测试集数据加载器。
    按8:1:1比例进行分层采样，排除other类别（类别0）。
    可以选择保存分割结果到文件，避免重复加载。
    
    Args:
        data_dir: PAMAP2数据目录路径
        batch_size: 批次大小
        window_size: 滑动窗口大小
        step: 滑动窗口步长
        train_ratio: 训练集比例（默认0.8）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.1）
        seed: 随机种子
        save_split: 是否保存分割结果
        split_file: 分割结果保存文件路径
        subject_split: 是否按受试者分割（默认False，使用分层抽样）
    """
    import os
    import pickle
    
    # 设置默认分割文件路径
    if split_file is None:
        split_file = os.path.join(data_dir, f"pamap2_split_seed{seed}.pkl")
    
    # 检查是否已有保存的分割结果
    if os.path.exists(split_file):
        print(f"📁 发现已保存的数据分割文件: {split_file}")
        print("🔄 加载已保存的数据分割...")
        
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        
        X_train, y_train = split_data['X_train'], split_data['y_train']
        X_val, y_val = split_data['X_val'], split_data['y_val']
        X_test, y_test = split_data['X_test'], split_data['y_test']
        
        print(f"✅ 成功加载已保存的数据分割")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
    else:
        # 设置随机种子
        np.random.seed(seed)
        
        # 加载所有受试者的数据
        all_subjects = ['101', '102', '103', '104', '105', '106', '107', '108', '109']
        print("Loading all data...")
        X_all, y_all, subject_ids = load_pamap2_data(data_dir, subjects=all_subjects, 
                                                     window_size=window_size, step=step, 
                                                     show_subject_distribution=True)
        
        # 分析原始数据中的类别分布
        print(f"原始数据: {len(X_all)} 样本")
        print("原始数据类别分布:")
        original_unique, original_counts = np.unique(y_all, return_counts=True)
        for label, count in zip(original_unique, original_counts):
            if label < len(ACTIVITY_NAMES):
                activity_name = ACTIVITY_NAMES[label]
            else:
                activity_name = f'unknown_{label}'
            print(f"  类别 {label} ({activity_name}): {count} 样本")
        
        # 不再需要排除类别0，因为在数据加载时已经排除了原始other类别
        # valid_mask = y_all != 0  # 排除类别0 (other)
        # X_all = X_all[valid_mask]
        # y_all = y_all[valid_mask]
        # subject_ids = subject_ids[valid_mask]
        print(f"包含所有有效类别: {len(X_all)} 样本")
        
        if subject_split:
            # 按受试者分割，避免数据泄露
            unique_subjects = np.unique(subject_ids)
            np.random.shuffle(unique_subjects)
            
            n_subjects = len(unique_subjects)
            n_train = int(n_subjects * train_ratio)
            n_val = int(n_subjects * val_ratio)
            
            train_subjects = unique_subjects[:n_train]
            val_subjects = unique_subjects[n_train:n_train+n_val]
            test_subjects = unique_subjects[n_train+n_val:]
            
            # 根据受试者分割数据
            train_mask = np.isin(subject_ids, train_subjects)
            val_mask = np.isin(subject_ids, val_subjects)
            test_mask = np.isin(subject_ids, test_subjects)
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_val, y_val = X_all[val_mask], y_all[val_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            print(f"Subject-based split:")
            print(f"  Train subjects: {sorted(train_subjects)} ({len(X_train)} samples)")
            print(f"  Val subjects: {sorted(val_subjects)} ({len(X_val)} samples)")
            print(f"  Test subjects: {sorted(test_subjects)} ({len(X_test)} samples)")
        else:
            # 按类别分层采样
            unique_classes = np.unique(y_all)
            print(f"有效类别: {unique_classes}")
            
            X_train_list, y_train_list = [], []
            X_val_list, y_val_list = [], []
            X_test_list, y_test_list = [], []
            
            for class_id in unique_classes:
                # 获取当前类别的所有样本
                class_mask = y_all == class_id
                X_class = X_all[class_mask]
                y_class = y_all[class_mask]
                
                # 计算每个集合的样本数
                n_samples = len(X_class)
                n_train = int(n_samples * train_ratio)
                n_val = int(n_samples * val_ratio)
                n_test = n_samples - n_train - n_val  # 剩余的全部给测试集
                
                # 随机打乱
                indices = np.random.permutation(n_samples)
                
                # 分割数据
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]
                
                # 添加到对应列表
                X_train_list.append(X_class[train_indices])
                y_train_list.append(y_class[train_indices])
                X_val_list.append(X_class[val_indices])
                y_val_list.append(y_class[val_indices])
                X_test_list.append(X_class[test_indices])
                y_test_list.append(y_class[test_indices])
                
                print(f"类别 {class_id} ({ACTIVITY_NAMES[class_id]}): 总样本={n_samples}, "
                      f"训练={n_train}, 验证={n_val}, 测试={n_test}")
            
            # 合并所有类别的数据
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)
            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
        
        # 保存分割结果
        if save_split:
            print(f"💾 保存数据分割结果到: {split_file}")
            split_data = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'seed': seed,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio
            }
            with open(split_file, 'wb') as f:
                pickle.dump(split_data, f)
            print("✅ 数据分割结果已保存")
    
    # 计算类别分布
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    val_unique, val_counts = np.unique(y_val, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    print(f"\n训练集类别分布:")
    for label, count in zip(train_unique, train_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    print(f"\n验证集类别分布:")
    for label, count in zip(val_unique, val_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    print(f"\n测试集类别分布:")
    for label, count in zip(test_unique, test_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    # 创建数据集
    train_dataset = PAMAP2WindowedDataset(X_train, y_train)
    val_dataset = PAMAP2WindowedDataset(X_val, y_val)
    test_dataset = PAMAP2WindowedDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    return train_loader, val_loader, test_loader

def create_train_loader(data_dir, batch_size=64, window_size=256, step=128, seed=42):
    """
    创建训练数据加载器
    """
    train_loader, _, _ = create_train_val_loaders(data_dir, batch_size, window_size, step, seed=seed)
    return train_loader

def create_val_loader(data_dir, batch_size=64, window_size=256, step=128, seed=42):
    """
    创建验证数据加载器
    """
    _, val_loader, _ = create_train_val_loaders(data_dir, batch_size, window_size, step, seed=seed)
    return val_loader

def create_test_loader(data_dir, batch_size=64, seed=42, subject_split=None):
    """
    创建测试集数据加载器，直接加载已保存的分割结果中的测试集。
    
    Args:
        data_dir: PAMAP2数据目录路径
        batch_size: 批次大小
        seed: 随机种子（用于找到对应的分割文件）
        subject_split: 分割模式，None表示自动检测已有文件
    """
    import os
    import pickle
    
    # 如果指定了subject_split，直接构建对应的文件路径
    if subject_split is not None:
        split_mode = "subject" if subject_split else "stratified"
        split_file = os.path.join(data_dir, f"pamap2_split_{split_mode}_seed{seed}.pkl")
    else:
        # 自动检测已有的分割文件
        possible_files = [
            os.path.join(data_dir, f"pamap2_split_seed{seed}.pkl"),  # 旧格式（create_train_val_test_loaders）
            os.path.join(data_dir, f"pamap2_split_stratified_seed{seed}.pkl"),  # 分层采样
            os.path.join(data_dir, f"pamap2_split_subject_seed{seed}.pkl")  # 受试者分割
        ]
        
        split_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                split_file = file_path
                break
    
    if split_file is None or not os.path.exists(split_file):
        available_files = []
        for file_path in possible_files:
            if os.path.exists(file_path):
                available_files.append(file_path)
        
        error_msg = f"未找到数据分割文件。\n"
        if available_files:
            error_msg += f"可用的分割文件: {available_files}\n"
        else:
            error_msg += f"尝试查找的文件: {possible_files}\n"
        error_msg += f"请先运行训练脚本创建数据分割。"
        raise FileNotFoundError(error_msg)
    
    print(f"📁 加载测试集数据: {split_file}")
    
    # 加载分割结果
    with open(split_file, 'rb') as f:
        split_data = pickle.load(f)
    
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    print(f"✅ 成功加载测试集数据: {len(X_test)} 样本")
    
    # 计算测试集类别分布
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print(f"\n测试集类别分布:")
    for label, count in zip(test_unique, test_counts):
        activity_name = ACTIVITY_NAMES[label] if label < len(ACTIVITY_NAMES) else f'unknown_{label}'
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    # 创建测试数据集
    test_dataset = PAMAP2WindowedDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 返回PAMAP2数据集的标准类别数量（12个类别）
    num_classes = len(ACTIVITY_NAMES)  # 12个类别
    return test_loader, num_classes


if __name__ == "__main__":
    data_dir = "dataset/PAMAP2"
    
    # 测试数据加载
    print("=== 测试修改后的PAMAP2数据加载器 ===")
    train_loader, val_loader, test_loader = create_train_val_test_loaders(data_dir, batch_size=32)
    
    print(f"\n数据加载器信息:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")
    
    # 进行数据质量检查
    print("\n=== 数据质量检查 ===")
    # 获取一些样本数据进行质量检查
    train_iter = iter(train_loader)
    X_sample, y_sample = next(train_iter)
    
    # 模拟受试者ID（实际使用中应该从数据加载器获取）
    sample_subject_ids = np.random.choice([101, 102, 103], size=len(y_sample))
    
    # 执行质量检查
    quality_report = validate_data_quality(X_sample.numpy(), y_sample.numpy(), sample_subject_ids)
    
    print(f"\n=== 质量检查结果 ===")
    print(f"检查的样本数: {quality_report['total_windows']}")
    print(f"数据形状: {quality_report['data_shape']}")
    if quality_report['issues']:
        print(f"发现问题: {len(quality_report['issues'])} 个")
    else:
        print(f"✅ 所有检查项目通过")
    
    # 获取训练集前2个样本
    train_iter = iter(train_loader)
    X_train_batch, y_train_batch = next(train_iter)
    for i in range(2):
        print(f"\n[Train Sample {i}]")
        print(f"  X shape: {X_train_batch[i].shape}")  # (C, T)
        print(f"  y label: {y_train_batch[i].item()}")
        activity_name = ACTIVITY_NAMES[y_train_batch[i].item()] if y_train_batch[i].item() < len(ACTIVITY_NAMES) else f'unknown_{y_train_batch[i].item()}'
        print(f"  Activity: {activity_name}")
        print(f"  X data (first 5 values of each channel):")
        for c in range(min(5, X_train_batch.shape[1])):  # 只显示前5个通道
            print(f"    Channel {c}: {X_train_batch[i][c][:5].tolist()}")
    
    # 获取验证集前2个样本
    val_iter = iter(val_loader)
    X_val_batch, y_val_batch = next(val_iter)
    for i in range(2):
        print(f"\n[Val Sample {i}]")
        print(f"  X shape: {X_val_batch[i].shape}")  # (C, T)
        print(f"  y label: {y_val_batch[i].item()}")
        activity_name = ACTIVITY_NAMES[y_val_batch[i].item()] if y_val_batch[i].item() < len(ACTIVITY_NAMES) else f'unknown_{y_val_batch[i].item()}'
        print(f"  Activity: {activity_name}")
        print(f"  X data (first 5 values of each channel):")
        for c in range(min(5, X_val_batch.shape[1])):  # 只显示前5个通道
            print(f"    Channel {c}: {X_val_batch[i][c][:5].tolist()}")
    
    # 获取测试集前2个样本
    test_iter = iter(test_loader)
    X_test_batch, y_test_batch = next(test_iter)
    for i in range(2):
        print(f"\n[Test Sample {i}]")
        print(f"  X shape: {X_test_batch[i].shape}")  # (C, T)
        print(f"  y label: {y_test_batch[i].item()}")
        activity_name = ACTIVITY_NAMES[y_test_batch[i].item()] if y_test_batch[i].item() < len(ACTIVITY_NAMES) else f'unknown_{y_test_batch[i].item()}'
        print(f"  Activity: {activity_name}")
        print(f"  X data (first 5 values of each channel):")
        for c in range(min(5, X_test_batch.shape[1])):  # 只显示前5个通道
            print(f"    Channel {c}: {X_test_batch[i][c][:5].tolist()}")
