import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

# MHEALTH数据集活动标签映射
# 基于MHEALTH数据集描述：12个物理活动，10个受试者，3个传感器设备
MHEALTH_ACTIVITY_LABELS = {
    0: "Null/Transition",  # 空值或过渡状态
    1: "Standing still",   # L1: 静止站立 (1分钟)
    2: "Sitting and relaxing",  # L2: 坐着放松 (1分钟)
    3: "Lying down",       # L3: 躺下 (1分钟)
    4: "Walking",          # L4: 步行 (1分钟)
    5: "Climbing stairs",  # L5: 爬楼梯 (1分钟)
    6: "Waist bends forward",  # L6: 腰部前弯 (20次)
    7: "Frontal elevation of arms",  # L7: 手臂前举 (20次)
    8: "Knees bending (crouching)",  # L8: 膝盖弯曲(蹲下) (20次)
    9: "Cycling",          # L9: 骑自行车 (1分钟)
    10: "Jogging",         # L10: 慢跑 (1分钟)
    11: "Running",         # L11: 跑步 (1分钟)
    12: "Jump front & back"  # L12: 前后跳跃 (20次)
}

# 传感器通道描述
# 数据采样率: 50Hz
# 传感器位置: 胸部、右手腕、左脚踝
SENSOR_CHANNELS = {
    # 胸部传感器 (Chest sensor)
    'chest_acc_x': 'alx',    # 胸部加速度计X轴
    'chest_acc_y': 'aly',    # 胸部加速度计Y轴  
    'chest_acc_z': 'alz',    # 胸部加速度计Z轴
    'chest_gyro_x': 'glx',   # 胸部陀螺仪X轴
    'chest_gyro_y': 'gly',   # 胸部陀螺仪Y轴
    'chest_gyro_z': 'glz',   # 胸部陀螺仪Z轴
    
    # 右手腕传感器 (Right wrist sensor)  
    'wrist_acc_x': 'arx',    # 右手腕加速度计X轴
    'wrist_acc_y': 'ary',    # 右手腕加速度计Y轴
    'wrist_acc_z': 'arz',    # 右手腕加速度计Z轴
    'wrist_gyro_x': 'grx',   # 右手腕陀螺仪X轴
    'wrist_gyro_y': 'gry',   # 右手腕陀螺仪Y轴
    'wrist_gyro_z': 'grz',   # 右手腕陀螺仪Z轴
}

class MHealthWindowedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_mhealth_data(data_path, window_size=128, step_size=64, exclude_null=True, test_subjects=None):
    """
    加载MHEALTH数据集并进行窗口化处理
    
    MHEALTH数据集描述:
    - 受试者: 10名志愿者 (subject1-subject10)
    - 传感器: 胸部和右手腕的加速度计+陀螺仪 (12通道)
    - 采样率: 50Hz
    - 活动类型: 12种 + 1种过渡状态 (共13类)
    
    窗口化参数:
    - window_size: 窗口大小 (默认128, 即2.56秒@50Hz)
    - step_size: 滑动步长 (默认64, 50%重叠)
    - exclude_null: 是否排除类别0(Null/Transition) (默认True, 推荐用于实验)
    
    传感器列名 (仅包含胸部和右手腕数据):
    - alx,aly,alz: 胸部加速度计 (chest accelerometer)
    - glx,gly,glz: 胸部陀螺仪 (chest gyroscope)  
    - arx,ary,arz: 右手腕加速度计 (right-arm accelerometer)
    - grx,gry,grz: 右手腕陀螺仪 (right-arm gyroscope)
    
    注意: exclude_null=True时，标签会重新映射为0-11，对应原始标签1-12
    """
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 过滤空值/过渡状态 (类别0)
    if exclude_null:
        print(f"过滤前总样本数: {len(df)}")
        df = df[df['Activity'] != 0]
        print(f"过滤后总样本数: {len(df)} (排除类别0: Null/Transition)")
        
        # 重新映射标签: 1-12 -> 0-11
        df['Activity'] = df['Activity'] - 1
    
    # 传感器列名 (12个通道: 胸部6个 + 右手腕6个)
    # 注意: 原始数据集还包含左脚踝传感器，但此CSV文件只包含胸部和右手腕数据
    sensor_cols = ['alx', 'aly', 'alz', 'glx', 'gly', 'glz',  # 胸部传感器
                   'arx', 'ary', 'arz', 'grx', 'gry', 'grz']  # 右手腕传感器
    
    # 如果没有指定测试受试者，默认使用最后两个受试者
    if test_subjects is None:
        test_subjects = ['subject9', 'subject10']
    
    # 分离训练和测试数据
    train_df = df[~df['subject'].isin(test_subjects)]
    test_df = df[df['subject'].isin(test_subjects)]
    
    print(f"MHEALTH数据集统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  训练数据: {len(train_df)} 样本")
    print(f"  测试数据: {len(test_df)} 样本")
    print(f"  训练受试者: {sorted(train_df['subject'].unique())}")
    print(f"  测试受试者: {sorted(test_df['subject'].unique())}")
    
    # 显示活动分布
    print(f"  活动类别分布:")
    for activity_id in sorted(df['Activity'].unique()):
        # 根据是否过滤调整标签显示
        if exclude_null:
            # 过滤后标签为0-11，对应原始1-12
            original_id = activity_id + 1
            if original_id in MHEALTH_ACTIVITY_LABELS:
                activity_name = MHEALTH_ACTIVITY_LABELS[original_id]
                count = len(df[df['Activity'] == activity_id])
                print(f"    {activity_id}: {activity_name} - {count} 样本")
        else:
            # 未过滤时直接使用原始标签
            if activity_id in MHEALTH_ACTIVITY_LABELS:
                activity_name = MHEALTH_ACTIVITY_LABELS[activity_id]
                count = len(df[df['Activity'] == activity_id])
                print(f"    {activity_id}: {activity_name} - {count} 样本")
    
    # 处理训练数据
    train_X, train_y = process_mhealth_split(train_df, sensor_cols, window_size, step_size)
    
    # 处理测试数据
    test_X, test_y = process_mhealth_split(test_df, sensor_cols, window_size, step_size)
    
    return train_X, train_y, test_X, test_y

def process_mhealth_split(df, sensor_cols, window_size=128, step=64):
    """
    处理mHealth数据的一个分割（训练或测试）
    
    采用滑窗方法将连续的传感器数据切分为固定长度的窗口。
    每个窗口包含window_size个时间步的12通道传感器数据。
    
    Args:
        df: 数据DataFrame
        sensor_cols: 传感器列名列表
        window_size: 窗口大小 (时间步数)
        step: 滑窗步长
        
    Returns:
        X: 窗口化的传感器数据 (N, window_size, 12)
        y: 对应的活动标签 (N,)
    """
    all_X, all_y = [], []
    
    # 按受试者和活动分组处理
    for (subject, activity), group in df.groupby(['subject', 'Activity']):
        if len(group) < window_size:
            continue  # 跳过太短的序列
            
        # 提取传感器数据
        sensor_data = group[sensor_cols].values  # (seq_len, 12)
        
        # 滑窗切分
        for start in range(0, len(sensor_data) - window_size + 1, step):
            window_data = sensor_data[start:start + window_size]  # (window_size, 12)
            all_X.append(window_data)
            all_y.append(activity)
    
    if len(all_X) == 0:
        return np.array([]), np.array([])
    
    X = np.stack(all_X)  # (N, window_size, 12)
    y = np.array(all_y)
    
    # [重点] 对每个通道归一化（全体样本+时间维）
    # 这确保了不同传感器通道的数据在相同的尺度上
    for c in range(X.shape[-1]):
        mean = X[:, :, c].mean()
        std = X[:, :, c].std()
        X[:, :, c] = (X[:, :, c] - mean) / (std + 1e-8)
    
    return X, y

def create_train_val_loaders(data_path, batch_size=64, val_split=0.1, window_size=128, step_size=64, 
                           test_subjects=None, exclude_null=True, seed=42):
    """
    从mHealth数据创建训练和验证数据加载器，保持类别比例。
    使用分层采样确保验证集和训练集具有相同的类别分布。
    
    MHEALTH数据集特点:
    - 类别不平衡: 类别0(Null/Transition)样本数远多于其他活动类别
    - 需要分层采样来保证验证集包含所有活动类别的代表性样本
    - 窗口重叠可能导致数据泄露，但在实际应用中是合理的
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        val_split: 验证集比例
        window_size: 滑窗大小
        step_size: 滑窗步长
        test_subjects: 测试受试者列表
        exclude_null: 是否排除类别0
        seed: 随机种子
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 载入数据
    train_X, train_y, _, _ = load_mhealth_data(data_path, window_size, step_size, exclude_null, test_subjects)
    
    if len(train_X) == 0:
        raise ValueError("训练数据为空，请检查数据路径和参数设置")
    
    # 计算每个类别的样本数量
    unique_labels, counts = np.unique(train_y, return_counts=True)
    print(f"训练集窗口化后的类别分布:")
    for label, count in zip(unique_labels, counts):
        activity_name = MHEALTH_ACTIVITY_LABELS.get(label, f"Unknown_{label}")
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    # 检查类别不平衡情况
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    print(f"  类别不平衡比例: {imbalance_ratio:.1f}:1 (最多类别/最少类别)")
    
    # 为每个类别计算验证集大小
    val_indices = []
    train_indices = []
    
    np.random.seed(seed)
    
    for label in unique_labels:
        # 找到该类别的所有样本索引
        label_indices = np.where(train_y == label)[0]
        
        # 计算该类别的验证集大小
        label_val_size = int(len(label_indices) * val_split)
        
        # 随机打乱该类别的索引
        np.random.shuffle(label_indices)
        
        # 分配验证集和训练集索引
        val_indices.extend(label_indices[:label_val_size])
        train_indices.extend(label_indices[label_val_size:])
    
    # 创建数据集
    full_dataset = MHealthWindowedDataset(train_X, train_y)
    
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
    val_labels = train_y[val_indices]
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    print(f"验证集类别分布:")
    for label, count in zip(val_unique, val_counts):
        activity_name = MHEALTH_ACTIVITY_LABELS.get(label, f"Unknown_{label}")
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    return train_loader, val_loader


def create_train_loader(data_file, batch_size=64, window_size=128, step_size=64, test_subjects=None, exclude_null=True, seed=42):
    """
    创建MHEALTH训练数据加载器
    
    Args:
        data_file: 数据文件路径
        batch_size: 批次大小
        window_size: 窗口大小
        step_size: 步长
        test_subjects: 测试受试者列表
        exclude_null: 是否排除空值类别
        seed: 随机种子
        
    Returns:
        训练数据加载器
    """
    train_loader, _ = create_train_val_loaders(
        data_path=data_file,
        batch_size=batch_size,
        val_split=0.1,
        window_size=window_size,
        step_size=step_size,
        test_subjects=test_subjects,
        exclude_null=exclude_null,
        seed=seed
    )
    return train_loader


def create_val_loader(data_file, batch_size=64, window_size=128, step_size=64, test_subjects=None, exclude_null=True, seed=42):
    """
    创建MHEALTH验证数据加载器
    
    Args:
        data_file: 数据文件路径
        batch_size: 批次大小
        window_size: 窗口大小
        step_size: 步长
        test_subjects: 测试受试者列表
        exclude_null: 是否排除空值类别
        seed: 随机种子
        
    Returns:
        验证数据加载器
    """
    _, val_loader = create_train_val_loaders(
        data_path=data_file,
        batch_size=batch_size,
        val_split=0.1,
        window_size=window_size,
        step_size=step_size,
        test_subjects=test_subjects,
        exclude_null=exclude_null,
        seed=seed
    )
    return val_loader


def create_test_loader(data_path, batch_size=64, window_size=128, step_size=64, test_subjects=None, exclude_null=True):
    """
    创建mHealth测试数据加载器
    """
    # 载入数据
    _, _, test_X, test_y = load_mhealth_data(data_path, window_size, step_size, exclude_null, test_subjects)
    
    if len(test_X) == 0:
        raise ValueError("测试数据为空，请检查数据路径和参数设置")
    
    test_ds = MHealthWindowedDataset(test_X, test_y)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 计算类别数量
    num_classes = len(np.unique(test_y))
    
    # 显示测试集类别分布
    unique_labels, counts = np.unique(test_y, return_counts=True)
    print(f"测试集类别分布:")
    for label, count in zip(unique_labels, counts):
        activity_name = MHEALTH_ACTIVITY_LABELS.get(label, f"Unknown_{label}")
        print(f"  类别 {label} ({activity_name}): {count} 样本")
    
    return test_loader, num_classes

if __name__ == "__main__":
    """
    MHEALTH数据集测试脚本
    
    数据集概述:
    - 来源: UCI Machine Learning Repository
    - 受试者: 10名志愿者 (不同年龄和体型)
    - 活动: 12种物理活动 (静态、动态、特定动作)
    - 传感器: 3个Shimmer2设备 (胸部、右手腕、左脚踝)
    - 采样率: 50Hz
    - 数据类型: 加速度计、陀螺仪、磁力计、ECG (本实现仅使用加速度计和陀螺仪)
    
    传感器配置:
    - 胸部传感器: 测量躯干运动，对步行、跑步等全身活动敏感
    - 右手腕传感器: 测量手臂运动，对举臂、弯腰等上肢活动敏感  
    - 左脚踝传感器: 测量腿部运动，对爬楼梯、跳跃等下肢活动敏感 (本CSV未包含)
    
    窗口化策略:
    - 窗口大小: 128个样本点 (2.56秒 @ 50Hz)
    - 重叠率: 50% (步长64个样本点)
    - 这种设置平衡了时间分辨率和计算效率
    """
    data_path = "/home/admin407/code/zyshe/WPDN/dataset/mhealth_raw_data.csv"

    # 创建数据加载器 (排除类别0)
    train_loader, val_loader = create_train_val_loaders(data_path, batch_size=32, val_split=0.1, exclude_null=True)
    test_loader, num_classes = create_test_loader(data_path, batch_size=32, exclude_null=True)

    print(f"\n数据加载器统计:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")
    print(f"  Test  batches: {len(test_loader)}")
    print(f"  Number of classes: {num_classes}")

    # 获取训练集前2个样本
    train_iter = iter(train_loader)
    X_train_batch, y_train_batch = next(train_iter)
    for i in range(2):
        print(f"\n[Train Sample {i}]")
        print(f"  X shape: {X_train_batch[i].shape}")  # (C, T) = (12, 128)
        # 根据是否过滤调整标签显示
        if True:  # exclude_null=True
            # 过滤后标签为0-11，对应原始1-12
            original_label = y_train_batch[i].item() + 1
            activity_name = MHEALTH_ACTIVITY_LABELS.get(original_label, "Unknown")
        else:
            activity_name = MHEALTH_ACTIVITY_LABELS.get(y_train_batch[i].item(), "Unknown")
        print(f"  y label: {y_train_batch[i].item()} ({activity_name})")
        print(f"  传感器数据 (每个通道前5个值):")
        
        # 显示胸部传感器数据
        print(f"    胸部加速度计 (alx,aly,alz):")
        for c in range(3):
            print(f"      Channel {c}: {X_train_batch[i][c][:5].tolist()}")
        print(f"    胸部陀螺仪 (glx,gly,glz):")
        for c in range(3, 6):
            print(f"      Channel {c}: {X_train_batch[i][c][:5].tolist()}")
            
        # 显示右手腕传感器数据
        print(f"    右手腕加速度计 (arx,ary,arz):")
        for c in range(6, 9):
            print(f"      Channel {c}: {X_train_batch[i][c][:5].tolist()}")
        print(f"    右手腕陀螺仪 (grx,gry,grz):")
        for c in range(9, 12):
            print(f"      Channel {c}: {X_train_batch[i][c][:5].tolist()}")

    # 获取验证集前1个样本
    val_iter = iter(val_loader)
    X_val_batch, y_val_batch = next(val_iter)
    print(f"\n[Val Sample 0]")
    print(f"  X shape: {X_val_batch[0].shape}")  # (C, T)
    # 根据是否过滤调整标签显示
    if True:  # exclude_null=True
        original_label = y_val_batch[0].item() + 1
        activity_name = MHEALTH_ACTIVITY_LABELS.get(original_label, "Unknown")
    else:
        activity_name = MHEALTH_ACTIVITY_LABELS.get(y_val_batch[0].item(), "Unknown")
    print(f"  y label: {y_val_batch[0].item()} ({activity_name})")

    # 获取测试集前1个样本
    test_iter = iter(test_loader)
    X_test_batch, y_test_batch = next(test_iter)
    print(f"\n[Test Sample 0]")
    print(f"  X shape: {X_test_batch[0].shape}")  # (C, T)
    # 根据是否过滤调整标签显示
    if True:  # exclude_null=True
        original_label = y_test_batch[0].item() + 1
        activity_name = MHEALTH_ACTIVITY_LABELS.get(original_label, "Unknown")
    else:
        activity_name = MHEALTH_ACTIVITY_LABELS.get(y_test_batch[0].item(), "Unknown")
    print(f"  y label: {y_test_batch[0].item()} ({activity_name})")