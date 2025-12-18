import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardCNN(nn.Module):
    """
    标准CNN分类器，用于时间序列分类
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 base_channels=48, num_blocks=3, dropout=0.25, verbose=False):
        super(StandardCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout = dropout
        
        # 构建卷积块 - 每个块有两个卷积层，但通道数适中
        layers = []
        current_channels = in_channels
        current_length = input_length
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            current_channels = out_channels
            current_length = current_length // 2
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 中等复杂度的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_channels, current_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(current_channels // 2, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"标准CNN分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"基础通道数: {self.base_channels}")
        print(f"卷积块数量: {self.num_blocks}")
        print(f"每块卷积层数: 2")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # 全局平均池化
        pooled_features = self.global_avg_pool(features)  # [batch_size, channels, 1]
        pooled_features = pooled_features.squeeze(-1)     # [batch_size, channels]
        
        # 分类
        output = self.classifier(pooled_features)
        
        return output


class LightweightCNN(nn.Module):
    """
    轻量化CNN模型 - 增强版（2万级参数）
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 base_channels=48, num_blocks=2, dropout=0.2, verbose=False):
        super(LightweightCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout = dropout
        
        # 构建卷积块
        layers = []
        current_channels = in_channels
        current_length = input_length
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            current_channels = out_channels
            current_length = current_length // 2
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 复杂分类器（两层）
        hidden_size = current_channels // 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_channels, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"轻量化CNN模型 - 增强版模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"基础通道数: {self.base_channels}")
        print(f"卷积块数量: {self.num_blocks}")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # 全局平均池化
        pooled_features = self.global_avg_pool(features)  # [batch_size, channels, 1]
        pooled_features = pooled_features.squeeze(-1)     # [batch_size, channels]
        
        # 分类
        output = self.classifier(pooled_features)
        
        return output








if __name__ == "__main__":
    print("=== 轻量级CNN模型测试 ===")
    
    # 测试参数
    batch_size, in_channels, seq_len = 4, 6, 128
    num_classes = 6
    x = torch.randn(batch_size, in_channels, seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 测试轻量级CNN
    print("\n1. 测试轻量级CNN分类器:")
    cnn_model = LightweightCNN(in_channels, num_classes, seq_len, verbose=True)
    cnn_out = cnn_model(x)
    print(f"轻量级CNN输出形状: {cnn_out.shape}")
    
    print("\n=== 轻量级CNN模型测试完成 ===")