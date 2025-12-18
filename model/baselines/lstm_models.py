import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardLSTM(nn.Module):
    """
    标准LSTM分类器，用于时间序列分类
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 hidden_size=128, num_layers=2, dropout=0.3, verbose=False):
        super(StandardLSTM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 双向LSTM
        )
        
        # 更复杂的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),  # *2 因为双向
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"标准LSTM分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"隐藏层大小: {self.hidden_size}")
        print(f"LSTM层数: {self.num_layers}")
        print(f"双向LSTM: 是")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        # 转换为LSTM期望的格式: [batch_size, seq_len, in_channels]
        x = x.transpose(1, 2)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # 分类
        output = self.classifier(last_output)
        
        return output


class LightweightLSTM(nn.Module):
    """
    轻量级LSTM分类器，用于时间序列分类
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 hidden_size=64, num_layers=1, dropout=0.2, verbose=False):
        super(LightweightLSTM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"轻量级LSTM分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"隐藏层大小: {self.hidden_size}")
        print(f"LSTM层数: {self.num_layers}")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        # 转换为LSTM期望的格式: [batch_size, seq_len, in_channels]
        x = x.transpose(1, 2)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 分类
        output = self.classifier(last_output)
        
        return output


class StandardGRU(nn.Module):
    """
    标准GRU分类器
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 hidden_size=128, num_layers=2, dropout=0.3, verbose=False):
        super(StandardGRU, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GRU层
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 双向GRU
        )
        
        # 更复杂的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),  # *2 因为双向
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"标准GRU分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"隐藏层大小: {self.hidden_size}")
        print(f"GRU层数: {self.num_layers}")
        print(f"双向GRU: 是")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        # 转换为GRU期望的格式: [batch_size, seq_len, in_channels]
        x = x.transpose(1, 2)
        
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        
        # 使用最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # 分类
        output = self.classifier(last_output)
        
        return output


class LightweightGRU(nn.Module):
    """
    轻量级GRU分类器
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 hidden_size=64, num_layers=1, dropout=0.2, verbose=False):
        super(LightweightGRU, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GRU层
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"轻量级GRU分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"隐藏层大小: {self.hidden_size}")
        print(f"GRU层数: {self.num_layers}")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        # 转换为GRU期望的格式: [batch_size, seq_len, in_channels]
        x = x.transpose(1, 2)
        
        # GRU前向传播
        gru_out, h_n = self.gru(x)
        
        # 使用最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 分类
        output = self.classifier(last_output)
        
        return output


if __name__ == "__main__":
    print("=== 轻量级RNN模型测试 ===")
    
    # 测试参数
    batch_size, in_channels, seq_len = 4, 6, 128
    num_classes = 6
    x = torch.randn(batch_size, in_channels, seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 测试轻量级LSTM
    print("\n1. 测试轻量级LSTM分类器:")
    lstm_model = LightweightLSTM(in_channels, num_classes, seq_len, verbose=True)
    lstm_out = lstm_model(x)
    print(f"LSTM输出形状: {lstm_out.shape}")
    
    # 测试轻量级GRU
    print("\n2. 测试轻量级GRU分类器:")
    gru_model = LightweightGRU(in_channels, num_classes, seq_len, verbose=True)
    gru_out = gru_model(x)
    print(f"GRU输出形状: {gru_out.shape}")
    
    print("\n=== 轻量级RNN模型测试完成 ===")