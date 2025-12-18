import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class StandardTransformer(nn.Module):
    """
    标准Transformer分类器
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 d_model=128, nhead=8, num_layers=4, dim_feedforward=512, 
                 dropout=0.1, verbose=False):
        super(StandardTransformer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        # 输入投影层
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 更简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"标准Transformer分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"模型维度: {self.d_model}")
        print(f"注意力头数: {self.nhead}")
        print(f"编码器层数: {self.num_layers}")
        print(f"前馈网络维度: {self.dim_feedforward}")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        # 转换为Transformer期望的格式: [batch_size, seq_len, in_channels]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 全局平均池化
        pooled_output = torch.mean(transformer_out, dim=1)  # [batch_size, d_model]
        
        # 分类
        output = self.classifier(pooled_output)
        
        return output


class LightweightTransformer(nn.Module):
    """
    轻量级Transformer分类器
    """
    def __init__(self, in_channels, num_classes, input_length, 
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 dropout=0.1, verbose=False):
        super(LightweightTransformer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        # 输入投影层
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        if verbose:
            self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"轻量级Transformer分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"模型维度: {self.d_model}")
        print(f"注意力头数: {self.nhead}")
        print(f"编码器层数: {self.num_layers}")
        print(f"前馈网络维度: {self.dim_feedforward}")
        print(f"Dropout率: {self.dropout}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")
    
    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        # 转换为Transformer期望的格式: [batch_size, seq_len, in_channels]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 全局平均池化
        pooled_output = torch.mean(transformer_out, dim=1)  # [batch_size, d_model]
        
        # 分类
        output = self.classifier(pooled_output)
        
        return output





if __name__ == "__main__":
    print("=== 轻量级Transformer模型测试 ===")
    
    # 测试参数
    batch_size, in_channels, seq_len = 4, 6, 128
    num_classes = 6
    x = torch.randn(batch_size, in_channels, seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 测试轻量级Transformer
    print("\n1. 测试轻量级Transformer分类器:")
    transformer_model = LightweightTransformer(in_channels, num_classes, seq_len, verbose=True)
    transformer_out = transformer_model(x)
    print(f"轻量级Transformer输出形状: {transformer_out.shape}")
    
    print("\n=== 轻量级Transformer模型测试完成 ===")