import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .wavelet_transform import TraditionalWaveletTransform

def generate_highpass_from_lowpass(lowpass_filter):
    """
    Generate high-pass filter from low-pass filter (QMF constraint)
    h_high[n] = (-1)^n * h_low[N-1-n]
    """
    N = lowpass_filter.shape[-1]
    sign = torch.tensor([(-1) ** n for n in range(N)], dtype=torch.float32, device=lowpass_filter.device)
    return lowpass_filter.flip(-1) * sign.view(1, 1, -1)

def even_shift_orthogonality_loss(h_low, h_high):
    """
    Calculate even-shift orthogonality constraint loss
    Ensure filters satisfy even-shift orthogonality property
    
    Corrected version: Properly handle autocorrelation of each filter, avoid cross-correlation interference
    """
    N = h_low.shape[-1]
    C = h_low.shape[0]  # Number of channels
    
    # Reshape filter dimensions: (C, 1, K) -> (1, C, K)
    h_low_reshaped = h_low.permute(1, 0, 2)  # (1, C, K)
    h_high_reshaped = h_high.permute(1, 0, 2)  # (1, C, K)
    
    # Calculate autocorrelation of each filter, use groups=C to avoid cross-correlation
    h_low_autocorr = F.conv1d(h_low_reshaped, h_low.flip(-1), padding=N-1, groups=C)  # (1, C, 2K-1)
    h_high_autocorr = F.conv1d(h_high_reshaped, h_high.flip(-1), padding=N-1, groups=C)  # (1, C, 2K-1)
    
    # Target vectors: 1 only at center position, 0 elsewhere
    target_low = torch.zeros_like(h_low_autocorr)
    target_low[:, :, N-1] = 1.0  # Set center position to 1
    
    target_high = torch.zeros_like(h_high_autocorr)
    target_high[:, :, N-1] = 1.0  # Set center position to 1
    
    # Calculate even-shift orthogonality loss for low-pass and high-pass filters
    loss_low = F.mse_loss(h_low_autocorr, target_low)
    loss_high = F.mse_loss(h_high_autocorr, target_high)
    
    return loss_low + loss_high

class WaveletPacketDecomposeBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=4, level=1, shared_lowpass_filter=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.level = level
        
        # 使用共享的滤波器或创建独立滤波器
        # 优化：使用单一滤波器处理所有通道，而不是每个通道一个滤波器
        if shared_lowpass_filter is not None:
            self.lowpass_filter = shared_lowpass_filter
            self.use_shared_filter = True
        else:
            # 优化：从 (in_channels, 1, kernel_size) 改为 (1, in_channels, kernel_size)
            # 这样一个滤波器就能处理所有输入通道
            self.lowpass_filter = nn.Parameter(torch.randn(1, in_channels, kernel_size))
            self.use_shared_filter = False
        
        # 激活函数
        self.act_low = nn.ReLU()
        self.act_high = nn.ReLU()
    
    def forward(self, x):
        # x: [B, C, T]
        
        # 从低通滤波器生成高通滤波器（QMF约束）
        highpass_filter = generate_highpass_from_lowpass(self.lowpass_filter)
        
        # 修复：直接使用滤波器进行分组卷积，保持通道数
        # 使用 groups=x.shape[1] 确保每个输入通道使用对应的滤波器
        low = F.conv1d(x, self.lowpass_filter, stride=2, padding=self.padding, groups=x.shape[1])
        high = F.conv1d(x, highpass_filter, stride=2, padding=self.padding, groups=x.shape[1])
        
        # 频带归一化和激活
        low = self.act_low(low)
        high = self.act_high(high)
        
        return low, high
    
    def get_orthogonality_loss(self):
        """计算偶移正交约束损失"""
        # 只有非共享滤波器才计算损失，避免重复计算
        if not self.use_shared_filter:
            highpass_filter = generate_highpass_from_lowpass(self.lowpass_filter)
            return even_shift_orthogonality_loss(self.lowpass_filter, highpass_filter)
        else:
            return torch.tensor(0.0, device=self.lowpass_filter.device)

class WaveletPacketDecomposer(nn.Module):
    def __init__(self, in_channels, kernel_size=4, decompose_levels=3, verbose=True):
        super().__init__()
        self.in_channels = in_channels
        self.verbose = verbose
        self.decompose_levels = decompose_levels
        
        # 修复：使用正确的滤波器形状 (in_channels, 1, kernel_size) 来保持通道数
        # 这样每个通道都有自己的滤波器，保持输出的通道数与输入一致
        self.shared_lowpass_filter = nn.Parameter(torch.randn(in_channels, 1, kernel_size))
        
        # 优化：不再需要为每个节点创建独立的分解器
        # 因为所有节点都使用同一个滤波器，只需要一个分解块即可
        self.decompose_block = WaveletPacketDecomposeBlock(
            in_channels, kernel_size, level=1, 
            shared_lowpass_filter=self.shared_lowpass_filter
        )
        
        if verbose:
            print(f"优化后的小波包分解器:")
            print(f"  - 输入通道数: {in_channels}")
            print(f"  - 分解级数: {decompose_levels}")
            print(f"  - 滤波器参数量: {kernel_size * in_channels} (原来: {kernel_size * in_channels * (2**decompose_levels - 1)})")
            print(f"  - 参数量减少: {((kernel_size * in_channels * (2**decompose_levels - 1) - kernel_size * in_channels) / (kernel_size * in_channels * (2**decompose_levels - 1)) * 100):.1f}%")
    
    def forward(self, x):
        # x: [B, C, T]
        
        # 优化：使用单一分解块进行逐级分解
        current_level_outputs = [x]  # 初始输入
        
        for level in range(1, self.decompose_levels + 1):
            next_level_outputs = []
            
            # 对当前级别的每个输出进行分解
            # 所有分解都使用同一个滤波器
            for signal in current_level_outputs:
                low, high = self.decompose_block(signal)
                next_level_outputs.extend([low, high])
            
            current_level_outputs = next_level_outputs
        
        # 最后一级的输出就是所有频带
        frequency_bands = current_level_outputs
        
        return frequency_bands
    
    def get_total_orthogonality_loss(self):
        """计算共享滤波器的偶移正交约束损失"""
        # 由于所有分解块共享同一组滤波器，只需计算一次损失
        highpass_filter = generate_highpass_from_lowpass(self.shared_lowpass_filter)
        return even_shift_orthogonality_loss(self.shared_lowpass_filter, highpass_filter)

class ParallelDecomposer(nn.Module):
    def __init__(self, in_channels, kernel_size=4, num_parallel_groups=4, decompose_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_parallel_groups = num_parallel_groups
        self.decompose_levels = decompose_levels
        
        # 创建多组独立的小波包分解器，每组使用不同的滤波器
        self.parallel_decomposers = nn.ModuleList()
        for i in range(num_parallel_groups):
            # 每组创建一个独立的小波包分解器
            decomposer = WaveletPacketDecomposer(
                in_channels=in_channels,
                kernel_size=kernel_size,
                decompose_levels=decompose_levels,
                verbose=False  # 避免重复打印
            )
            self.parallel_decomposers.append(decomposer)
    
    def forward(self, x):
        # x: [B, C, T]
        parallel_outputs = []
        
        # 每个并行分解器进行完整的逐级分解
        for i, decomposer in enumerate(self.parallel_decomposers):
            frequency_bands = decomposer(x)  # 进行完整的逐级分解
            parallel_outputs.append(frequency_bands)
        
        return parallel_outputs
    
    def get_orthogonality_loss(self):
        """计算所有并行分解器的正交约束损失"""
        total_loss = 0.0
        for decomposer in self.parallel_decomposers:
            loss = decomposer.get_total_orthogonality_loss()
            total_loss += loss
        return total_loss / len(self.parallel_decomposers)

class TimeFrequencyMapGenerator(nn.Module):
    def __init__(self, in_channels, input_length=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_length = input_length
        
        # 计算第三级分解后的信号长度
        self.level3_length = input_length // 8  # 经过3次下采样
        
    def forward(self, frequency_bands):
        # frequency_bands: 8个频带，每个形状为 [B, C, T_level3]
        # 其中 T_level3 = input_length // 8
        
        batch_size = frequency_bands[0].shape[0]
        
        # 将所有频带堆叠成时频图
        # 形状: [B, 8, C, T_level3] -> [B, C, 8, T_level3]
        stacked_bands = torch.stack(frequency_bands, dim=1)  # [B, 8, C, T_level3]
        time_freq_map = stacked_bands.permute(0, 2, 1, 3)   # [B, C, 8, T_level3]
        
        return time_freq_map

class MultiTimeFrequencyMapGenerator(nn.Module):
    def __init__(self, in_channels, input_length=128, num_parallel_groups=4):
        super().__init__()
        self.in_channels = in_channels
        self.input_length = input_length
        self.num_parallel_groups = num_parallel_groups
        
        # 时频图生成器
        self.time_freq_generator = TimeFrequencyMapGenerator(in_channels, input_length)
        
    def forward(self, parallel_outputs):
        # parallel_outputs: 4组频带列表，每组包含完整的逐级分解结果
        time_freq_maps = []
        
        for i, frequency_bands in enumerate(parallel_outputs):
            # 每组已经是完整的频带列表，直接生成时频图
            time_freq_map = self.time_freq_generator(frequency_bands)
            time_freq_maps.append(time_freq_map)
        
        # 将多张时频图沿通道维度拼接
        # 每张时频图: [B, C, freq_bins, time_bins]
        # 拼接后: [B, num_parallel_groups*C, freq_bins, time_bins]
        combined_time_freq_map = torch.cat(time_freq_maps, dim=1)
        
        return combined_time_freq_map, time_freq_maps
    
    def get_orthogonality_loss(self):
        """MultiTimeFrequencyMapGenerator本身不包含分解器，返回0"""
        return torch.tensor(0.0)



class LightweightWaveletPacketCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=6, input_length=128, kernel_size=4,
                 use_parallel=False, num_parallel_groups=4,
                 use_traditional_wavelet=False, wavelet_type='db4', wavelet_levels=3,
                 decompose_levels=3, verbose=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.use_parallel = use_parallel
        self.num_parallel_groups = num_parallel_groups
        self.use_traditional_wavelet = use_traditional_wavelet
        self.wavelet_type = wavelet_type
        self.wavelet_levels = wavelet_levels
        self.decompose_levels = decompose_levels
        
        if use_traditional_wavelet:
            # 使用传统小波变换
            self.traditional_wavelet = TraditionalWaveletTransform(
                wavelet=wavelet_type, 
                levels=wavelet_levels
            )
            # 计算传统小波的输出形状
            freq_bins, time_bins = self.traditional_wavelet.get_output_shape(input_length)
            time_freq_shape = (freq_bins, time_bins)
        elif use_parallel:
            # 使用并行分解结构
            self.parallel_decomposer = ParallelDecomposer(in_channels, kernel_size, num_parallel_groups, decompose_levels)
            self.multi_time_freq_generator = MultiTimeFrequencyMapGenerator(in_channels, input_length, num_parallel_groups)
            # 计算时频图的形状 - 根据分解级数动态计算
            num_frequency_bands = 2 ** decompose_levels  # 分解级数为N时，有2^N个频带
            time_freq_shape = (num_frequency_bands, input_length // (2 ** decompose_levels))
        else:
            # 使用单一分解器
            self.decomposer = WaveletPacketDecomposer(in_channels, kernel_size, decompose_levels, verbose)
            self.time_freq_generator = TimeFrequencyMapGenerator(in_channels, input_length)
            # 计算时频图的形状 - 根据分解级数动态计算
            num_frequency_bands = 2 ** decompose_levels  # 分解级数为N时，有2^N个频带
            time_freq_shape = (num_frequency_bands, input_length // (2 ** decompose_levels))
        
        # 移除轻量级Conv2D分类器，使用标准Conv2D分类器
        self.classifier = self._create_standard_classifier(in_channels, num_classes, time_freq_shape, use_parallel, num_parallel_groups)

    def _create_standard_classifier(self, in_channels, num_classes, time_freq_shape, use_parallel, num_parallel_groups):
        """创建标准Conv2D分类器"""
        actual_in_channels = in_channels * num_parallel_groups if use_parallel else in_channels

        return nn.Sequential(
            nn.Conv2d(actual_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        if verbose:
            if use_traditional_wavelet:
                print("使用传统小波变换")
            else:
                print("使用学习型小波包分解")
            print(f"使用并行分解: {use_parallel}")
            
            # 计算模型参数数量
            total_params = sum(p.numel() for p in self.parameters())
            print(f"模型参数数量: {total_params:,}")
    
    def forward(self, x):
        # x: [B, C, T]
        
        if self.use_traditional_wavelet:
            # 使用传统小波变换
            time_freq_map = self.traditional_wavelet(x)
        elif self.use_parallel:
            # 使用并行分解结构
            parallel_outputs = self.parallel_decomposer(x)
            time_freq_map, _ = self.multi_time_freq_generator(parallel_outputs)
        else:
            # 使用单一分解器
            frequency_bands = self.decomposer(x)
            time_freq_map = self.time_freq_generator(frequency_bands)
        
        # 分类
        output = self.classifier(time_freq_map)
        
        return output
    
    def get_orthogonality_loss(self):
        """计算正交约束损失"""
        if self.use_traditional_wavelet:
            # 传统小波不需要正交约束损失
            return torch.tensor(0.0)
        elif self.use_parallel:
            # 并行分解器的正交约束损失
            return self.parallel_decomposer.get_orthogonality_loss()
        else:
            # 单一分解器的正交约束损失
            return self.decomposer.get_total_orthogonality_loss()

# ----------------------------
# 测试代码
# ----------------------------
if __name__ == "__main__":
    # 创建测试数据
    x = torch.randn(4, 6, 128)

    print("=== 原始轻量化版本 ===")
    model_light = LightweightWaveletPacketCNN(
        in_channels=6, 
        input_length=128, 
        num_classes=6,
        classifier_type="lightweight"
    )

    print("输入形状:", x.shape)
    out_light = model_light(x)
    print("输出形状:", out_light.shape)

    # 计算参数量
    total_params_light = sum(p.numel() for p in model_light.parameters())
    print(f"总参数量: {total_params_light:,}")

    # 分别计算各部分参数量
    if hasattr(model_light, 'decomposer'):
        wavelet_params = sum(p.numel() for p in model_light.decomposer.parameters())
        print(f"小波分解器参数量: {wavelet_params:,}")
    classifier_params_light = sum(p.numel() for p in model_light.classifier.parameters())
    print(f"分类器参数量: {classifier_params_light:,}")

    # 正交约束损失
    orth_loss = model_light.get_orthogonality_loss()
    print(f"正交约束损失: {orth_loss.item():.6f}")

    print("\n=== 超轻量化版本 ===")
    model_ultra = LightweightWaveletPacketCNN(
        in_channels=6,
        input_length=128,
        num_classes=6
    )

    print("输入形状:", x.shape)
    out_ultra = model_ultra(x)
    print("输出形状:", out_ultra.shape)

    # 计算参数量
    total_params_ultra = sum(p.numel() for p in model_ultra.parameters())
    print(f"总参数量: {total_params_ultra:,}")

    # 分别计算各部分参数量
    if hasattr(model_ultra, 'decomposer'):
        wavelet_params = sum(p.numel() for p in model_ultra.decomposer.parameters())
        print(f"小波分解器参数量: {wavelet_params:,}")
    classifier_params_ultra = sum(p.numel() for p in model_ultra.classifier.parameters())
    print(f"分类器参数量: {classifier_params_ultra:,}")

    # 正交约束损失
    orth_loss = model_ultra.get_orthogonality_loss()
    print(f"正交约束损失: {orth_loss.item():.6f}")

    print("\n=== 对比结果 ===")
    print(f"参数量减少: {total_params_light - total_params_ultra:,} ({((total_params_light - total_params_ultra) / total_params_light * 100):.1f}%)")
    print(f"分类器参数量减少: {classifier_params_light - classifier_params_ultra:,} ({((classifier_params_light - classifier_params_ultra) / classifier_params_light * 100):.1f}%)")
    print(f"输出形状一致: {out_light.shape == out_ultra.shape}")