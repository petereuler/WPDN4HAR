import torch
import time
import numpy as np
from model.model_wpdn import LightweightWaveletPacketCNN
from model.baselines.cnn_models import LightweightCNN
from model.traditional_wavelet_packet import TraditionalWaveletPacketCNN

def profile_model(model, input_data, num_runs=100, warmup_runs=10):
    """性能分析函数"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_data)
    
    # 实际测试
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_data)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times)
    }

def analyze_model_complexity():
    """分析不同模型的复杂度"""
    # 测试参数
    batch_size = 1
    in_channels = 6
    num_classes = 6
    input_length = 128
    
    # 创建测试数据
    x = torch.randn(batch_size, in_channels, input_length)
    
    print("=== 模型性能对比分析 ===\n")
    
    # 1. 标准CNN
    print("1. 标准CNN:")
    cnn_model = LightweightCNN(in_channels, num_classes, input_length, verbose=False)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    cnn_perf = profile_model(cnn_model, x)
    print(f"   参数量: {cnn_params:,}")
    print(f"   推理时间: {cnn_perf['mean_time_ms']:.3f} ± {cnn_perf['std_time_ms']:.3f} ms")
    
    # 2. 传统小波包CNN
    print("\n2. 传统小波包CNN:")
    wavelet_trad_model = TraditionalWaveletPacketCNN(
        in_channels=in_channels, 
        num_classes=num_classes, 
        input_length=input_length,
        classifier_type='ultra_lightweight'
    )
    wavelet_trad_params = sum(p.numel() for p in wavelet_trad_model.parameters())
    wavelet_trad_perf = profile_model(wavelet_trad_model, x)
    print(f"   参数量: {wavelet_trad_params:,}")
    print(f"   推理时间: {wavelet_trad_perf['mean_time_ms']:.3f} ± {wavelet_trad_perf['std_time_ms']:.3f} ms")
    
    # 3. 轻量化小波包CNN (单一分解器)
    print("\n3. 轻量化小波包CNN (单一分解器):")
    wavelet_lite_single = LightweightWaveletPacketCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_length=input_length,
        classifier_type="ultra_lightweight",
        use_parallel=False,
        decompose_levels=3,
        verbose=False
    )
    wavelet_lite_single_params = sum(p.numel() for p in wavelet_lite_single.parameters())
    wavelet_lite_single_perf = profile_model(wavelet_lite_single, x)
    print(f"   参数量: {wavelet_lite_single_params:,}")
    print(f"   推理时间: {wavelet_lite_single_perf['mean_time_ms']:.3f} ± {wavelet_lite_single_perf['std_time_ms']:.3f} ms")
    
    # 4. 轻量化小波包CNN (并行分解器)
    print("\n4. 轻量化小波包CNN (并行分解器):")
    wavelet_lite_parallel = LightweightWaveletPacketCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_length=input_length,
        classifier_type="ultra_lightweight",
        use_parallel=True,
        num_parallel_groups=4,
        decompose_levels=3,
        verbose=False
    )
    wavelet_lite_parallel_params = sum(p.numel() for p in wavelet_lite_parallel.parameters())
    wavelet_lite_parallel_perf = profile_model(wavelet_lite_parallel, x)
    print(f"   参数量: {wavelet_lite_parallel_params:,}")
    print(f"   推理时间: {wavelet_lite_parallel_perf['mean_time_ms']:.3f} ± {wavelet_lite_parallel_perf['std_time_ms']:.3f} ms")
    
    # 分析结果
    print("\n=== 分析结果 ===")
    print(f"参数量对比:")
    print(f"  CNN: {cnn_params:,}")
    print(f"  传统小波: {wavelet_trad_params:,}")
    print(f"  轻量小波(单): {wavelet_lite_single_params:,}")
    print(f"  轻量小波(并行): {wavelet_lite_parallel_params:,}")
    
    print(f"\n推理时间对比:")
    print(f"  CNN: {cnn_perf['mean_time_ms']:.3f} ms")
    print(f"  传统小波: {wavelet_trad_perf['mean_time_ms']:.3f} ms")
    print(f"  轻量小波(单): {wavelet_lite_single_perf['mean_time_ms']:.3f} ms")
    print(f"  轻量小波(并行): {wavelet_lite_parallel_perf['mean_time_ms']:.3f} ms")
    
    print(f"\n效率分析 (参数量/推理时间):")
    print(f"  CNN: {cnn_params/cnn_perf['mean_time_ms']:.0f} params/ms")
    print(f"  传统小波: {wavelet_trad_params/wavelet_trad_perf['mean_time_ms']:.0f} params/ms")
    print(f"  轻量小波(单): {wavelet_lite_single_params/wavelet_lite_single_perf['mean_time_ms']:.0f} params/ms")
    print(f"  轻量小波(并行): {wavelet_lite_parallel_params/wavelet_lite_parallel_perf['mean_time_ms']:.0f} params/ms")

if __name__ == "__main__":
    analyze_model_complexity()
