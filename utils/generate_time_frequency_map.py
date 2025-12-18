#!/usr/bin/env python3
"""
WPDN模型时频图生成器
生成标准的时频图，确保每个像素点都是方方正正的
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from model.model_wpdn import LightweightWaveletPacketCNN
import os

# 设置matplotlib参数确保方正像素
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

class TimeFrequencyVisualizer:
    def __init__(self, model_path=None, device='cpu'):
        """
        初始化时频图可视化器
        
        Args:
            model_path: 预训练模型路径（可选）
            device: 计算设备
        """
        self.device = device
        
        # 创建WPDN模型
        self.model = LightweightWaveletPacketCNN(
            in_channels=6,
            num_classes=6,
            input_length=128,
            kernel_size=4,
            classifier_type="ultra_lightweight",
            use_parallel=False,
            decompose_levels=3,
            verbose=False
        ).to(device)
        
        # 加载预训练权重（如果提供）
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"已加载预训练模型: {model_path}")
        
        self.model.eval()
        
        # 频带标签
        self.frequency_bands = [
            'Band 0 (LL)', 'Band 1 (LH)', 'Band 2 (HL)', 'Band 3 (HH)',
            'Band 4 (LLL)', 'Band 5 (LLH)', 'Band 6 (LHL)', 'Band 7 (LHH)'
        ]
        
        # 创建自定义颜色映射
        self.create_custom_colormap()
    
    def create_custom_colormap(self):
        """创建适合时频图的自定义颜色映射"""
        # 蓝-白-红颜色映射
        colors = ['#000080', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
        self.cmap = LinearSegmentedColormap.from_list('time_freq', colors, N=256)
        
        # 热力图颜色映射
        self.cmap_hot = plt.cm.hot
        
        # 灰度颜色映射
        self.cmap_gray = plt.cm.gray
    
    def extract_time_frequency_features(self, x, normalize=True):
        """
        提取时频特征
        
        Args:
            x: 输入数据 [B, C, T]
            normalize: 是否对时频图进行归一化处理
            
        Returns:
            frequency_bands: 8个频带的特征
            time_freq_map: 时频图 [B, C, freq_bins, time_bins]
        """
        with torch.no_grad():
            # 通过小波包分解器获取频带
            if hasattr(self.model, 'decomposer'):
                frequency_bands = self.model.decomposer(x)
            else:
                # 如果没有分解器，创建一个临时的
                from model.model_wpdn import WaveletPacketDecomposer
                decomposer = WaveletPacketDecomposer(
                    in_channels=x.shape[1],
                    kernel_size=4,
                    decompose_levels=3,
                    verbose=False
                ).to(self.device)
                frequency_bands = decomposer(x)
            
            # 生成时频图
            if hasattr(self.model, 'time_freq_generator'):
                time_freq_map = self.model.time_freq_generator(frequency_bands)
            else:
                # 手动生成时频图
                batch_size = frequency_bands[0].shape[0]
                stacked_bands = torch.stack(frequency_bands, dim=1)  # [B, 8, C, T]
                time_freq_map = stacked_bands.permute(0, 2, 1, 3)   # [B, C, 8, T]
            
            # 归一化处理
            if normalize:
                # 对每个样本和通道分别进行归一化
                for b in range(time_freq_map.shape[0]):
                    for c in range(time_freq_map.shape[1]):
                        data = time_freq_map[b, c]
                        data_min = data.min()
                        data_max = data.max()
                        if data_max > data_min:  # 避免除零
                            time_freq_map[b, c] = (data - data_min) / (data_max - data_min)
                        else:
                            time_freq_map[b, c] = torch.zeros_like(data)
        
        return frequency_bands, time_freq_map
    
    def plot_single_channel_time_frequency(self, time_freq_map, channel_idx=0, 
                                         sample_idx=0, title_prefix="", 
                                         colormap='custom', save_path=None):
        """
        绘制单通道时频图，确保像素方正
        
        Args:
            time_freq_map: 时频图数据 [B, C, freq_bins, time_bins]
            channel_idx: 通道索引
            sample_idx: 样本索引
            title_prefix: 标题前缀
            colormap: 颜色映射 ('custom', 'hot', 'gray')
            save_path: 保存路径
        """
        # 选择颜色映射
        if colormap == 'custom':
            cmap = self.cmap
        elif colormap == 'hot':
            cmap = self.cmap_hot
        else:
            cmap = self.cmap_gray
        
        # 提取单通道数据
        data = time_freq_map[sample_idx, channel_idx].cpu().numpy()  # [freq_bins, time_bins]
        freq_bins, time_bins = data.shape
        
        # 创建图形，确保方正像素
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 使用imshow绘制，确保像素方正
        im = ax.imshow(data, 
                      cmap=cmap,
                      aspect='equal',  # 确保像素方正
                      origin='lower',  # 频率从下到上
                      interpolation='nearest',  # 保持像素边界清晰
                      extent=[0, time_bins, 0, freq_bins])
        
        # 设置标题和标签
        ax.set_title(f'{title_prefix}Time-Frequency Map - Channel {channel_idx}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (Sample Points)', fontsize=14)
        ax.set_ylabel('Frequency Bands (Wavelet Packet Decomposition)', fontsize=14)
        
        # 设置频率带标签
        ax.set_yticks(np.arange(freq_bins) + 0.5)
        ax.set_yticklabels(self.frequency_bands, fontsize=10)
        
        # 设置时间轴标签
        time_ticks = np.linspace(0, time_bins, min(6, time_bins + 1))
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f'{int(t)}' for t in time_ticks])
        
        # 添加网格线，突出像素边界
        ax.set_xticks(np.arange(time_bins + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(freq_bins + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Amplitude', fontsize=12)
        
        # 添加数值标注（如果像素不太多）
        if freq_bins <= 8 and time_bins <= 20:
            for i in range(freq_bins):
                for j in range(time_bins):
                    text_color = 'white' if data[i, j] < (data.max() + data.min()) / 2 else 'black'
                    ax.text(j + 0.5, i + 0.5, f'{data[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=8)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"时频图已保存到: {save_path}")
        
        return fig, ax
    
    def plot_multi_channel_time_frequency(self, time_freq_map, sample_idx=0, 
                                        title_prefix="", colormap='custom', 
                                        save_path=None):
        """
        绘制多通道时频图对比
        
        Args:
            time_freq_map: 时频图数据 [B, C, freq_bins, time_bins]
            sample_idx: 样本索引
            title_prefix: 标题前缀
            colormap: 颜色映射
            save_path: 保存路径
        """
        # 选择颜色映射
        if colormap == 'custom':
            cmap = self.cmap
        elif colormap == 'hot':
            cmap = self.cmap_hot
        else:
            cmap = self.cmap_gray
        
        batch_size, num_channels, freq_bins, time_bins = time_freq_map.shape
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 计算全局颜色范围
        global_min = time_freq_map[sample_idx].min().item()
        global_max = time_freq_map[sample_idx].max().item()
        
        for ch in range(min(num_channels, 6)):  # 最多显示6个通道
            data = time_freq_map[sample_idx, ch].cpu().numpy()
            
            im = axes[ch].imshow(data,
                               cmap=cmap,
                               aspect='equal',
                               origin='lower',
                               interpolation='nearest',
                               vmin=global_min,
                               vmax=global_max,
                               extent=[0, time_bins, 0, freq_bins])
            
            axes[ch].set_title(f'Channel {ch}', fontsize=14, fontweight='bold')
            axes[ch].set_xlabel('Time', fontsize=12)
            axes[ch].set_ylabel('Frequency Bands', fontsize=12)
            
            # 设置频率带标签
            axes[ch].set_yticks(np.arange(freq_bins) + 0.5)
            axes[ch].set_yticklabels(self.frequency_bands, fontsize=8)
            
            # 添加网格
            axes[ch].set_xticks(np.arange(time_bins + 1) - 0.5, minor=True)
            axes[ch].set_yticks(np.arange(freq_bins + 1) - 0.5, minor=True)
            axes[ch].grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.3)
        
        # 添加全局颜色条
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Amplitude', fontsize=12)
        
        plt.suptitle(f'{title_prefix}Multi-Channel Time-Frequency Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"多通道时频图已保存到: {save_path}")
        
        return fig, axes
    
    def load_ucihar_activity_sample(self, activity_id, batch_size=1):
        """
        从UCIHAR数据集加载特定活动类型的样本
        
        Args:
            activity_id: 活动ID (0-5对应WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
            batch_size: 批次大小
            
        Returns:
            sample_data: UCIHAR特定活动样本数据 [B, C, T]
        """
        from dataset_process.dataset_UCIHAR import load_uci_har_split
        
        # 活动标签映射
        activity_names = {
            0: "WALKING",
            1: "WALKING_UPSTAIRS", 
            2: "WALKING_DOWNSTAIRS",
            3: "SITTING",
            4: "STANDING",
            5: "LAYING"
        }
        
        # 加载UCIHAR训练数据
        data_dir = "dataset/UCIHAR"
        X, y = load_uci_har_split(data_dir, split='train', window_size=128, step=64)
        
        # 找到指定活动的所有样本
        activity_indices = np.where(y == activity_id)[0]
        
        if len(activity_indices) == 0:
            print(f"⚠️ No {activity_names.get(activity_id, 'unknown')} samples found")
            return None
        
        # 随机选择样本
        selected_indices = np.random.choice(activity_indices, 
                                          size=min(batch_size, len(activity_indices)), 
                                          replace=False)
        
        # 提取选中的样本
        selected_samples = X[selected_indices]  # [B, T, C]
        selected_labels = y[selected_indices]
        
        # 转换为PyTorch张量并调整维度顺序为 [B, C, T]
        sample_data = torch.tensor(selected_samples, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        
        print(f"📊 Loaded {len(selected_samples)} real UCIHAR {activity_names.get(activity_id, 'unknown')} samples")
        print(f"   Sample shape: {sample_data.shape}")
        print(f"   Data range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
        
        return sample_data

    def load_ucihar_walking_sample(self, batch_size=4):
        """
        从UCIHAR数据集加载真实的walking样本
        
        Args:
            batch_size: 批次大小
            
        Returns:
            sample_data: UCIHAR walking样本数据 [B, C, T]
        """
        from dataset_process.dataset_UCIHAR import load_uci_har_split
        
        # 加载UCIHAR训练数据
        data_dir = "dataset/UCIHAR"
        X, y = load_uci_har_split(data_dir, split='train', window_size=128, step=64)
        
        # 找到所有walking样本 (标签0对应WALKING)
        walking_indices = np.where(y == 0)[0]
        
        if len(walking_indices) == 0:
            print("⚠️ No walking samples found, using synthetic data instead")
            return self.generate_sample_data(batch_size)
        
        # 随机选择walking样本
        selected_indices = np.random.choice(walking_indices, 
                                          size=min(batch_size, len(walking_indices)), 
                                          replace=False)
        
        # 提取选中的样本
        selected_samples = X[selected_indices]  # [B, T, C]
        selected_labels = y[selected_indices]
        
        # 转换为PyTorch张量并调整维度顺序为 [B, C, T]
        sample_data = torch.tensor(selected_samples, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        
        print(f"📊 Loaded {len(selected_samples)} real UCIHAR walking samples")
        print(f"   Sample shape: {sample_data.shape}")
        print(f"   Labels: {selected_labels}")
        print(f"   Data range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
        
        return sample_data

    def generate_sample_data(self, batch_size=4):
        """
        生成示例数据（备用方法）
        
        Args:
            batch_size: 批次大小
            
        Returns:
            sample_data: 示例数据 [B, C, T]
        """
        # 生成模拟的传感器数据
        time_steps = 128
        channels = 6
        
        # 创建包含多种频率成分的信号
        t = np.linspace(0, 4, time_steps)
        sample_data = []
        
        for b in range(batch_size):
            batch_data = []
            for c in range(channels):
                # 组合多个频率成分
                signal = (np.sin(2 * np.pi * 1 * t) +  # 低频
                         0.5 * np.sin(2 * np.pi * 5 * t) +  # 中频
                         0.3 * np.sin(2 * np.pi * 10 * t) +  # 高频
                         0.1 * np.random.randn(time_steps))  # 噪声
                
                # 添加通道特异性
                signal *= (1 + 0.2 * c)
                batch_data.append(signal)
            
            sample_data.append(batch_data)
        
        return torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        
    def save_raw_time_frequency_image(self, time_freq_map, channel_idx=0, 
                                    sample_idx=0, save_path=None, 
                                    normalize=True, format='png', colormap='gray'):
        """
        Save time-frequency map as raw 2D image where each pixel value 
        corresponds to the time-frequency map value at that position
        
        Args:
            time_freq_map: Time-frequency map data [B, C, freq_bins, time_bins]
            channel_idx: Channel index
            sample_idx: Sample index
            save_path: Save path
            normalize: Whether to normalize values to 0-255 range
            format: Image format ('png', 'jpg', 'tiff')
            colormap: Color scheme ('gray', 'hot', 'jet', 'viridis', 'plasma', 'cool', 'warm')
        """
        from PIL import Image
        import matplotlib.cm as cm
        
        # Extract single channel data
        data = time_freq_map[sample_idx, channel_idx].cpu().numpy()  # [freq_bins, time_bins]
        
        # Flip vertically to match standard image coordinate system
        data = np.flipud(data)
        
        # Normalize data to 0-1 range for colormap application
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data_normalized = (data - data_min) / (data_max - data_min)
        else:
            data_normalized = np.zeros_like(data)
        
        # Apply colormap
        if colormap == 'gray':
            # Grayscale
            if normalize:
                pixel_data = (data_normalized * 255).astype(np.uint8)
            else:
                pixel_data = np.clip(data, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixel_data, mode='L')
        else:
            # Apply matplotlib colormap
            colormap_func = getattr(cm, colormap, cm.viridis)
            colored_data = colormap_func(data_normalized)  # Returns RGBA
            
            # Convert to RGB (remove alpha channel) and scale to 0-255
            rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb_data, mode='RGB')
        
        # Save image
        if save_path:
            img.save(save_path, format=format.upper())
            print(f"Raw time-frequency image ({colormap}) saved to: {save_path}")
            print(f"Image size: {img.size} (width x height)")
            if colormap == 'gray':
                print(f"Pixel value range: [{pixel_data.min()}, {pixel_data.max()}]")
            else:
                print(f"Original data range: [{data_min:.4f}, {data_max:.4f}]")
        
        return img, data
    
    def save_raw_time_frequency_data(self, time_freq_map, channel_idx=0, 
                                   sample_idx=0, save_path=None):
        """
        Save time-frequency map as raw numpy array
        
        Args:
            time_freq_map: Time-frequency map data [B, C, freq_bins, time_bins]
            channel_idx: Channel index
            sample_idx: Sample index
            save_path: Save path (.npy file)
        """
        # Extract single channel data
        data = time_freq_map[sample_idx, channel_idx].cpu().numpy()  # [freq_bins, time_bins]
        
        # Save as numpy array
        if save_path:
            np.save(save_path, data)
            print(f"Raw time-frequency data saved to: {save_path}")
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Value range: [{data.min():.6f}, {data.max():.6f}]")
        
        return data

def main():
    """Main function: Generate time-frequency maps for all six activities"""
    print("🎯 WPDN Time-Frequency Map Generator for All Activities")
    print("=" * 60)
    
    # Create visualizer
    visualizer = TimeFrequencyVisualizer(device='cpu')
    
    # Activity names mapping
    activity_names = {
        0: "WALKING",
        1: "WALKING_UPSTAIRS", 
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING"
    }
    
    # Create output directory
    output_dir = "time_frequency_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate time-frequency maps for each activity
    for activity_id in range(6):
        activity_name = activity_names[activity_id]
        print(f"\n🏃 Processing {activity_name} (ID: {activity_id})...")
        print("-" * 50)
        
        # Load activity-specific sample
        sample_data = visualizer.load_ucihar_activity_sample(activity_id, batch_size=1)
        
        if sample_data is None:
            print(f"⚠️ Skipping {activity_name} due to no available samples")
            continue
            
        print(f"Data shape: {sample_data.shape}")
        
        # Extract time-frequency features with normalization
        print("🔄 Extracting time-frequency features (with normalization)...")
        frequency_bands, time_freq_map = visualizer.extract_time_frequency_features(sample_data, normalize=True)
        print(f"Time-frequency map shape: {time_freq_map.shape}")
        print(f"Number of frequency bands: {len(frequency_bands)}")
        print(f"Normalized value range: [{time_freq_map.min():.4f}, {time_freq_map.max():.4f}]")
        
        # Check if channel 5 exists
        num_channels = time_freq_map.shape[1]
        if num_channels > 5:
            # Generate standard time-frequency plot for channel 5
            print(f"📊 Generating standard time-frequency plot for {activity_name} channel 5...")
            fig, ax = visualizer.plot_single_channel_time_frequency(
                time_freq_map,
                channel_idx=5,
                sample_idx=0,
                title_prefix=f"{activity_name} - ",
                colormap='custom',
                save_path=f"{output_dir}/{activity_name.lower()}_ch5_timefreq_plot.pdf"
            )
            plt.close(fig)  # Close to free memory
            
            # Generate raw time-frequency image for channel 5
            print(f"💾 Generating raw time-frequency image for {activity_name} channel 5...")
            visualizer.save_raw_time_frequency_image(
                time_freq_map,
                channel_idx=5,
                sample_idx=0,
                colormap='viridis',
                save_path=f"{output_dir}/{activity_name.lower()}_ch5_raw_viridis.png"
            )
            
            # Save original values as numpy array for channel 5
            raw_data = visualizer.save_raw_time_frequency_data(
                time_freq_map,
                channel_idx=5,
                sample_idx=0,
                save_path=f"{output_dir}/{activity_name.lower()}_ch5_data.npy"
            )
            
            # Display statistics for channel 5
            print(f"📈 {activity_name} Channel 5 time-frequency map statistics:")
            ch5_data = time_freq_map[0, 5, :, :]  # Get channel 5 data
            print(f"   Number of frequency bands: {ch5_data.shape[0]}")
            print(f"   Number of time points: {ch5_data.shape[1]}")
            print(f"   Value range: [{ch5_data.min():.4f}, {ch5_data.max():.4f}]")
            print(f"   Mean value: {ch5_data.mean():.4f}")
            print(f"   Standard deviation: {ch5_data.std():.4f}")
            
            print(f"✅ {activity_name} time-frequency maps generated successfully!")
        else:
            print(f"❌ Channel 5 not available for {activity_name}. Total channels: {num_channels}")
    
    print(f"\n🎯 All activity time-frequency maps generated successfully!")
    print(f"📁 Files saved to '{output_dir}' directory")
    print("\n📋 Generated files for each activity:")
    print("   - [activity]_ch5_timefreq_plot.pdf (standard plot)")
    print("   - [activity]_ch5_raw_viridis.png (raw image)")
    print("   - [activity]_ch5_data.npy (numpy array)")

if __name__ == "__main__":
    main()