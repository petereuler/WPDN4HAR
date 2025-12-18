#!/usr/bin/env python3
"""
将时频图像转换为PDF格式的脚本
Convert time-frequency images to PDF format
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def convert_png_to_pdf_clean(png_path, pdf_path):
    """
    将PNG图像转换为PDF格式（纯图片，无标题和文字）
    
    Args:
        png_path (str): PNG文件路径
        pdf_path (str): 输出PDF文件路径
    """
    
    if not os.path.exists(png_path):
        print(f"❌ PNG文件不存在: {png_path}")
        return False
    
    try:
        # 创建PDF文件
        with PdfPages(pdf_path) as pdf:
            # 读取PNG图像
            img = mpimg.imread(png_path)
            
            # 创建图形，设置合适的尺寸
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 显示图像，完全填充画布
            ax.imshow(img)
            ax.axis('off')  # 隐藏坐标轴
            
            # 移除所有边距和空白
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # 保存到PDF，去除所有边距
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close(fig)
        
        print(f"✅ 纯图片PDF文件已保存: {pdf_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return False

def convert_with_data_info(png_path, data_path, pdf_path):
    """
    将PNG图像和数据信息一起转换为PDF
    
    Args:
        png_path (str): PNG文件路径
        data_path (str): NPY数据文件路径
        pdf_path (str): 输出PDF文件路径
    """
    
    if not os.path.exists(png_path):
        print(f"❌ PNG文件不存在: {png_path}")
        return False
    
    try:
        # 创建PDF文件
        with PdfPages(pdf_path) as pdf:
            # 读取PNG图像
            img = mpimg.imread(png_path)
            
            # 读取数据信息（如果存在）
            data_info = ""
            if os.path.exists(data_path):
                data = np.load(data_path)
                data_info = f"""
数据统计信息:
• 数据形状: {data.shape}
• 数值范围: [{data.min():.4f}, {data.max():.4f}]
• 平均值: {data.mean():.4f}
• 标准差: {data.std():.4f}
• 数据类型: {data.dtype}
"""
            
            # 创建图形
            fig = plt.figure(figsize=(12, 10))
            
            # 主图像
            ax1 = plt.subplot(2, 1, 1)
            ax1.imshow(img)
            ax1.set_title("Channel 5 Raw Time-Frequency Map (Viridis Colormap)", 
                         fontsize=16, fontweight='bold', pad=20)
            ax1.axis('off')
            
            # 数据信息
            if data_info:
                ax2 = plt.subplot(2, 1, 2)
                ax2.text(0.05, 0.95, data_info, transform=ax2.transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                ax2.axis('off')
            
            # 添加元数据
            fig.text(0.02, 0.02, f"Generated from: {os.path.basename(png_path)}", 
                    fontsize=8, alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存到PDF
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        print(f"✅ 包含数据信息的PDF文件已保存: {pdf_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return False

def main():
    """主函数"""
    
    # 定义文件路径
    output_dir = "time_frequency_outputs"
    png_file = f"{output_dir}/raw_time_freq_ch5_viridis.png"
    
    # 输出PDF文件路径
    clean_pdf = f"{output_dir}/channel5_time_frequency_clean.pdf"
    
    print("🔄 开始转换PNG图像为纯图片PDF格式...")
    
    # 生成纯图片PDF（无标题、无文字）
    print("\n🖼️  生成纯图片PDF版本...")
    success = convert_png_to_pdf_clean(png_file, clean_pdf)
    
    # 总结
    print(f"\n{'='*50}")
    print("📋 转换结果总结:")
    if success:
        print(f"✅ 纯图片PDF: {clean_pdf}")
        print("🎉 纯图片PDF转换完成!")
    else:
        print("❌ PDF转换失败!")

if __name__ == "__main__":
    main()