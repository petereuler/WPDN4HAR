#!/usr/bin/env python3
"""
将所有六种运动状态的时频图像转换为纯净的PDF文件
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

def convert_png_to_pdf_clean(png_path, pdf_path):
    """
    将PNG图像转换为纯净的PDF（无标题、无文字）
    
    Args:
        png_path: PNG文件路径
        pdf_path: 输出PDF文件路径
    """
    try:
        # 读取PNG图像
        img = Image.open(png_path)
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 显示图像，不添加任何文字或标题
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
        
        # 移除所有边距和空白
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 保存为PDF
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        
        print(f"✅ 成功转换: {os.path.basename(png_path)} -> {os.path.basename(pdf_path)}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败 {png_path}: {e}")
        return False

def main():
    """主函数：转换所有活动的时频图像为PDF"""
    print("🎯 将所有运动状态的时频图像转换为PDF")
    print("=" * 50)
    
    # 输入输出目录
    input_dir = "time_frequency_outputs"
    
    # 活动名称列表
    activity_names = [
        "walking",
        "walking_upstairs", 
        "walking_downstairs",
        "sitting",
        "standing",
        "laying"
    ]
    
    # 中文活动名称映射
    activity_chinese_names = {
        "walking": "行走",
        "walking_upstairs": "上楼", 
        "walking_downstairs": "下楼",
        "sitting": "坐着",
        "standing": "站立",
        "laying": "躺下"
    }
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    success_count = 0
    total_count = 0
    
    # 处理每个活动
    for activity in activity_names:
        total_count += 1
        
        # 构建文件路径
        png_file = f"{activity}_ch5_raw_viridis.png"
        png_path = os.path.join(input_dir, png_file)
        
        # 输出PDF文件名
        chinese_name = activity_chinese_names.get(activity, activity)
        pdf_file = f"{activity}_ch5_clean.pdf"
        pdf_path = os.path.join(input_dir, pdf_file)
        
        print(f"\n🏃 处理 {chinese_name} ({activity})...")
        
        # 检查PNG文件是否存在
        if not os.path.exists(png_path):
            print(f"⚠️ PNG文件不存在: {png_file}")
            continue
        
        # 转换为PDF
        if convert_png_to_pdf_clean(png_path, pdf_path):
            success_count += 1
            
            # 验证生成的PDF文件
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path)
                print(f"   📄 PDF文件大小: {file_size} bytes")
            else:
                print(f"   ❌ PDF文件生成失败")
    
    # 总结
    print(f"\n📊 转换完成统计:")
    print(f"   成功转换: {success_count}/{total_count}")
    print(f"   输出目录: {input_dir}")
    
    if success_count > 0:
        print(f"\n✅ 成功生成 {success_count} 个纯净PDF文件!")
        print("📋 生成的PDF文件:")
        
        # 列出生成的PDF文件
        for activity in activity_names:
            pdf_file = f"{activity}_ch5_clean.pdf"
            pdf_path = os.path.join(input_dir, pdf_file)
            if os.path.exists(pdf_path):
                chinese_name = activity_chinese_names.get(activity, activity)
                print(f"   - {pdf_file} ({chinese_name})")
    else:
        print("❌ 没有成功转换任何文件")

if __name__ == "__main__":
    main()