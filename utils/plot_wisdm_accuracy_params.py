#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WISDM数据集精度-参数分布图绘制脚本
基于表格数据绘制各种方法在WISDM数据集上的精度与参数量的二维分布图
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from adjustText import adjust_text  # 用于自动调整文本位置避免重叠

# 设置现代化样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 设置字体支持
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'

def plot_wisdm_accuracy_params():
    """绘制WISDM数据集上各方法的精度-参数分布图"""
    
    # WISDM数据集的数据 (方法名, 精度%, 参数量K)
    data = [
        # 基础方法
        ('LSTM', 95.54, 573.0),
        ('LSTM-lite', 94.86, 18.1),
        ('GRU', 98.92, 440.1),
        ('GRU-lite', 97.42, 13.6),
        ('Transformer', 96.06, 802.2),
        ('Transformer-lite', 97.79, 67.6),
        ('CNN', 99.72, 282.2),
        ('CNN-lite', 97.29, 19.6),
        ('ResNet', 98.95, 966.2),
        ('ResNet-lite', 97.85, 97.0),
        ('WPDN', 96.10, 0.8),
        
        # 对比方法
        ('TinyHAR', 83.47, 16.2),
        ('HS-CNN', 97.76, 530.0),
        ('HS-ResNet', 99.02, 830.0),
        ('CE-HAR', 99.04, 430.0),
        ('L-MHTCN', 99.98, 210.0),
        ('PQ-CNN', 95.12, 47.5),
    ]
    
    # 分离数据
    methods = [item[0] for item in data]
    accuracies = [item[1] for item in data]
    params = [item[2] for item in data]
    
    # 创建图形 - 使用更小的方形尺寸
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # 设置背景渐变
    ax.set_facecolor('#fafafa')
    
    # 定义现代化颜色方案
    color_scheme = {
        'wpdn': '#E74C3C',      # 鲜艳红色
        'lightweight': '#3498DB',  # 蓝色
        'standard': '#2ECC71',     # 绿色
        'comparison': '#F39C12',   # 橙色
        'grid': '#E8E8E8'         # 浅灰色网格
    }
    
    # 定义颜色和标记样式
    colors = []
    markers = []
    sizes = []
    edge_colors = []
    alphas = []
    
    for method in methods:
        if method == 'WPDN':
            colors.append(color_scheme['wpdn'])
            markers.append('*')
            sizes.append(300)  # 更大的星形
            edge_colors.append('#C0392B')  # 深红色边框
            alphas.append(0.9)
        elif 'lite' in method.lower():
            colors.append(color_scheme['lightweight'])
            markers.append('o')
            sizes.append(120)
            edge_colors.append('#2980B9')  # 深蓝色边框
            alphas.append(0.8)
        elif method in ['TinyHAR', 'HS-CNN', 'HS-ResNet', 'CE-HAR', 'L-MHTCN', 'PQ-CNN']:
            colors.append(color_scheme['comparison'])
            markers.append('^')
            sizes.append(120)
            edge_colors.append('#E67E22')  # 深橙色边框
            alphas.append(0.8)
        else:
            colors.append(color_scheme['standard'])
            markers.append('s')
            sizes.append(120)
            edge_colors.append('#27AE60')  # 深绿色边框
            alphas.append(0.8)
    
    # 绘制散点图 - 使用更精美的样式和手动标签定位
    texts = []  # 存储所有文本标注对象
    
    # 定义手动标签偏移位置，避免重叠 - 优化版本，适应文本框布局
    label_offsets = {
        'WPDN': (0, 50),           # 右上方，突出显示，增大偏移
        'GRU': (-20, -60),           # 左上方，增大偏移
        'GRU-lite': (-20, 20),     # 左下方，增大偏移
        'Transformer-lite': (-30, 20),  # 右下方，增大偏移
        'CNN': (10, 15),            # 右上方，增大偏移
        'CNN-lite': (10, -15),      # 左上方，增大偏移
        'ResNet': (10, -25),         # 右上方，增大偏移
        'ResNet-lite': (15, -15),  # 左下方，增大偏移
        'LSTM': (-30, -10),          # 右下方，增大偏移
        'LSTM-lite': (-15, -15),     # 左上方，增大偏移
        'Transformer': (40, -15),    # 右上方，增大偏移
        'TinyHAR': (-10, 15),      # 左下方，增大偏移
        'HS-CNN': (35, -25),        # 右下方，增大偏移
        'HS-ResNet': (0, 50),     # 左上方，增大偏移
        'CE-HAR': (-40, -30),        # 右下方，增大偏移
        'L-MHTCN': (-30, 15),        # 左上方，增大偏移
        'PQ-CNN': (5, -15)        # 左上方，增大偏移
    }
    
    for i, (method, acc, param) in enumerate(data):
        scatter = ax.scatter(param, acc, c=colors[i], marker=markers[i], s=sizes[i], 
                           alpha=alphas[i], edgecolors=edge_colors[i], linewidth=2,
                           zorder=5 if method == 'WPDN' else 3)
        
        # 为WPDN添加光晕效果
        if method == 'WPDN':
            ax.scatter(param, acc, c=colors[i], marker=markers[i], s=sizes[i]*1.5, 
                      alpha=0.3, edgecolors='none', zorder=2)
        
        # 获取标签偏移位置
        offset = label_offsets.get(method, (15, 15))
        
        # 为所有方法添加带文本框和连接线的标签
        if method == 'WPDN':
            # WPDN使用特殊样式 - 红色填充白色字体，圆角文本框
            text = ax.annotate(method, (param, acc), xytext=offset, 
                        textcoords='offset points', fontsize=18, fontweight='bold',
                        color='white', ha='center', va='center', zorder=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#E74C3C', 
                                 edgecolor='#C0392B', linewidth=2.5, alpha=0.95),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                      color='#E74C3C', lw=2.5, alpha=0.9, 
                                      shrinkA=5, shrinkB=5))
        else:
            # 需要移除连接线的方法列表
            methods_without_lines = ['LSTM-lite', 'TinyHAR', 'PQ-CNN', 'LSTM', 'GRU-lite', 'CNN-lite', 'L-MHTCN', 'CNN', 'Transformer-lite', 'ResNet-lite', 'Transformer']
            
            if method in methods_without_lines:
                # 这些方法只显示文字，无文本框和连接线
                text = ax.annotate(method, (param, acc), xytext=offset, 
                            textcoords='offset points', fontsize=18, fontweight='normal',
                            color='black', ha='center', va='center', zorder=8,
                            bbox=None,
                            arrowprops=None)
            else:
                # 其他方法显示文字和连接线，无文本框
                text = ax.annotate(method, (param, acc), xytext=offset, 
                            textcoords='offset points', fontsize=18, fontweight='normal',
                            color='black', ha='center', va='center', zorder=8,
                            bbox=None,
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15',
                                          color='black', lw=1.8, alpha=0.85,
                                          shrinkA=3, shrinkB=3))
        
        texts.append(text)
    
    # 移除自动调整，使用手动定位
    # adjust_text已被手动定位替代，确保标签位置清晰合理
    
    # 设置坐标轴 - 现代化样式，进一步放大轴标签字体
    ax.set_xlabel('Parameters (K)', fontsize=22, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Accuracy (%)', fontsize=22, fontweight='bold', color='#2C3E50')
    # 移除标题
    # ax.set_title('Accuracy vs Parameters on WISDM Dataset',
    #              fontsize=22, fontweight='bold', color='#2C3E50', pad=20)
    
    # 增大刻度标签字体并美化
    ax.tick_params(axis='both', which='major', labelsize=20, colors='#34495E')
    ax.tick_params(axis='both', which='minor', labelsize=20, colors='#34495E')
    
    # 设置现代化网格 - 调整为更明显的网格线
    ax.grid(True, alpha=0.7, linestyle='-', linewidth=1.2, color='#D0D0D0')
    ax.set_axisbelow(True)  # 网格在数据点下方
    
    # 设置坐标轴范围 - 自适应调整
    param_min, param_max = min(params), max(params)
    acc_min, acc_max = min(accuracies), max(accuracies)
    
    # X轴保持对数刻度，但调整范围
    ax.set_xscale('log')
    ax.set_xlim(param_min * 0.5, param_max * 2)
    
    # Y轴范围调整，留出适当边距
    acc_range = acc_max - acc_min
    ax.set_ylim(acc_min - acc_range * 0.05, acc_max + acc_range * 0.1)
    
    # 美化坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#BDC3C7')
    
    # 创建现代化图例
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color_scheme['wpdn'], 
                  markersize=15, markeredgecolor='#C0392B', markeredgewidth=2,
                  label='WPDN (Our Method)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_scheme['lightweight'], 
                  markersize=10, markeredgecolor='#2980B9', markeredgewidth=1.5,
                  label='Lightweight Methods', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color_scheme['standard'], 
                  markersize=10, markeredgecolor='#27AE60', markeredgewidth=1.5,
                  label='Standard Methods', linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=color_scheme['comparison'], 
                  markersize=10, markeredgecolor='#E67E22', markeredgewidth=1.5,
                  label='Related Works', linestyle='None')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=22, 
                      frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                      facecolor='white', edgecolor='#BDC3C7', borderpad=1)
    
    # 添加帕累托前沿线 - 美化版本
    # 找到帕累托最优点
    pareto_points = []
    sorted_data = sorted(data, key=lambda x: x[2])  # 按参数量排序
    
    max_acc = 0
    for method, acc, param in sorted_data:
        if acc > max_acc:
            pareto_points.append((param, acc))
            max_acc = acc
    
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        
        # 绘制帕累托前沿线，使用渐变效果
        ax.plot(pareto_x, pareto_y, '--', color='#E74C3C', linewidth=3, alpha=0.8, 
               label='Pareto Frontier', zorder=4)
        ax.plot(pareto_x, pareto_y, '--', color='#F39C12', linewidth=1.5, alpha=0.6, 
               zorder=3)  # 添加内层线条创建渐变效果
    
    # 调整布局并保存 - 高质量输出
    plt.tight_layout(pad=2.0)
    
    # 保存为高分辨率图片
    plt.savefig('/home/admin407/code/zyshe/WPDN/wisdm_accuracy_params_plot.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.savefig('/home/admin407/code/zyshe/WPDN/wisdm_accuracy_params_plot.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    
    print("Charts saved as:")
    print("- wisdm_accuracy_params_plot.png")
    print("- wisdm_accuracy_params_plot.pdf")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n=== WISDM Dataset Statistics ===")
    print(f"Total methods: {len(data)}")
    print(f"Accuracy range: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
    print(f"Parameter range: {min(params):.1f}K - {max(params):.1f}K")
    print(f"WPDN accuracy: {data[10][1]:.2f}%")
    print(f"WPDN parameters: {data[10][2]:.1f}K")
    
    # 计算WPDN相对于其他方法的优势
    wpdn_params = data[10][2]
    wpdn_acc = data[10][1]
    
    print(f"\n=== WPDN Advantage Analysis ===")
    print(f"WPDN parameter reduction compared to smallest other method: {((min([p for p in params if p != wpdn_params]) - wpdn_params) / min([p for p in params if p != wpdn_params]) * 100):.1f}%")
    
    better_acc_methods = [m for m, a, p in data if a > wpdn_acc]
    if better_acc_methods:
        print(f"Methods with higher accuracy than WPDN: {', '.join(better_acc_methods)}")
    
    return plt.gcf()

if __name__ == "__main__":
    plot_wisdm_accuracy_params()