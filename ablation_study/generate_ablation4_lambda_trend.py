#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从最新的 ablation4 结果JSON生成lambda趋势图与CSV。
查找文件名模式：ablation_4_orthogonality_loss_results_*.json
输出：
- ablation4_lambda_trend_<timestamp>.png / .pdf
- ablation4_lambda_trend_<timestamp>.csv
"""
import os
import glob
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt

RESULT_PATTERN = "ablation_4_orthogonality_loss_results_*.json"


def load_latest_results():
    files = glob.glob(RESULT_PATTERN)
    if not files:
        raise FileNotFoundError("未找到消融实验4的结果JSON文件")
    # 按文件名时间排序，选择最新
    files.sort(reverse=True)
    latest = files[0]
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return latest, data


def parse_results(data):
    rows = []
    for k, v in data.items():
        try:
            w = float(k)
        except (TypeError, ValueError):
            w = float(str(k))
        if isinstance(v, dict) and "error" not in v:
            rows.append({
                "orth_weight": w,
                "use_orth": v.get("use_orthogonality_loss", w > 0),
                "best_val_acc": v.get("best_val_acc"),
                "test_accuracy": v.get("test_accuracy"),
                "test_f1": v.get("test_f1"),
                "final_orth_loss": v.get("final_orthogonality_loss")
            })
        else:
            rows.append({
                "orth_weight": w,
                "use_orth": v.get("use_orthogonality_loss", w > 0) if isinstance(v, dict) else (w > 0),
                "best_val_acc": None,
                "test_accuracy": None,
                "test_f1": None,
                "final_orth_loss": None
            })
    rows.sort(key=lambda r: r["orth_weight"])
    return rows


def save_csv(rows, timestamp):
    csv_file = f"ablation4_lambda_trend_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["orth_weight", "use_orth", "best_val_acc", "test_accuracy", "test_f1", "final_orth_loss"])
        for r in rows:
            writer.writerow([
                r["orth_weight"], r["use_orth"], r["best_val_acc"], r["test_accuracy"], r["test_f1"], r["final_orth_loss"]
            ])
    return csv_file


def plot_trend(rows, timestamp):
    weights = [r["orth_weight"] for r in rows if r["test_accuracy"] is not None]
    accs = [r["test_accuracy"] for r in rows if r["test_accuracy"] is not None]
    orths = [r["final_orth_loss"] for r in rows if r["final_orth_loss"] is not None]

    if not weights:
        raise RuntimeError("结果中没有有效的accuracy数据，无法绘制趋势图")

    plt.figure(figsize=(9,6), dpi=150)
    ax1 = plt.gca()
    ax1.plot(weights, accs, marker='o', color='#1f77b4', label='Test Accuracy')
    ax1.set_xlabel('Orthogonality Loss Weight (lambda)', fontsize=16)
    ax1.set_ylabel('Test Accuracy', fontsize=16, color='#1f77b4')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.6, linestyle='--', linewidth=0.9)

    if len(orths) == len(weights):
        ax2 = ax1.twinx()
        ax2.plot(weights, orths, marker='s', color='#ff7f0e', label='Final Orth Loss')
        ax2.set_ylabel('Final Orthogonality Loss', fontsize=16, color='#ff7f0e')
        ax2.tick_params(axis='y', labelsize=12, labelcolor='#ff7f0e')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best', fontsize=12)
    else:
        ax1.legend(loc='best', fontsize=12)

    plt.title('Ablation4: Effect of lambda on performance', fontsize=18)
    plt.tight_layout()

    png = f"ablation4_lambda_trend_{timestamp}.png"
    pdf = f"ablation4_lambda_trend_{timestamp}.pdf"
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()
    return png, pdf


def main():
    latest_file, data = load_latest_results()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = parse_results(data)
    csv_path = save_csv(rows, timestamp)
    png, pdf = plot_trend(rows, timestamp)
    print(f"使用结果文件: {latest_file}")
    print(f"已生成CSV: {csv_path}")
    print(f"已生成趋势图: {png}, {pdf}")


if __name__ == '__main__':
    main()