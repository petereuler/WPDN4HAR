# Raspberry Pi Inference Profiling

这个目录是独立的树莓派推理 profiling 包，当前内置的模型只有：

- `wavelet_lite`
- `cnn_lite`

只关注这些指标：

- 推理时间：`mean / median / p95 / p99`
- 单样本延迟：`mean_per_sample_ms`
- 吞吐：`throughput_samples_per_s`
- 模型大小：`total_params / model_size_mb`
- 进程峰值 RSS：`peak_cpu_rss_mb / peak_cpu_rss_delta_mb`

不包含：

- 准确率
- 混淆矩阵
- t-SNE
- matplotlib / sklearn / thop

## 目录

- `profile_edge_inference.py`：主脚本
- `config.py`：最小数据集配置
- `model_factory.py`：本地模型工厂
- `model/`：当前主工程同步过来的最小模型副本
- `checkpoints/`：建议把 `.pth` 放这里
- `results/`：运行后生成

## 树莓派安装

先准备 Python 环境，再安装最小依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` 在树莓派上要装对应 ARM 版本。若默认安装失败，需要换成你树莓派系统对应的 PyTorch wheel。

## 运行示例

在这个目录里执行：

```bash
python profile_edge_inference.py --dataset UCIHAR --mode cnn_lite --device cpu --batch-sizes 1 8 16 --num-tests 100
```

如果权重放在 `checkpoints/best_cnn_lite_UCIHAR.pth`，脚本会自动尝试加载。

不想加载权重时：

```bash
python profile_edge_inference.py --dataset UCIHAR --mode cnn_lite --device cpu --no-checkpoint
```

测试 `wavelet_lite` 并启用部署重参数化：

```bash
python profile_edge_inference.py --dataset UCIHAR --mode wavelet_lite --device cpu --batch-sizes 1 8 --num-tests 100 --switch-to-deploy --num-parallel-groups 4
```

当前默认 `num_parallel_groups=4`，和主工程里这版 `wavelet_lite` 的训练配置保持一致。

CPU 下会默认尝试启用 `wavelet_lite` 的 fast classifier。如果你想禁用它做对照：

```bash
python profile_edge_inference.py --dataset UCIHAR --mode wavelet_lite --device cpu --no-wavelet-cpu-fast-classifier
```

在 CPU 上调线程：

```bash
python profile_edge_inference.py --dataset UCIHAR --mode cnn_lite --device cpu --cpu-tune --cpu-threads 4 --cpu-interop-threads 1
```

## 输出

结果会写到 `results/`：

- `edge_profile_*.json`
- `edge_profile_*.csv`

CSV 每一行对应一个 batch size，适合直接比较不同模型在树莓派上的延迟和体积影响。
