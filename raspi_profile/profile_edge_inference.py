import argparse
import csv
import json
import os
import resource
import statistics
import sys
import time
from datetime import datetime

import torch

from config import Config, ModelConfig
from model_factory import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Profile edge inference on Raspberry Pi")
    parser.add_argument("--dataset", type=str, default="UCIHAR", choices=["UCIHAR", "WISDM", "PAMAP2", "MHEALTH"])
    parser.add_argument("--mode", type=str, default="wavelet_lite", choices=["wavelet_lite", "cnn_lite"])
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--wavelet-type", type=str, default="db4")
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument("--decompose-levels", type=int, default=3)
    parser.add_argument("--num-parallel-groups", type=int, default=2)
    parser.add_argument("--switch-to-deploy", action="store_true", default=True)
    parser.add_argument("--no-switch-to-deploy", dest="switch_to_deploy", action="store_false")
    parser.add_argument("--wavelet-cpu-fast-classifier", action="store_true", default=True)
    parser.add_argument("--no-wavelet-cpu-fast-classifier", dest="wavelet_cpu_fast_classifier", action="store_false")
    parser.add_argument("--cpu-tune", action="store_true", default=False)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cpu-interop-threads", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--sampling-rate", type=float, default=50.0)
    parser.add_argument("--stream-points", type=int, default=1280)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--hop-size", type=int, default=64)
    return parser.parse_args()


def maybe_tune_cpu(args, device):
    if device.type != "cpu":
        return
    torch.backends.mkldnn.enabled = True
    if args.cpu_tune and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if args.cpu_tune:
        try:
            torch.set_num_interop_threads(args.cpu_interop_threads)
        except RuntimeError:
            pass


def get_ru_maxrss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024 / 1024
    return rss / 1024


def compute_model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return {
        "total_params": int(params),
        "model_size_mb": float((param_bytes + buffer_bytes) / 1024 / 1024),
    }


def benchmark_inference(model, device, input_shape, num_tests, batch_size):
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    warmup = max(20, min(100, num_tests))
    baseline_rss = get_ru_maxrss_mb()
    model.eval()
    times = []

    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy_input)
        for _ in range(num_tests):
            start = time.perf_counter()
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    peak_rss = get_ru_maxrss_mb()
    sorted_times = sorted(times)
    mean_ms = sum(times) / len(times)
    median_ms = statistics.median(times)
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    p99_ms = sorted_times[int(len(sorted_times) * 0.99)]
    throughput = 1000.0 * batch_size / mean_ms if mean_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "mean_per_sample_ms": mean_ms / batch_size,
        "throughput_samples_per_s": throughput,
        "peak_cpu_rss_mb": peak_rss,
        "peak_cpu_rss_delta_mb": max(0.0, peak_rss - baseline_rss),
    }


def benchmark_streaming(model, device, input_shape, args):
    channels, window_size = input_shape
    total_points = int(args.stream_points)
    hop_size = int(args.hop_size)
    sampling_rate = float(args.sampling_rate)
    stream = torch.randn(total_points, channels, device=device)
    buffer = torch.zeros(window_size, channels, device=device)
    times = []
    inference_count = 0

    model.eval()
    with torch.inference_mode():
        for idx in range(total_points):
            buffer = torch.roll(buffer, shifts=-1, dims=0)
            buffer[-1] = stream[idx]
            if idx + 1 < window_size:
                continue
            if (idx + 1 - window_size) % hop_size != 0:
                continue
            window = buffer.transpose(0, 1).unsqueeze(0)
            start = time.perf_counter()
            model(window)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
            inference_count += 1

    mean_ms = sum(times) / len(times) if times else 0.0
    required_windows_per_s = sampling_rate / hop_size if hop_size > 0 else 0.0
    achieved_windows_per_s = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "num_inferences": inference_count,
        "mean_ms": mean_ms,
        "mean_per_sample_ms": mean_ms,
        "required_windows_per_s": required_windows_per_s,
        "achieved_windows_per_s": achieved_windows_per_s,
        "realtime_factor": achieved_windows_per_s / required_windows_per_s if required_windows_per_s > 0 else 0.0,
    }


def save_benchmark_results(args, dataset_config, ckpt_info, model_stats, rows):
    results_dir = Config.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"edge_profile_{args.mode}_{dataset_config.name}_{timestamp}"
    json_path = os.path.join(results_dir, f"{base_name}.json")
    csv_path = os.path.join(results_dir, f"{base_name}.csv")
    payload = {
        "dataset": dataset_config.name,
        "mode": args.mode,
        "device": args.device,
        "checkpoint": ckpt_info,
        "model_stats": model_stats,
        "benchmarks": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def save_stream_results(args, dataset_config, ckpt_info, model_stats, row):
    results_dir = Config.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"edge_stream_{args.mode}_{dataset_config.name}_{timestamp}"
    json_path = os.path.join(results_dir, f"{base_name}.json")
    csv_path = os.path.join(results_dir, f"{base_name}.csv")
    payload = {
        "dataset": dataset_config.name,
        "mode": args.mode,
        "device": args.device,
        "checkpoint": ckpt_info,
        "model_stats": model_stats,
        "streaming": row,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return json_path, csv_path


def main():
    args = parse_args()
    dataset_config = Config.get_dataset_config(args.dataset)
    device = Config.setup_device(args.device)
    maybe_tune_cpu(args, device)

    model_config = ModelConfig(
        mode=args.mode,
        wavelet_type=args.wavelet_type,
        wavelet_levels=args.wavelet_levels,
        decompose_levels=args.decompose_levels,
        num_parallel_groups=args.num_parallel_groups,
    )
    model = ModelFactory.create_model(args.mode, dataset_config, model_config, device)

    checkpoint_path = args.checkpoint_path or Config.get_checkpoint_path(args.mode, dataset_config.name)
    checkpoint_loaded = False
    checkpoint_error = None
    if not args.no_checkpoint:
        try:
            checkpoint_loaded = ModelFactory.load_model_weights(model, checkpoint_path, device)
        except Exception as exc:
            checkpoint_error = str(exc)
            print(f"Checkpoint load failed: {exc}")

    if (
        args.switch_to_deploy
        and args.mode == "wavelet_lite"
        and hasattr(model, "decomposer")
        and hasattr(model.decomposer, "switch_to_deploy")
    ):
        model.decomposer.switch_to_deploy()

    if (
        args.wavelet_cpu_fast_classifier
        and device.type == "cpu"
        and args.mode == "wavelet_lite"
        and hasattr(model, "enable_cpu_fast_classifier")
    ):
        model.enable_cpu_fast_classifier()
        print("Enabled CPU fast classifier for wavelet_lite")

    input_shape = (dataset_config.in_channels, dataset_config.input_length)
    model_stats = compute_model_stats(model)
    ckpt_info = {
        "path": checkpoint_path,
        "loaded": checkpoint_loaded,
        "error": checkpoint_error,
    }

    if args.streaming:
        row = benchmark_streaming(model, device, input_shape, args)
        json_path, csv_path = save_stream_results(args, dataset_config, ckpt_info, model_stats, row)
        print(json.dumps(row, indent=2))
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV: {csv_path}")
        return

    rows = []
    for batch_size in args.batch_sizes:
        row = benchmark_inference(model, device, input_shape, args.num_tests, int(batch_size))
        row.update(model_stats)
        rows.append(row)

    print("\n=== Edge Inference Summary ===")
    print("{:<8} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}".format(
        "Batch", "Mean(ms)", "P95(ms)", "Mean/sample", "Throughput", "Params", "SizeMB"
    ))
    for row in rows:
        print("{:<8} {:>10.4f} {:>10.4f} {:>12.4f} {:>12.2f} {:>10} {:>10.3f}".format(
            row["batch_size"],
            row["mean_ms"],
            row["p95_ms"],
            row["mean_per_sample_ms"],
            row["throughput_samples_per_s"],
            row["total_params"],
            row["model_size_mb"],
        ))

    json_path, csv_path = save_benchmark_results(args, dataset_config, ckpt_info, model_stats, rows)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
