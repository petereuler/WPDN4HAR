"""
集中管理实验配置。

这个文件是训练、测试、多数据集实验的唯一入口：
1. 选择当前要跑的模型/数据集
2. 维护模型默认参数
3. 维护数据集默认参数
4. 维护特定“模型-数据集”组合的覆盖项
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ExperimentSelection:
    """一次实验的目标：某个模型在某个数据集上。"""
    mode: str
    dataset: str


@dataclass
class ExperimentPreset:
    """按模块拆分的配置覆盖项。"""
    dataset: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    test: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)


# 当前默认入口。训练/测试脚本都从这里取目标实验。
ACTIVE_TRAIN_EXPERIMENT = ExperimentSelection(mode="wavelet_lite", dataset="WISDM")
ACTIVE_TEST_EXPERIMENT = ExperimentSelection(mode="wavelet_lite", dataset="WISDM")

# 多数据集实验入口。
MULTI_DATASET_MODE = "wavelet_lite"
MULTI_DATASET_SWEEP: List[str] = ["UCIHAR", "WISDM", "PAMAP2", "MHEALTH"]
MULTI_DATASET_REPEAT = 1


# 全局默认配置。所有实验都会先继承这一层。
GLOBAL_PRESET = ExperimentPreset(
    training={
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "optimizer_type": "Adam",
        "orth_weight": 0.01,
    },
    test={
        "batch_size": 50,
        "num_inference_tests": 100,
        "tsne_perplexity": 30,
        "inference_batch_size": 1,
    },
    runtime={
        "device": "auto",
        "test_device": "cpu",
        "switch_to_deploy": True,
        "enable_wavelet_cpu_fast_path": True,
    },
)


# 按模型控制默认超参数。
MODEL_PRESETS: Dict[str, ExperimentPreset] = {
    "wavelet_traditional": ExperimentPreset(
        model={
            "wavelet_type": "db4",
            "wavelet_levels": 3,
        },
        runtime={
            "switch_to_deploy": False,
            "enable_wavelet_cpu_fast_path": False,
        },
    ),
    "wavelet_lite": ExperimentPreset(
        model={
            "wavelet_type": "db4",
            "wavelet_levels": 3,
            "decompose_levels": 2,
            "use_parallel_wavelet_kernels": True,
            "num_parallel_groups": 4,
            "classifier_factor_rank": 10,
            "classifier_feature_groups": None,
        },
        runtime={
            "switch_to_deploy": True,
            "enable_wavelet_cpu_fast_path": True,
        },
    ),
    "lstm": ExperimentPreset(),
    "gru": ExperimentPreset(),
    "transformer": ExperimentPreset(),
    "cnn": ExperimentPreset(),
    "resnet": ExperimentPreset(),
    "lstm_lite": ExperimentPreset(),
    "gru_lite": ExperimentPreset(),
    "transformer_lite": ExperimentPreset(),
    "cnn_lite": ExperimentPreset(),
    "resnet_lite": ExperimentPreset(),
}


# 按数据集控制默认超参数。
DATASET_PRESETS: Dict[str, ExperimentPreset] = {
    "UCIHAR": ExperimentPreset(
        dataset={"split_type": "stratified"},
    ),
    "WISDM": ExperimentPreset(
        dataset={"split_type": "stratified"},
    ),
    "PAMAP2": ExperimentPreset(
        dataset={"split_type": "stratified"},
    ),
    "MHEALTH": ExperimentPreset(
        dataset={
            "split_type": "subject_independent",
            "step_size": 64,
            "exclude_null": True,
        },
    ),
}


# 按“模型-数据集”组合控制超参数。
# 这里是后续做专项调参的主入口，尤其适合 wavelet_lite。
MODEL_DATASET_PRESETS: Dict[Tuple[str, str], ExperimentPreset] = {
    ("wavelet_lite", "UCIHAR"): ExperimentPreset(
        model={
            "decompose_levels": 2,
            "use_parallel_wavelet_kernels": True,
            "num_parallel_groups": 4,
            "classifier_factor_rank": 10,
        },
    ),
    ("wavelet_lite", "WISDM"): ExperimentPreset(
        model={
            "decompose_levels": 2,
            "use_parallel_wavelet_kernels": True,
            "num_parallel_groups": 2,
            "classifier_factor_rank": 8,
        },
    ),
    ("wavelet_lite", "PAMAP2"): ExperimentPreset(
        model={
            "decompose_levels": 2,
            "use_parallel_wavelet_kernels": True,
            "num_parallel_groups": 4,
            "classifier_factor_rank": 10,
        },
    ),
    ("wavelet_lite", "MHEALTH"): ExperimentPreset(
        model={
            "decompose_levels": 2,
            "use_parallel_wavelet_kernels": True,
            "num_parallel_groups": 4,
            "classifier_factor_rank": 10,
        },
    ),
}
