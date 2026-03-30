"""
实验运行时工具。

职责：
1. 从 experiment_config.py 解析统一实验配置
2. 为训练/测试构造 stage-specific 的 DatasetConfig / ModelConfig
3. 提供统一的模型准备逻辑，避免 train/test/experiment 脚本重复实现
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from experiment_config import (
    ACTIVE_TEST_EXPERIMENT,
    ACTIVE_TRAIN_EXPERIMENT,
    DATASET_PRESETS,
    ExperimentPreset,
    ExperimentSelection,
    GLOBAL_PRESET,
    MODEL_DATASET_PRESETS,
    MODEL_PRESETS,
)
from utils.config import Config, DatasetConfig, ModelConfig, TestConfig, TrainingConfig
from utils.dataset_utils import DatasetLoader
from utils.model_factory import ModelFactory


SECTION_KEYS = ("dataset", "model", "training", "test", "runtime")


@dataclass
class ExperimentBundle:
    """解析后的实验上下文。"""

    selection: ExperimentSelection
    stage: str
    resolved: Dict[str, Dict[str, Any]]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    device: torch.device
    training_config: Optional[TrainingConfig] = None
    test_config: Optional[TestConfig] = None


def _empty_sections() -> Dict[str, Dict[str, Any]]:
    return {key: {} for key in SECTION_KEYS}


def _merge_preset(target: Dict[str, Dict[str, Any]], preset: Optional[ExperimentPreset]) -> None:
    if preset is None:
        return
    for key in SECTION_KEYS:
        target[key].update(copy.deepcopy(getattr(preset, key)))


def _filtered_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in values.items() if k in cls.__dataclass_fields__}


def resolve_experiment(selection: ExperimentSelection) -> Dict[str, Dict[str, Any]]:
    """解析某个模型-数据集组合的最终配置。"""
    resolved = _empty_sections()
    _merge_preset(resolved, GLOBAL_PRESET)
    _merge_preset(resolved, MODEL_PRESETS.get(selection.mode))
    _merge_preset(resolved, DATASET_PRESETS.get(selection.dataset))
    _merge_preset(resolved, MODEL_DATASET_PRESETS.get((selection.mode, selection.dataset)))
    return resolved


def _build_dataset_config(selection: ExperimentSelection, resolved: Dict[str, Dict[str, Any]], stage: str) -> DatasetConfig:
    dataset_config = Config.get_dataset_config(selection.dataset)

    for key, value in resolved["dataset"].items():
        setattr(dataset_config, key, value)

    if stage == "train":
        batch_size = resolved["training"].get("batch_size")
    else:
        batch_size = resolved["test"].get("batch_size")

    if batch_size is not None:
        dataset_config.batch_size = batch_size

    return dataset_config


def _build_model_config(
    selection: ExperimentSelection,
    resolved: Dict[str, Dict[str, Any]],
    device_str: str,
) -> ModelConfig:
    values = copy.deepcopy(resolved["model"])
    values["mode"] = selection.mode
    values["device"] = device_str
    return ModelConfig(**_filtered_kwargs(ModelConfig, values))


def _build_training_config(resolved: Dict[str, Dict[str, Any]]) -> TrainingConfig:
    return TrainingConfig(**_filtered_kwargs(TrainingConfig, resolved["training"]))


def _build_test_config(resolved: Dict[str, Dict[str, Any]]) -> TestConfig:
    return TestConfig(**_filtered_kwargs(TestConfig, resolved["test"]))


def build_experiment_bundle(
    selection: ExperimentSelection,
    stage: str,
    device_override: Optional[str] = None,
) -> ExperimentBundle:
    resolved = resolve_experiment(selection)
    runtime_device = device_override or resolved["runtime"].get("device", "auto")
    if stage == "test":
        runtime_device = device_override or resolved["runtime"].get("test_device", runtime_device)

    device = Config.setup_device(runtime_device)
    dataset_config = _build_dataset_config(selection, resolved, stage=stage)
    model_config = _build_model_config(selection, resolved, str(device))

    training_config = _build_training_config(resolved) if stage == "train" else None
    test_config = _build_test_config(resolved) if stage == "test" else None

    return ExperimentBundle(
        selection=selection,
        stage=stage,
        resolved=resolved,
        dataset_config=dataset_config,
        model_config=model_config,
        device=device,
        training_config=training_config,
        test_config=test_config,
    )


def build_active_train_bundle(device_override: Optional[str] = None) -> ExperimentBundle:
    return build_experiment_bundle(ACTIVE_TRAIN_EXPERIMENT, stage="train", device_override=device_override)


def build_active_test_bundle(device_override: Optional[str] = None) -> ExperimentBundle:
    return build_experiment_bundle(ACTIVE_TEST_EXPERIMENT, stage="test", device_override=device_override)


def create_model_for_bundle(bundle: ExperimentBundle) -> torch.nn.Module:
    return ModelFactory.create_model(
        bundle.model_config.mode,
        bundle.dataset_config,
        bundle.model_config,
        bundle.device,
    )


def create_training_setup(bundle: ExperimentBundle):
    train_loader = DatasetLoader.create_train_loader(bundle.dataset_config)
    val_loader = DatasetLoader.create_val_loader(bundle.dataset_config)
    model = create_model_for_bundle(bundle)
    return model, train_loader, val_loader


def create_test_setup(bundle: ExperimentBundle):
    test_loader = DatasetLoader.create_test_loader(bundle.dataset_config)
    class_names = DatasetLoader.get_class_names(bundle.dataset_config)
    model = create_model_for_bundle(bundle)
    return model, test_loader, class_names


def prepare_model_for_inference(model: torch.nn.Module, bundle: ExperimentBundle) -> torch.nn.Module:
    """统一推理准备逻辑。"""
    runtime = bundle.resolved["runtime"]

    if (
        runtime.get("switch_to_deploy", True)
        and bundle.model_config.mode in ["wavelet_lite", "wavelet_traditional"]
        and hasattr(model, "switch_to_deploy")
    ):
        model.switch_to_deploy()

    if (
        runtime.get("enable_wavelet_cpu_fast_path", True)
        and bundle.model_config.mode == "wavelet_lite"
        and bundle.device.type == "cpu"
        and hasattr(model, "enable_cpu_fast_classifier")
    ):
        model.enable_cpu_fast_classifier()

    return model


def get_checkpoint_path(bundle: ExperimentBundle) -> str:
    return Config.get_model_checkpoint_path(bundle.model_config.mode, bundle.dataset_config.name)
