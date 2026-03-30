from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DatasetConfig:
    name: str
    in_channels: int
    num_classes: int
    input_length: int
    kernel_size: int


@dataclass
class ModelConfig:
    mode: str
    wavelet_type: str = "db4"
    wavelet_levels: int = 3
    decompose_levels: int = 2
    use_parallel_wavelet_kernels: bool = True
    num_parallel_groups: int = 4
    classifier_factor_rank: Optional[int] = 10
    classifier_feature_groups: Optional[int] = None

    @property
    def wavelet_rank_max(self) -> Optional[int]:
        return self.classifier_factor_rank

    @wavelet_rank_max.setter
    def wavelet_rank_max(self, value: Optional[int]) -> None:
        self.classifier_factor_rank = value

    @property
    def wavelet_out_feature_groups(self) -> Optional[int]:
        return self.classifier_feature_groups

    @wavelet_out_feature_groups.setter
    def wavelet_out_feature_groups(self, value: Optional[int]) -> None:
        self.classifier_feature_groups = value


class Config:
    DATASET_CONFIGS = {
        "UCIHAR": DatasetConfig("UCIHAR", 6, 6, 128, 6),
        "WISDM": DatasetConfig("WISDM", 3, 6, 128, 6),
        "PAMAP2": DatasetConfig("PAMAP2", 18, 12, 256, 6),
        "MHEALTH": DatasetConfig("MHEALTH", 12, 12, 128, 6),
    }

    @classmethod
    def get_dataset_config(cls, dataset_name: str) -> DatasetConfig:
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return cls.DATASET_CONFIGS[dataset_name]

    @staticmethod
    def setup_device(device_str: str = "auto") -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @staticmethod
    def get_checkpoint_path(mode: str, dataset_name: str) -> str:
        return f"checkpoints/best_{mode}_{dataset_name}.pth"

    @staticmethod
    def get_results_dir() -> str:
        return "results"
