import os
import torch

from config import DatasetConfig, ModelConfig
from model.cnn_models import LightweightCNN
from model.model_wpdn import LightweightWaveletPacketCNN


class ModelFactory:
    @staticmethod
    def create_model(
        mode: str,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        device: torch.device,
    ) -> torch.nn.Module:
        if mode == "wavelet_lite":
            model = LightweightWaveletPacketCNN(
                in_channels=dataset_config.in_channels,
                num_classes=dataset_config.num_classes,
                input_length=dataset_config.input_length,
                kernel_size=dataset_config.kernel_size,
                use_parallel=False,
                num_parallel_groups=model_config.num_parallel_groups,
                wavelet_type=model_config.wavelet_type,
                wavelet_levels=model_config.wavelet_levels,
                decompose_levels=model_config.decompose_levels,
                verbose=False,
            ).to(device)
            print("wavelet_lite")
            print(f"  decompose_levels={model_config.decompose_levels}")
            print("  classifier=band-local sparse 1D head")
            return model

        if mode == "cnn_lite":
            model = LightweightCNN(
                in_channels=dataset_config.in_channels,
                num_classes=dataset_config.num_classes,
                input_length=dataset_config.input_length,
                verbose=False,
            ).to(device)
            print("cnn_lite")
            return model

        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> bool:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
        return True

