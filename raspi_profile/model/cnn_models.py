import torch
import torch.nn as nn


class LightweightCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        input_length,
        base_channels=48,
        num_blocks=2,
        dropout=0.2,
        verbose=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout = dropout

        layers = []
        current_channels = in_channels
        current_length = input_length
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            layers.extend(
                [
                    nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = out_channels
            current_length = current_length // 2

        self.feature_extractor = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        hidden_size = current_channels // 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_channels, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print("LightweightCNN")
            print(f"  in_channels={in_channels}")
            print(f"  input_length={input_length}")
            print(f"  num_classes={num_classes}")
            print(f"  params={total_params}")

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.classifier(x)

