import torch
import torch.nn as nn


class ConvFeatureClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)
        return self.classifier(x)
