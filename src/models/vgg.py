"""VGG-16 (Simonyan & Zisserman, ICLR 2015) student implementation.

PA2 instruction: implement every layer directly from the paper, without using
``torchvision.models.vgg16`` or ``timm.create_model``.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.heads import MultiTaskHead


# 논문 Table 1의 configuration D(VGG-16)를 문서화한 참고용 목록이다.
# 실제 모델은 과제 요구사항에 맞춰 아래 nn.Sequential에 모든 레이어를 직접 적는다.
VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def make_vgg_layers(cfg: list | None = None, batch_norm: bool = False) -> nn.Sequential:
    """Build the VGG-16 feature extractor by hard-coding Table 1-D.

    ``cfg`` and ``batch_norm`` are kept only for compatibility with the starter
    skeleton. The original VGG paper uses 3x3 conv layers, ReLU, and 2x2
    max-pooling; it does not use BatchNorm in configuration D.
    """
    if cfg is not None and cfg != VGG16_CFG:
        raise ValueError("This PA2 implementation supports only VGG-16 configuration D.")
    _ = batch_norm  # 논문 기반 구현에서는 BatchNorm을 추가하지 않는다.

    return nn.Sequential(
        # 1번 블록: 224x224 -> 112x112, 채널 3 -> 64
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # 2번 블록: 112x112 -> 56x56, 채널 64 -> 128
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # 3번 블록: 56x56 -> 28x28, 채널 128 -> 256
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # 4번 블록: 28x28 -> 14x14, 채널 256 -> 512
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # 5번 블록: 14x14 -> 7x7, 채널 512 -> 512
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class VGG16(nn.Module):
    """VGG-16 configuration D with a multi-task classification head."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = make_vgg_layers(VGG16_CFG)

        # 5번의 max-pooling 뒤 224x224 입력은 7x7 feature map이 된다.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 논문은 FC-4096 -> FC-4096 -> FC-1000 구조를 사용한다.
        # 이 과제에서는 마지막 FC-1000 대신 weather/scene/timeofday head를 붙인다.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.head = MultiTaskHead(in_features=4096, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.head(x)
