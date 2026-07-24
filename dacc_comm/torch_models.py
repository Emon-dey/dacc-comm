from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "The DACC torch path requires PyTorch. Install torch in the active environment."
        ) from exc
    return torch, nn, F


@dataclass(frozen=True)
class TorchDaccConfig:
    max_height: int = 1080
    max_width: int = 1920
    block_size: int = 33
    max_measurements: int = 1089
    latent_channels: int = 512
    controller_hidden: int = 64
    controller_steps: int = 4
    window_min: int = 256
    window_max: int = 2560
    ratios: tuple[int, ...] = (10, 20, 30, 40, 50, 60)


def build_resnet18_encoder_refiner(config: TorchDaccConfig | None = None):

    torch, nn, F = _torch()
    cfg = config or TorchDaccConfig()

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            out = F.relu(out + self.shortcut(x), inplace=True)
            return out

    class ResNet18Backbone(nn.Module):
        def __init__(self, in_channels: int = 1, base: int = 64) -> None:
            super().__init__()
            self.in_planes = base
            self.conv1 = nn.Conv2d(in_channels, base, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(base)
            self.layer1 = self._make_layer(BasicBlock, base, 2, stride=1)
            self.layer2 = self._make_layer(BasicBlock, base * 2, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, base * 4, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, cfg.latent_channels, 2, stride=2)

        def _make_layer(self, block, planes: int, blocks: int, stride: int):
            layers = [block(self.in_planes, planes, stride)]
            self.in_planes = planes
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return self.layer4(x)

    class DaccImageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_pixels = cfg.block_size * cfg.block_size
            if cfg.max_measurements > self.block_pixels:
                raise ValueError("max_measurements cannot exceed block_size**2")
            self.sensing = nn.Parameter(torch.empty(cfg.max_measurements, self.block_pixels))
            nn.init.xavier_normal_(self.sensing)
            self.sensing_scale = nn.Parameter(torch.tensor(0.01))
            self.encoder = ResNet18Backbone(in_channels=1)
            self.refiner = nn.Sequential(
                nn.Conv2d(cfg.latent_channels, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def _measurement_count(self, compression_ratio: int) -> int:
            retained = max(1.0 - float(compression_ratio) / 100.0, 0.01)
            return max(1, min(cfg.max_measurements, int(round(cfg.max_measurements * retained))))

        def _pad_to_blocks(self, x):
            h, w = x.shape[-2:]
            pad_h = (cfg.block_size - h % cfg.block_size) % cfg.block_size
            pad_w = (cfg.block_size - w % cfg.block_size) % cfg.block_size
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            return x, (h, w)

        def sense(self, x, compression_ratio: int):
            x_pad, original_shape = self._pad_to_blocks(x)
            m = self._measurement_count(compression_ratio)
            phi = torch.tanh(self.sensing[:m]) * self.sensing_scale
            blocks = F.unfold(x_pad, kernel_size=cfg.block_size, stride=cfg.block_size)
            measurements = torch.einsum("mp,bpl->bml", phi, blocks)
            h_blocks = x_pad.shape[-2] // cfg.block_size
            w_blocks = x_pad.shape[-1] // cfg.block_size
            measurements = measurements.reshape(x.shape[0], m, h_blocks, w_blocks)
            return measurements, phi, original_shape, x_pad.shape[-2:]

        def initialize_from_measurements(self, measurements, phi, padded_shape: tuple[int, int]):
            b, m, h_blocks, w_blocks = measurements.shape
            measurements_flat = measurements.reshape(b, m, h_blocks * w_blocks)
            blocks = torch.einsum("mp,bml->bpl", phi, measurements_flat)
            return F.fold(blocks, output_size=padded_shape, kernel_size=cfg.block_size, stride=cfg.block_size)

        def encode_features(self, x, compression_ratio: int):
            measurements, phi, _, padded_shape = self.sense(x, compression_ratio)
            x_est = self.initialize_from_measurements(measurements, phi, padded_shape)
            return self.encoder(x_est)

        def encode_payload(self, x, compression_ratio: int):
            measurements, _, original_shape, padded_shape = self.sense(x, compression_ratio)
            return measurements, original_shape, padded_shape

        def reconstruct_from_payload(self, measurements, original_shape: tuple[int, int]):
            m = measurements.shape[1]
            phi = torch.tanh(self.sensing[:m]) * self.sensing_scale
            padded_h = measurements.shape[-2] * cfg.block_size
            padded_w = measurements.shape[-1] * cfg.block_size
            x_est = self.initialize_from_measurements(measurements, phi, (padded_h, padded_w))
            features = self.encoder(x_est)
            return self.reconstruct_from_features(features, original_shape)

        def reconstruct_from_features(self, features, original_shape: tuple[int, int]):
            rec_small = self.refiner(features)
            return F.interpolate(rec_small, size=original_shape, mode="bilinear", align_corners=False)

        def forward(self, x, compression_ratio: int):
            measurements, phi, original_shape, padded_shape = self.sense(x, compression_ratio)
            x_est = self.initialize_from_measurements(measurements, phi, padded_shape)
            features = self.encoder(x_est)
            rec = self.reconstruct_from_features(features, original_shape)
            return rec, features

    return DaccImageModel()


def save_torch_checkpoint(model, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    torch, _, _ = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_torch_checkpoint(model, path: str | Path, device: str = "cpu"):
    torch, _, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    if "quantized_state_dict" in payload:
        state = dequantize_state_dict(payload["quantized_state_dict"], payload["weight_quantization"])
    else:
        state = payload.get("state_dict", payload)
    model.load_state_dict(state)
    return model


def quantize_state_dict_int8(state_dict: dict[str, Any]):

    torch, _, _ = _torch()
    quantized = {}
    metadata = {"scheme": "per_tensor_symmetric_int8_weights"}
    for name, tensor in state_dict.items():
        if torch.is_floating_point(tensor) and tensor.numel() > 0:
            max_abs = torch.max(torch.abs(tensor)).item()
            scale = max(max_abs / 127.0, 1e-8)
            quantized[name] = {
                "q": torch.round(tensor.detach().cpu() / scale).clamp(-128, 127).to(torch.int8),
                "scale": float(scale),
                "dtype": str(tensor.dtype),
            }
        else:
            quantized[name] = {"value": tensor.detach().cpu()}
    return quantized, metadata


def dequantize_state_dict(quantized_state: dict[str, Any], metadata: dict[str, Any] | None = None):
    torch, _, _ = _torch()
    state = {}
    for name, item in quantized_state.items():
        if isinstance(item, dict) and "q" in item:
            state[name] = item["q"].to(torch.float32) * float(item["scale"])
        elif isinstance(item, dict) and "value" in item:
            state[name] = item["value"]
        else:
            state[name] = item
    return state
