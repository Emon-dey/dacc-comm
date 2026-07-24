from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .predictor import FEATURE_COLUMNS
from .torch_models import (
    TorchDaccConfig,
    build_resnet18_encoder_refiner,
    load_torch_checkpoint,
    quantize_state_dict_int8,
    save_torch_checkpoint,
)


def _torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for dacc-torch commands.") from exc
    return torch, F, (DataLoader, Dataset)


def train_controller(
    csv_path: str | Path,
    output: str | Path,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    alpha: float = 1.0,
    beta: float = 1.0,
    aux_weight: float = 0.05,
) -> None:
    torch, F, (DataLoader, Dataset) = _torch()
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError as exc:
        raise RuntimeError(
            "pytorch-tabnet is required for train-torch-controller. "
            "Install it with `python -m pip install pytorch-tabnet` in the active environment."
        ) from exc
    df = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + ["window_size", "compression_ratio", "packet_loss", "latency"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    x_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    window = df["window_size"].to_numpy(dtype=np.float32)
    ratio_values = np.array((10, 20, 30, 40, 50, 60), dtype=np.int64)
    ratio = df["compression_ratio"].to_numpy(dtype=np.int64)
    ratio_idx = np.array([int(np.argmin(np.abs(ratio_values - r))) for r in ratio], dtype=np.int64)
    packet_loss = df["packet_loss"].to_numpy(dtype=np.float32)
    latency = df["latency"].to_numpy(dtype=np.float32)
    cfg = TorchDaccConfig()
    mean = x_raw.mean(axis=0)
    std = np.maximum(x_raw.std(axis=0), 1e-6)
    x = (x_raw - mean) / std
    window_norm = ((window - cfg.window_min) / (cfg.window_max - cfg.window_min)).clip(0.0, 1.0)
    ratio_norm = (ratio.astype(np.float32) / 100.0).clip(0.0, 1.0)
    packet_loss_mean = np.float32(packet_loss.mean())
    packet_loss_std = np.float32(max(packet_loss.std(), 1e-6))
    latency_mean = np.float32(latency.mean())
    latency_std = np.float32(max(latency.std(), 1e-6))
    packet_loss_z = (packet_loss - packet_loss_mean) / packet_loss_std
    latency_z = (latency - latency_mean) / latency_std

    loss_x = np.column_stack([x, window_norm]).astype(np.float32)
    loss_y = packet_loss_z.reshape(-1, 1).astype(np.float32)
    latency_x = np.column_stack([x, ratio_norm]).astype(np.float32)
    latency_y = latency_z.reshape(-1, 1).astype(np.float32)
    common_params = {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 4,
        "gamma": 1.3,
        "lambda_sparse": 1e-3,
        "optimizer_fn": torch.optim.AdamW,
        "optimizer_params": {"lr": lr},
        "mask_type": "sparsemax",
        "device_name": device,
        "seed": 7,
        "verbose": 0,
    }
    packet_loss_model = TabNetRegressor(**common_params)
    latency_model = TabNetRegressor(**common_params)
    packet_loss_model.fit(
        loss_x,
        loss_y,
        max_epochs=epochs,
        batch_size=batch_size,
        virtual_batch_size=max(1, min(batch_size, 32)),
        patience=0,
        drop_last=False,
    )
    latency_model.fit(
        latency_x,
        latency_y,
        max_epochs=epochs,
        batch_size=batch_size,
        virtual_batch_size=max(1, min(batch_size, 32)),
        patience=0,
        drop_last=False,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "controller_backend": "pytorch_tabnet_response",
            "packet_loss_model": packet_loss_model,
            "latency_model": latency_model,
            "feature_mean": mean,
            "feature_std": std,
            "feature_columns": FEATURE_COLUMNS,
            "ratios": ratio_values,
            "window_candidates": np.arange(cfg.window_min, cfg.window_max + 1, 10, dtype=np.int64),
            "packet_loss_mean": packet_loss_mean,
            "packet_loss_std": packet_loss_std,
            "latency_mean": latency_mean,
            "latency_std": latency_std,
            "controller_loss": "TabNet g(V,d,Dr,Rs,W)->packet_loss and h(V,d,Dr,Rs,C)->latency; inference minimizes predicted responses",
            "alpha": float(alpha),
            "beta": float(beta),
            "aux_weight": float(aux_weight),
        },
        output,
    )


def predict_controller(checkpoint: str | Path, features: dict[str, float], device: str = "cpu") -> dict[str, float]:
    torch, _, _ = _torch()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("controller_backend") != "pytorch_tabnet_response":
        raise RuntimeError("unsupported controller checkpoint; train a new checkpoint with train-torch-controller")
    cfg = TorchDaccConfig()
    mean = payload["feature_mean"]
    std = payload["feature_std"]
    ratios = payload["ratios"].astype(np.int64)
    windows = payload["window_candidates"].astype(np.int64)
    x = np.array([features[c] for c in FEATURE_COLUMNS], dtype=np.float32)
    x = (x - mean) / std
    window_norm = ((windows.astype(np.float32) - cfg.window_min) / (cfg.window_max - cfg.window_min)).clip(0.0, 1.0)
    loss_x = np.column_stack([np.repeat(x[None, :], len(windows), axis=0), window_norm])
    packet_loss_pred = payload["packet_loss_model"].predict(loss_x.astype(np.float32)).reshape(-1)
    ratio_norm = (ratios.astype(np.float32) / 100.0).clip(0.0, 1.0)
    latency_x = np.column_stack([np.repeat(x[None, :], len(ratios), axis=0), ratio_norm])
    latency_pred = payload["latency_model"].predict(latency_x.astype(np.float32)).reshape(-1)
    return {
        "window_size": int(windows[int(np.argmin(packet_loss_pred))]),
        "compression_ratio": int(ratios[int(np.argmin(latency_pred))]),
    }


def train_image_model(
    image_dir: str | Path,
    output: str | Path,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 2,
    patch_size: int = 256,
    device: str = "cpu",
    quantized_output: str | Path | None = None,
) -> None:
    torch, F, (DataLoader, Dataset) = _torch()
    image_paths = sorted(Path(image_dir).glob("*.png")) + sorted(Path(image_dir).glob("*.tif"))
    if not image_paths:
        raise ValueError(f"no images found in {image_dir}")

    class ImageDataset(Dataset):
        def __len__(self):
            return len(image_paths)

        def __getitem__(self, idx):
            arr = np.asarray(Image.open(image_paths[idx]).convert("YCbCr").split()[0], dtype=np.float32) / 255.0
            h, w = arr.shape
            arr = arr[: min(h, patch_size), : min(w, patch_size)]
            padded = np.zeros((patch_size, patch_size), dtype=np.float32)
            padded[: arr.shape[0], : arr.shape[1]] = arr
            return torch.from_numpy(padded).unsqueeze(0)

    model = build_resnet18_encoder_refiner(TorchDaccConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(ImageDataset(), batch_size=batch_size, shuffle=True)
    ratios = (10, 20, 30, 40, 50, 60)

    for epoch in range(epochs):
        model.train()
        ratio = ratios[epoch % len(ratios)]
        for xb in loader:
            xb = xb.to(device)
            rec, _ = model(xb, ratio)
            features = model.encode_features(xb, ratio)
            features_tx = _select_feature_channels(features, ratio)
            features_qdq, _, _ = _fake_int8_affine_quant_dequant(features_tx)
            features_full = _restore_feature_channels(features_qdq, features.shape[1])
            rec_q = model.reconstruct_from_features(features_full, tuple(xb.shape[-2:]))
            loss = F.mse_loss(rec_q, xb) + 0.25 * F.mse_loss(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    extra = {
        "ratios": np.array(ratios),
        "image_training_loss": "MSE(reconstruct(dequantize(INT8(Fconv))), image)+0.25*MSE(reconstruct(Fconv), image)",
        "feature_quantization": "straight_through_affine_int8_encoded_features",
        "sensing": "OPINE-style universal 33x33 block matrix with nested row subsets before ResNet18 feature extraction",
    }
    save_torch_checkpoint(model, output, extra)
    if quantized_output:
        export_quantized_image_model(output, quantized_output, device=device)


def encode_with_image_model(
    image_path: str | Path,
    checkpoint: str | Path,
    compression_ratio: int,
    output: str | Path,
    predicted_window: int = 1024,
    device: str = "cpu",
) -> None:
    torch, _, _ = _torch()
    model = build_resnet18_encoder_refiner(TorchDaccConfig()).to(device)
    load_torch_checkpoint(model, checkpoint, device=device)
    arr = np.asarray(Image.open(image_path).convert("YCbCr").split()[0], dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        features = model.encode_features(x, compression_ratio)
        total_feature_channels = features.shape[1]
        features = _select_feature_channels(features, compression_ratio)
        payload = features.cpu().numpy()
    payload_q, scale, zero_point = _quantize_int8_affine_np(payload)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        payload_q=payload_q,
        scale=np.array(scale, dtype=np.float32),
        zero_point=np.array(zero_point, dtype=np.int32),
        compression_ratio=np.array(compression_ratio, dtype=np.int32),
        predicted_window=np.array(predicted_window, dtype=np.int32),
        original_shape=np.array(arr.shape, dtype=np.int32),
        feature_channels=np.array(total_feature_channels, dtype=np.int32),
        retained_feature_channels=np.array(payload.shape[1], dtype=np.int32),
        image_name=np.array(Path(image_path).name),
        payload_kind=np.array("encoded_features_int8"),
        model_type=np.array("opine_style_resnet18_feature_ptq"),
    )


def decode_with_image_model(packet_path: str | Path, checkpoint: str | Path, output: str | Path, device: str = "cpu") -> None:
    torch, _, _ = _torch()
    model = build_resnet18_encoder_refiner(TorchDaccConfig()).to(device)
    load_torch_checkpoint(model, checkpoint, device=device)
    packet = np.load(packet_path, allow_pickle=False)
    features = (packet["payload_q"].astype(np.float32) - int(packet["zero_point"])) * float(packet["scale"])
    features_t = torch.from_numpy(features).to(device)
    original_shape = tuple(map(int, packet["original_shape"]))
    model.eval()
    with torch.no_grad():
        payload_kind = str(packet["payload_kind"]) if "payload_kind" in packet else "measurements_int8"
        if payload_kind == "encoded_features_int8":
            if "feature_channels" in packet:
                features_t = _restore_feature_channels(features_t, int(packet["feature_channels"]))
            rec = model.reconstruct_from_features(features_t, original_shape).squeeze().cpu().numpy()
        else:
            rec = model.reconstruct_from_payload(features_t, original_shape).squeeze().cpu().numpy()
    arr = np.clip(rec * 255.0, 0, 255).astype(np.uint8)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output)


def eval_torch_images(
    image_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    ratios: list[int],
    predicted_window: int = 1024,
    device: str = "cpu",
    max_images: int | None = None,
) -> None:
    image_paths = sorted(Path(image_dir).glob("*.png")) + sorted(Path(image_dir).glob("*.tif"))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for ratio in ratios:
        packet_dir = out_root / f"packets_ratio_{ratio}"
        recon_dir = out_root / f"recon_ratio_{ratio}"
        packet_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            try:
                packet_path = packet_dir / f"{image_path.stem}.dacc.torch.npz"
                recon_path = recon_dir / f"{image_path.stem}_dacc_torch_rec.png"
                encode_with_image_model(image_path, checkpoint, ratio, packet_path, predicted_window, device)
                decode_with_image_model(packet_path, checkpoint, recon_path, device)
                ref = np.asarray(Image.open(image_path).convert("YCbCr").split()[0], dtype=np.float32) / 255.0
                rec = np.asarray(Image.open(recon_path), dtype=np.float32) / 255.0
                rows.append(
                    {
                        "image": image_path.name,
                        "compression_ratio": ratio,
                        "psnr": _psnr(rec, ref),
                        "ssim": _simple_ssim(rec, ref),
                        "payload_bytes": packet_path.stat().st_size,
                        "reconstruction": str(recon_path),
                    }
                )
            except Exception as exc:
                print(f"skipping {image_path}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "torch_image_quality_by_ratio.csv", index=False)
    if df.empty:
        raise RuntimeError("no torch image results were generated; check image paths and checkpoint")
    grouped = df.groupby("compression_ratio")[["psnr", "ssim", "payload_bytes"]].mean().reset_index()
    grouped.to_csv(out_root / "torch_image_quality_table.csv", index=False)
    print(grouped.to_string(index=False))


def export_quantized_image_model(
    checkpoint: str | Path,
    output: str | Path,
    device: str = "cpu",
) -> None:
    torch, _, _ = _torch()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("state_dict", payload)
    quantized_state, quant_meta = quantize_state_dict_int8(state)
    out_payload = {
        "quantized_state_dict": quantized_state,
        "weight_quantization": quant_meta,
        "source_checkpoint": str(checkpoint),
        "ratios": payload.get("ratios", np.array((10, 20, 30, 40, 50, 60))),
        "feature_quantization": payload.get("feature_quantization", "symmetric_int8_payload"),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output)


def _fake_int8_quant_dequant(tensor):
    torch, _, _ = _torch()
    max_abs = torch.amax(torch.abs(tensor), dim=tuple(range(1, tensor.ndim)), keepdim=True).clamp_min(1e-8)
    scale = max_abs / 127.0
    quantized = torch.round(tensor / scale).clamp(-128, 127)
    dequantized = quantized * scale
    return tensor + (dequantized - tensor).detach(), scale, quantized


def _fake_int8_affine_quant_dequant(tensor):
    torch, _, _ = _torch()
    reduce_dims = tuple(range(1, tensor.ndim))
    min_val = torch.amin(tensor, dim=reduce_dims, keepdim=True)
    max_val = torch.amax(tensor, dim=reduce_dims, keepdim=True)
    scale = ((max_val - min_val) / 255.0).clamp_min(1e-8)
    zero_point = torch.round(-128.0 - min_val / scale).clamp(-128, 127)
    quantized = torch.round(tensor / scale + zero_point).clamp(-128, 127)
    dequantized = (quantized - zero_point) * scale
    return tensor + (dequantized - tensor).detach(), scale, quantized


def _quantize_int8_affine_np(array: np.ndarray) -> tuple[np.ndarray, float, int]:
    min_val = float(np.min(array))
    max_val = float(np.max(array))
    scale = max((max_val - min_val) / 255.0, 1e-8)
    zero_point = int(np.clip(round(-128.0 - min_val / scale), -128, 127))
    quantized = np.round(array / scale + zero_point).clip(-128, 127).astype(np.int8)
    return quantized, scale, zero_point


def _feature_channel_count(total_channels: int, compression_ratio: int) -> int:
    retained = max(1.0 - float(compression_ratio) / 100.0, 0.01)
    return max(1, min(total_channels, int(round(total_channels * retained))))


def _select_feature_channels(features, compression_ratio: int):
    channels = _feature_channel_count(features.shape[1], compression_ratio)
    return features[:, :channels]


def _restore_feature_channels(features, total_channels: int):
    if features.shape[1] >= total_channels:
        return features[:, :total_channels]
    torch, _, _ = _torch()
    pad_shape = list(features.shape)
    pad_shape[1] = total_channels - features.shape[1]
    padding = torch.zeros(*pad_shape, dtype=features.dtype, device=features.device)
    return torch.cat([features, padding], dim=1)


def _psnr(rec: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((rec - ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _simple_ssim(rec: np.ndarray, ref: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = float(rec.mean())
    mu_y = float(ref.mean())
    var_x = float(rec.var())
    var_y = float(ref.var())
    cov = float(((rec - mu_x) * (ref - mu_y)).mean())
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)))
