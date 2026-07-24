from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .predictor import FEATURE_COLUMNS
from .torch_pipeline import (
    decode_with_image_model,
    eval_torch_images,
    encode_with_image_model,
    export_quantized_image_model,
    predict_controller,
    train_controller,
    train_image_model,
)


def _feature_dict(args: argparse.Namespace) -> dict[str, float]:
    return {name: float(getattr(args, name)) for name in FEATURE_COLUMNS}


def cmd_network_results(args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = ["distance", "velocity", "packet_loss", "latency", "window_size", "compression_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    df.groupby("velocity")[["packet_loss", "latency"]].mean().to_csv(out_dir / "velocity_packet_loss_latency.csv")
    df.groupby("distance")[["packet_loss", "latency"]].mean().to_csv(out_dir / "distance_packet_loss_latency.csv")
    df[["window_size", "compression_ratio", "packet_loss", "latency"]].describe().to_csv(out_dir / "network_summary.csv")

    plt.figure(figsize=(7, 4))
    plt.scatter(df["window_size"], df["packet_loss"], s=18, alpha=0.7)
    plt.xlabel("Receiver window size (bytes)")
    plt.ylabel("Packet loss")
    plt.tight_layout()
    plt.savefig(out_dir / "window_vs_packet_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(df["compression_ratio"], df["latency"], s=18, alpha=0.7)
    plt.xlabel("Compression ratio (%)")
    plt.ylabel("Latency (s)")
    plt.tight_layout()
    plt.savefig(out_dir / "compression_vs_latency.png", dpi=180)
    plt.close()
    print(f"wrote network result artifacts to {out_dir}")


def cmd_train_torch_controller(args: argparse.Namespace) -> None:
    _run_torch_command(
        lambda: train_controller(
            args.csv,
            args.output,
            args.epochs,
            args.lr,
            args.batch_size,
            args.device,
            args.alpha,
            args.beta,
            args.aux_weight,
        )
    )
    print(f"wrote {args.output}")


def cmd_predict_torch_controller(args: argparse.Namespace) -> None:
    result = _run_torch_command(lambda: predict_controller(args.checkpoint, _feature_dict(args), args.device))
    print(f"window_size={result['window_size']}")
    print(f"compression_ratio={result['compression_ratio']}")


def cmd_train_torch_image_model(args: argparse.Namespace) -> None:
    _run_torch_command(
        lambda: train_image_model(
            args.image_dir,
            args.output,
            args.epochs,
            args.lr,
            args.batch_size,
            args.patch_size,
            args.device,
            args.quantized_output,
        )
    )
    print(f"wrote {args.output}")
    if args.quantized_output:
        print(f"wrote {args.quantized_output}")


def cmd_encode_torch_image(args: argparse.Namespace) -> None:
    _run_torch_command(
        lambda: encode_with_image_model(
            args.image,
            args.checkpoint,
            args.compression_ratio,
            args.output,
            args.predicted_window,
            args.device,
        )
    )
    print(f"wrote {args.output}")


def cmd_decode_torch_image(args: argparse.Namespace) -> None:
    _run_torch_command(lambda: decode_with_image_model(args.packet, args.checkpoint, args.output, args.device))
    print(f"wrote {args.output}")


def cmd_eval_torch_images(args: argparse.Namespace) -> None:
    _run_torch_command(
        lambda: eval_torch_images(
            args.image_dir,
            args.checkpoint,
            args.output_dir,
            args.ratios,
            args.predicted_window,
            args.device,
            args.max_images,
        )
    )


def cmd_export_quantized_image_model(args: argparse.Namespace) -> None:
    _run_torch_command(lambda: export_quantized_image_model(args.checkpoint, args.output, args.device))
    print(f"wrote {args.output}")


def _run_torch_command(fn):
    try:
        return fn()
    except RuntimeError as exc:
        if "PyTorch is required" in str(exc) or "requires PyTorch" in str(exc):
            raise SystemExit(str(exc)) from None
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DACC-Comm proposed-method tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("network-results")
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", default="dacc_runs/network_results")
    p.set_defaults(func=cmd_network_results)

    p = sub.add_parser("train-torch-controller")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="dacc_runs/models/tabnet_controller.pt")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--alpha", type=float, default=1.0, help="compatibility metadata; response TabNet training uses packet-loss labels directly")
    p.add_argument("--beta", type=float, default=1.0, help="compatibility metadata; response TabNet training uses latency labels directly")
    p.add_argument("--aux-weight", type=float, default=0.05, help="compatibility metadata retained in the checkpoint")
    p.set_defaults(func=cmd_train_torch_controller)

    p = sub.add_parser("predict-torch-controller")
    p.add_argument("--checkpoint", required=True)
    _add_network_args(p)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_predict_torch_controller)

    p = sub.add_parser("train-torch-image-model")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output", default="dacc_runs/models/resnet18_image_model.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quantized-output", help="PTQ INT8-weight checkpoint to export after training")
    p.set_defaults(func=cmd_train_torch_image_model)

    p = sub.add_parser("export-quantized-image-model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_export_quantized_image_model)

    p = sub.add_parser("encode-torch-image")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--compression-ratio", type=int, required=True)
    p.add_argument("--predicted-window", type=int, default=1024)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_encode_torch_image)

    p = sub.add_parser("decode-torch-image")
    p.add_argument("--packet", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_decode_torch_image)

    p = sub.add_parser("eval-torch-images")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="dacc_runs/torch_image_eval")
    p.add_argument("--ratios", nargs="+", type=int, default=[10, 20, 30, 40, 50, 60])
    p.add_argument("--predicted-window", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-images", type=int)
    p.set_defaults(func=cmd_eval_torch_images)

    return parser


def _add_network_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--velocity", type=float, default=15.0)
    parser.add_argument("--distance", type=float, default=50.0)
    parser.add_argument("--data-rate", dest="data_rate", type=float, default=1.2)
    parser.add_argument("--rssi", type=float, default=-70.0)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
