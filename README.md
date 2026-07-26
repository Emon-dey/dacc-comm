# DACC-Comm Code

## Environment

Packages to be installed

```bash
python -m pip install torch torchvision pytorch-tabnet pandas numpy pillow scikit-learn scipy matplotlib
```

Use `--device cuda` only on a CUDA-capable machine. Use `--device cpu` otherwise.

## Data

Controller training CSV:

```text
velocity,distance,data_rate,rssi,window_size,compression_ratio,packet_loss,latency
```


Image model training data:

```text
.png or .tif images in one directory
```

## Train Controller

The controller uses one shared multi-head `pytorch-tabnet` model:

```text
velocity,distance,data_rate,rssi -> window_size,compression_ratio
```

During training, repeated network/action rows are grouped by network condition.
The target `window_size` and `compression_ratio` are selected from the same row
with the best combined packet-loss/latency score:

```text
score = alpha * normalized_packet_loss + beta * normalized_latency
```

```bash
python scripts/dacc_comm.py train-torch-controller \
  --csv data/network/my_network_log.csv \
  --output dacc_runs/models/tabnet_controller.pt \
  --epochs 150 \
  --batch-size 64 \
  --lr 0.001 \
  --alpha 1.0 \
  --beta 1.0 \
  --device cuda
```

## Train Image Model

This trains the universal sensing matrix, ResNet18-style
encoder, and quantized encoded-feature reconstruction path.

```bash
python scripts/dacc_comm.py train-torch-image-model \
  --image-dir data/images/train \
  --output dacc_runs/models/resnet18_image_model.pt \
  --quantized-output dacc_runs/models/resnet18_image_model_ptq.pt \
  --epochs 150 \
  --batch-size 2 \
  --patch-size 256 \
  --lr 0.001 \
  --device cuda
```


## Inference

Predict adaptive controls:

```bash
python scripts/dacc_comm.py predict-torch-controller \
  --checkpoint dacc_runs/models/tabnet_controller.pt \
  --velocity 25 \
  --distance 75 \
  --data-rate 1.0 \
  --rssi -78 \
  --device cuda
```

Encode an image with the predicted compression ratio and window:

```bash
python scripts/dacc_comm.py encode-torch-image \
  --image data/images/inference/image_001.png \
  --checkpoint dacc_runs/models/resnet18_image_model_ptq.pt \
  --compression-ratio 40 \
  --predicted-window 1720 \
  --output dacc_runs/inference/image_001_packet.npz \
  --device cuda
```

The `.npz` packet contains the transmitted INT8 encoded-feature payload plus
scale, zero point, compression ratio, predicted window, and original image
shape. The receiver uses the same image-model checkpoint and dequantizes this
payload before reconstruction.

Decode/reconstruct:

```bash
python scripts/dacc_comm.py decode-torch-image \
  --packet dacc_runs/inference/image_001_packet.npz \
  --checkpoint dacc_runs/models/resnet18_image_model_ptq.pt \
  --output dacc_runs/inference/image_001_reconstructed.png \
  --device cuda
```

## Evaluate Image Quality

```bash
python scripts/dacc_comm.py eval-torch-images \
  --image-dir data/images/inference \
  --checkpoint dacc_runs/models/resnet18_image_model_ptq.pt \
  --output-dir dacc_runs/eval \
  --ratios 10 20 30 40 50 60 \
  --device cuda
```

Outputs:

```text
dacc_runs/eval/torch_image_quality_by_ratio.csv
dacc_runs/eval/torch_image_quality_table.csv
```
