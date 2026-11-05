# SIP Guardian — Recovered Model Checkpoints & Training Artifacts

## 1. Overview

This directory preserves the YOLOv8 weapon detection model checkpoints trained on Raspberry Pi 5 (`raspberrypi`, `192.168.0.125`). These weights were recovered during emergency operations to prevent data loss from SD card deterioration and background service overwrite risks.

A secondary mirror is archived outside the Git repository at `/home/bchbenjamin/.sip_checkpoints_backup/`.

## 2. Model & Dataset Specifications

- **Architecture:** Ultralytics YOLOv8n (Nano)
- **Input Resolution:** `416 x 416`
- **Classes (`nc: 2`):**
  - `0: gun`
  - `1: knife`
- **Training Hyperparameters:**
  - `batch: 4`
  - `epochs: 150`
  - `patience: 0` (no early stop, full run)
  - `optimizer: auto`
  - `device: cpu`

## 3. Checkpoint Manifest & Verification

| Filename | Description | Epoch | Size (Bytes) | SHA-256 Checksum | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| `epoch80_best_map.pt` | **Peak Performance Checkpoint** | 80 | 12,330,111 | `ac06e9281a7f20208549e739b9b1dd4f20e426b209d12027ac9122d920c80c90` | **0.8156** | **0.6092** |
| `epoch145.pt` | **Canonical Pre-Restart Checkpoint** | 145 | 12,337,855 | `965aac342859795632bd7be872a3933bdf7737bd03f6ff2740072cc38b9c8b63` | 0.7852 | 0.5837 |
| `last_backup.pt` | **Pruned Deployment Weight (150 ep)** | 150 | 6,243,370 | `b58bcfaf27affe711ed22d3a5e406653980d10ff7294d265741f3fefedcf4772` | 0.7858 | 0.5839 |
| `epoch140.pt` | Converged Checkpoint | 140 | 12,337,279 | `832e3184654a4ce7f661cd715ad262052063818ec433872f296224bb7f129e87` | 0.7851 | 0.5826 |
| `best_restarted_epoch2.pt` | Restarted Run Best | 2 | 12,342,527 | `06f257719af4b3f4166d01e84d3d4c4c19ea57710975d746bfb22bdad9155578` | N/A | N/A |
| `last_restarted_epoch2.pt` | Restarted Run Last | 2 | 12,342,527 | `06f257719af4b3f4166d01e84d3d4c4c19ea57710975d746bfb22bdad9155578` | N/A | N/A |

## 4. Associated Training Artifacts

The `manifests/` subfolder includes:
- `args.yaml`: Complete training argument configuration used during the 150-epoch run.
- `results.csv`: Epoch-by-epoch loss, precision, recall, and mAP metrics.
- `results.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`: Visual evaluation curves and matrices.
- `P_curve.png`, `R_curve.png`, `PR_curve.png`, `F1_curve.png`: Model confidence thresholds.
- `train_batch*.jpg`, `val_batch*.jpg`: Visual ground truth vs model prediction batches.

## 5. Usage in Edge & Backend Deployments

For low-memory and edge deployment on Raspberry Pi or Node inference backend:
```python
from ultralytics import YOLO

# Load peak performance checkpoint
model = YOLO("artifacts/model-checkpoints/recovered/epoch80_best_map.pt")

# Or load pruned lightweight deployment weights (6.2 MB)
model_edge = YOLO("artifacts/model-checkpoints/recovered/last_backup.pt")

results = model.predict(source=0, conf=0.45, imgsz=416)
```
