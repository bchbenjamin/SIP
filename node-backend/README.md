# AI-Powered Street Safety Device Network

Prototype implementation of the **AI-Powered Street Safety Device Network** (ASIP 24UTAI13), developed at Atria Institute of Technology, Bengaluru.

The system connects a Laptop (visual sensor node + browser dashboard) to a Raspberry Pi 5 (Edge Compute Core running YOLOv8 threat detection and two-tier response logic).

---

## Architecture Overview

```
Laptop Sensor Node          →   Pi 5 Edge Core              →   Outputs
Webcam/Mic Capture              YOLOv8 + Threat Tiering
(stream_client.py)              (edge_server.py)
WebSocket JPEG frames    →      Annotated frames + JSON   →   Dashboard (port 8766)
                                                           →   Tier 1: Siren/LED deterrence
                                                           →   Tier 2: Police/EMS alert API
```

---

## Quick Start

See **[RUN_GUIDE.md](RUN_GUIDE.md)** for the exact step-by-step commands to start and stop the system.

**Three components run in order:**
1. Raspberry Pi edge server (`edge_server.py`) — via SSH
2. Next.js dashboard (`npm run dev`) — on laptop
3. Laptop camera stream client (`stream_client.py`) — on laptop

Network config is in `.env` — update `IP_ADDRESS` if the Pi's DHCP address changes.

---

## Autonomous Edge Training

The Pi runs a 24/7 autonomous YOLOv8 training pipeline independent of this laptop:

- **Systemd autostart:** `weapon-training.service` starts training on every boot and auto-recovers from crashes
- **Dataset registry:** `dataset_registry.py` tracks all datasets persistently — sources, licenses, status, training runs
- **Dataset audit:** `dataset_audit.py` validates images, YOLO labels, class IDs, and detects duplicates
- **Power monitor:** `power_monitor.py` runs as an independent systemd service — polls `vcgencmd get_throttled` every 60s, sends ONE Telegram alert after >30 min of continuous undervoltage
- **Datasets:** Unified dataset with 3 classes: `gun`, `knife`, `dangerous_animal`
  - Weapon: Roboflow Gun+Knife Detection, Kaggle Weapon Detection
  - Animals: Zenodo NESTLER (Safari, Elephants, Foxes), Kaggle Pet Detection, TigDog
- **Checkpoint verification:** `train_autonomous.py` verifies dataset manifest and checkpoint class compatibility before resuming
- **Telegram bot:** Remote monitoring and image inference via Telegram — message the bot for a live progress update or send an image for weapon detection
- **Script:** `pi/train_autonomous.py` — handles manifest verification, checkpoint compatibility, OOM fallback, and continuous 150-epoch runs

See **[CONTEXT.md](CONTEXT/CONTEXT.md)** Sections 16–18 for full details on the training pipeline, batch processing, and Telegram integration.

---

## Repository Structure

```
Prototype/
├── pi/                     # Raspberry Pi scripts
│   ├── edge_server.py      # Main WebSocket inference server
│   ├── train_autonomous.py # Autonomous YOLOv8 training with manifest verification
│   ├── download_and_merge.py # Dataset downloader and merger (Zenodo, Kaggle, TigDog)
│   ├── dataset_registry.py # Persistent dataset tracking with atomic JSON writes
│   ├── dataset_audit.py    # Forensic audit: checksum, class ID, duplicate, split leakage
│   ├── power_monitor.py    # Independent undervoltage monitor (systemd service)
│   ├── telegram_bot.py     # Interactive Telegram bot (status + image inference)
│   ├── telegram_notify.py  # Automated progress notifications with IPv4/IPv6
│   ├── app_batch.py        # Batch video processing (frame-by-frame detection)
│   ├── sound_event_batch.py# Batch audio processing (scream detection)
│   └── config.json / requirements.txt
├── laptop/
│   └── stream_client.py    # Webcam capture → WebSocket stream to Pi
├── dashboard/              # Next.js control room dashboard
├── Samples/                # Dataset samples and research paper figure candidates
│   ├── weapon_samples/     # Raw weapon dataset images
│   ├── fig3_candidates/    # Video detection screenshots (research paper Fig. 3)
│   ├── fig4/               # Training curve (research paper Fig. 4)
│   ├── paper_candidates/   # Best images selected for paper figures
│   └── weapon-detection-1/ # Downloaded Roboflow weapon dataset
├── CONTEXT/
│   ├── CONTEXT.md          # Master project reference (single source of truth)
│   └── Automating SIP Prototype Batch Processing.md  # Raw session log
├── RUN_GUIDE.md            # Start/stop commands (authoritative)
├── RASPBERRY_PI_SETUP.md   # First-time Pi setup guide
└── THREAT_TESTING_GUIDE.md # How to demo threat detection
```

---

## Documentation

| Document | Purpose |
|---|---|
| [CONTEXT.md](CONTEXT/CONTEXT.md) | Master reference: full system design, datasets, training pipeline, Telegram bot, research paper status, team details |
| [RUN_GUIDE.md](RUN_GUIDE.md) | How to start and stop the live demo system |
| [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) | First-time Pi hardware and OS setup |
| [THREAT_TESTING_GUIDE.md](THREAT_TESTING_GUIDE.md) | How to test and demonstrate threat detection |

---

## Network Configuration

The Pi is reachable via two interfaces:

| Interface | Address | Use |
|-----------|---------|-----|
| WiFi (wlan0) | `192.168.0.125` | Normal operation |
| USB gadget (usb0) | `10.42.0.1/24` | Recovery / no-network scenarios |

Credentials are stored in `.env` (gitignored):

```env
IP_ADDRESS=192.168.0.125
USERNAME=benjamin
PASSWORD=13122006
```

Dashboard WebSocket config: `dashboard/.env.local`
```env
NEXT_PUBLIC_PI_WS_URL=ws://<PI_IP>:8766
```
