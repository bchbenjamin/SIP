# Researched Data

## GitHub Repositories for Laptop Prototyping [03-04-2026]

### Camera (Visual Detection)

| Repo | Why it's useful |
|---|---|
| [codershiyar/object-detection-using-webcam](https://github.com/codershiyar/object-detection-using-webcam) | YOLOv8 + OpenCV on webcam, adjustable confidence threshold, snapshot saving. Best starting point. |
| [tooshiNoko/Real-Time-Object-Detection-with-YOLOv8-and-OpenCV](https://github.com/tooshiNoko/Real-Time-Object-Detection-with-YOLOv8-and-OpenCV) | Specifically highlights knives with red bounding boxes — directly relevant to your threat detection use case |
| [RizwanMunawar/yolov8-object-tracking](https://github.com/RizwanMunawar/yolov8-object-tracking) | Adds tracking on top of detection — important for following a suspect across frames |
| [anugraheeth/Real-Time-Object-Detection-with-YOLOv8-and-Audio-Feedback](https://github.com/anugraheeth/Real-Time-Object-Detection-with-YOLOv8-and-Audio-Feedback) | YOLOv8 + audio alerts on detection — this is very close to your actual system behavior |
| [JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8) | Purpose-built weapon + knife detector with YOLOv8, uses a Roboflow dataset |

---

### Microphone (Audio / Scream Detection)

| Repo | Why it's useful |
|---|---|
| [robertanto/Real-Time-Sound-Event-Detection](https://github.com/robertanto/Real-Time-Sound-Event-Detection) | Live microphone → YAMNet model → classifies 521 sound events including Screaming (class 11), Yell, Shout, Gunshot. Plug and play. |
| [plaffitte/scream-detection](https://github.com/plaffitte/scream-detection) | Academic thesis repo specifically on scream/shout detection using neural networks — directly matches your use case |
| [tyiannak/pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) | Full audio analysis library — MFCC extraction, real-time classification, segmentation. The backbone for building your own audio classifier |
| [daisukelab/ml-sound-classifier](https://github.com/daisukelab/ml-sound-classifier) | Real-time mic classification using Keras/TensorFlow, even has a Raspberry Pi-compatible lighter model |

---

## Datasets — All Aspects of the Project

### Visual Threat Detection

| Dataset | What it covers | Link |
|---|---|---|
| **UCF-Crime** | 1,900 CCTV surveillance videos across 13 crime classes — Robbery, Fighting, Assault, Vandalism, Burglary, Shooting, Road Accident, and more. The gold standard dataset for your project. | [Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) |
| **UCF-Crime Annotation (UCA)** | Frame-level annotations on top of UCF-Crime — better for training | [Kaggle](https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset) |
| **Real-Time CCTV Anomaly Detection Dataset** | Curated for direct CCTV anomaly detection tasks | [Kaggle](https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance) |
| **CamNuvem** | 486 real-world robbery surveillance videos — more robbery-specific than UCF | [PubMed paper with download link](https://pubmed.ncbi.nlm.nih.gov/36560385/) |

---

### Weapon Detection

| Dataset | What it covers | Link |
|---|---|---|
| **Gun + Knife Detection (Mahad Ahmed)** | 8,451 labeled images of guns and knives | [Roboflow](https://universe.roboflow.com/mahad-ahmed/gun-and-knife-detection) |
| **CCTV Knife Detection (Simuletic)** | Synthetic CCTV knife detection dataset, CC BY 4.0 license | [Roboflow](https://universe.roboflow.com/simuletic/cctv-knife-detection-dataset-zkkaf) / [Kaggle](https://www.kaggle.com/datasets/simuletic/cctv-knife-detection-dataset) |
| **Weapon Detection CCTV v3** | 11 classes including knives, pistols, rifles, other handheld threats | [Roboflow](https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset) |
| **Weapon Detection (yolov7test)** | 9,672 weapon images with pre-trained model | [Roboflow](https://universe.roboflow.com/yolov7test-u13vc/weapon-detection-m7qso) |
| **Weapons Dataset (Knife/Grenade/Gun/Pistol)** | 1,362 images across 4 threat categories | [Roboflow](https://universe.roboflow.com/weapons-dataset/weapons-dataset-os1ki) |

---

### Audio / Scream Detection

| Dataset | What it covers | Link |
|---|---|---|
| **Human Screaming Detection Dataset** | Audio classification for emergency detection, scream vs non-scream | [Kaggle](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset) |
| **Audio Dataset of Scream and Non-Scream** | Binary scream classification dataset | [Kaggle](https://www.kaggle.com/datasets/aananehsansiam/audio-dataset-of-scream-and-non-scream) |
| **Scream Dataset (thesis)** | Various human scream recordings | [Kaggle](https://www.kaggle.com/datasets/sanzidaakterarusha/scream-dataset) |
| **Google AudioSet — Screaming class** | Massive dataset, screaming is class entry — labeled 10-second YouTube clips | [Google AudioSet](https://research.google.com/audioset/dataset/screaming.html) |
| **IUEC Database (IIIT-Delhi)** | Distress sounds (scream + cry) in urban environments — indoor, outdoor, crowd, machinery contexts. Most realistic for your deployment scenario. | [IIIT-Delhi](https://www.iiitd.edu.in/~anils/taslp/taslp_distress_detection.html) |
| **UrbanSound8K** | 8,732 urban sound clips across 10 classes including street sounds — good for background noise training | [Search on Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) |
| **ESC-50** | 2,000 environmental audio clips — 50 classes, widely used for ambient/non-threat sound classification | [GitHub](https://github.com/karolpiczak/ESC-50) |

---

### Wildlife / Animal Detection

| Dataset | What it covers | Link |
|---|---|---|
| **iWildCam (Kaggle competition)** | Camera trap wildlife images, multiple species, real conditions | [Kaggle](https://www.kaggle.com/c/iwildcam-2019-fgvc6) |
| **WCS Camera Traps** | 1.4M camera trap images, 675 species from 12 countries | [LILA BC](https://lila.science/datasets/wcscameratraps) |
| **Object Detection Wildlife (YOLO format)** | Wildlife detection in YOLO-ready format for direct training | [Kaggle](https://www.kaggle.com/datasets/ankanghosh651/object-detection-wildlife-dataset-yolo-format) |
| **Animals Detection Images Dataset** | General animal detection dataset | [Kaggle](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset) |
| **Google SpeciesNet / MegaDetector** | Google's pre-trained model for detecting animals, humans, vehicles in camera trap images — you can use this directly | [GitHub](https://github.com/google/cameratrapai) |

---

## Where to Start Right Now

For your **laptop prototype this week**, the cleanest path is:

**Camera:** Clone [codershiyar/object-detection-using-webcam](https://github.com/codershiyar/object-detection-using-webcam), run it as-is with `yolov8n.pt` (the nano model, fast on CPU), then swap in the weapon detection dataset from Roboflow to fine-tune it for knives/guns.

**Microphone:** Clone [robertanto/Real-Time-Sound-Event-Detection](https://github.com/robertanto/Real-Time-Sound-Event-Detection) — it already detects Screaming (class 11) out of the box with no training needed, using Google's YAMNet model on your laptop mic.

Both run on CPU, no GPU required.