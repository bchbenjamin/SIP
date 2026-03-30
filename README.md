# Surveillance Setup (Webcam Object Detection & Microphone Sound Event Detection)

This repository contains two deep learning intelligence projects adapted to work efficiently in real-time. 

1. **Object Detection via Webcam:** Detects common objects or can be fine-tuned to detect specific items (like weapons) using YOLOv8. 
2. **Audio Event Detection:** Analyzes real-time microphone audio to identify sound classes (e.g., Screaming, Explosions) using Google's YAMNet.

---

## 📸 Section 1: Object & Weapon Detection Using Webcam

This sub-project uses Ultralytics YOLOv8. By default, it runs the `yolov8n.pt` base model that comes pre-trained on the COCO dataset (fast and lightweight CPU performance). A script to fine-tune it with a custom dataset from Roboflow Universe is completely provided.

### Configuration & Installation

1. Navigate to the project directory:
   ```bash
   cd object-detection-using-webcam
   ```
2. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```
3. *(If not already installed)* Install the necessary dependencies:
   ```bash
   ```bash
   pip install ultralytics opencv-python roboflow python-dotenv
   ```

4. Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to customize settings like the `CAMERA_INDEX` and model configuration!*

### Running the Base (Out-Of-The-Box) Model

If you just want to run the base real-time detection without fine-tuning, simply run:
```bash
python app.py
```
This loads up the base YOLO nano model and opens your webcam feed. 

### Fine-Tuning the Model for Weapons (Knives/Guns)

1. Open `train_weapon.py`.
2. Update the `workspace_name` and `project_name` variables on line 26/27 with the exact names from the Roboflow dataset URL you choose to use.
3. Your Roboflow Private Key needs to be injected into the script. Keep it confidential.
4. Run the fine-tuning script:
   ```bash
   python train_weapon.py
   ```
5. Depending on your CPU, training for 10 epochs may take a few minutes. Once completed, the best trained weights will automatically be saved to `runs/detect/train/weights/best.pt`.
6. Our code in `app.py` has been updated to **automatically load your fine-tuned weapon weights** if they exist. To test the fine-tuned model, simply run it again:
   ```bash
   python app.py
   ```

---

## 🎤 Section 2: Real-Time Sound Event Detection (YAMNet)

This project connects to your local microphone and uses YAMNet (a deep net that predicts 521 audio event classes, including 'Screaming' as class 11).

### Troubleshooting Python Dependency Errors (Tensorflow Not Found)

> **Note:** We attempted to automatically install dependencies for this module via a new Python `venv`, but it failed stating `ERROR: No matching distribution found for tensorflow`.

**Why does this happen?**
This error indicates a Python version mismatch. Standard `tensorflow` distributions for Windows do not currently have pre-built binaries for the very latest versions of Python (e.g., Python 3.12+). 

**How to circumvent this issue:**
1. Install an older version of Python (specifically **Python 3.9, 3.10, or 3.11**) from python.org.
2. Ensure you delete the current `venv` folder in `Real-Time-Sound-Event-Detection`.
3. Create a new virtual environment specifying the older python version:
   ```bash
   py -3.10 -m venv venv
   ```
4. Activate it and retry installing the requirements:
   ```bash
   cd Real-Time-Sound-Event-Detection
   venv\Scripts\activate
   pip install tensorflow keras librosa matplotlib numpy sounddevice pandas pyaudio python-dotenv
   ```

5. Create a `.env` file for your configuration:
   ```bash
   cp .env.example .env
   ```
   *Here you can adjust params such as the microphone index `MIC_INDEX` or the specific output classes `YAMNET_CLASSES` you'd like to identify in real-time.*

### Running the Sound Event Detection

Once the environment is properly configured, execute:

```bash
python sound_event_detection.py
```
The console will display `recording...` and a plot will appear showing probability scores for tracked event classes in real-time. By default (line 16 in code), the plotter tracks "Speech", "Music", "Explosion", and "Silence," but the model natively outputs probabilities for **all 521 classes, including Screaming (Class #11)**.

---

## 🛠️ Build Steps Overview

For posterity, here is a breakdown of the specific steps taken to establish this workspace:

1. **Cloned the Repositories:**
   Ran `git clone https://github.com/codershiyar/object-detection-using-webcam.git` and `git clone https://github.com/robertanto/Real-Time-Sound-Event-Detection.git`.
2. **Setup Object Detection Venv:**
   Created a local `venv` and used `pip` to install `ultralytics`, `opencv-python`, and `roboflow`.
3. **Created Weapon Fine-Tuning Script:**
   Created a new python script `train_weapon.py` leveraging the `roboflow` Python pip package. It authorizes with the provided Roboflow API Key, pulls the YOLOv8-formatted dataset, and spawns the `YOLOv8n` dataset trainer for 10 epochs targeting the CPU. 
4. **Modified WebCam Script (`app.py`):**
   Updated `app.py` to use `os.path.exists()` logic to dynamically check if the fine-tuned weights (`runs/detect/train/weights/best.pt`) existed. If found, it routes traffic to the weapons model. Otherwise, it gracefully falls back to the generalized `yolov8n.pt` base model.
5. **Setup Audio Venv:**
   Attempted to construct a standard `venv` and deployed `pip install` on the required libraries (`tensorflow`, `pyaudio`, etc.) inside the audio project. Documented the known TensorFlow compatibility workaround utilizing an older version of Python.
