"""
Weapon Detection Fine-tuning Script

This script downloads a weapon detection dataset (e.g., knives, guns) from
Roboflow Universe and fine-tunes a pre-trained YOLOv8 Nano model on it.

Before running this script:
1. Copy .env.example to .env and fill in your Roboflow API keys.
   Your keys are available at: https://app.roboflow.com/ -> Settings -> Roboflow API
2. Update ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT in your .env file with the
   exact workspace and project names from the Roboflow dataset URL you wish to use.
   Format: universe.roboflow.com/<WORKSPACE>/<PROJECT>
3. Ensure 'roboflow', 'ultralytics', and 'python-dotenv' are installed.

The script will save the best trained weights to 'runs/detect/train/weights/best.pt',
which app.py will automatically pick up for real-time weapon detection via webcam.
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO


def load_config():
    """
    Load environment variables from the local .env file.

    Returns:
        dict: A dictionary containing the Roboflow API key, workspace,
              project name, and dataset version number.

    Raises:
        SystemExit: If any required environment variable is missing.
    """
    load_dotenv()

    required = {
        "api_key": os.getenv("ROBOFLOW_PRIVATE_KEY"),
        "workspace": os.getenv("ROBOFLOW_WORKSPACE"),
        "project": os.getenv("ROBOFLOW_PROJECT"),
        "version": os.getenv("ROBOFLOW_VERSION", "1"),
    }

    missing = [k for k, v in required.items() if not v or "your-" in str(v) or "your_" in str(v)]
    if missing:
        print(f"[ERROR] Missing or unconfigured .env variables: {missing}")
        print("  -> Copy .env.example to .env and fill in your Roboflow credentials.")
        raise SystemExit(1)

    return required


def download_dataset(config):
    """
    Authenticate with Roboflow and download the weapon detection dataset
    in YOLOv8 format.

    Args:
        config (dict): Configuration dict from load_config() containing
                       api_key, workspace, project, and version.

    Returns:
        roboflow.core.dataset.Dataset: The downloaded Roboflow dataset object,
                                       which contains the local path via .location.
    """
    print(f"Connecting to Roboflow workspace '{config['workspace']}'...")
    rf = Roboflow(api_key=config["api_key"])
    project = rf.workspace(config["workspace"]).project(config["project"])
    version = project.version(int(config["version"]))

    print(f"Downloading dataset '{config['project']}' v{config['version']} in YOLOv8 format...")
    dataset = version.download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset


def train_model(dataset):
    """
    Fine-tune the YOLOv8 Nano model on the downloaded weapon detection dataset.

    The YOLOv8 Nano ('yolov8n.pt') is the smallest and fastest variant,
    making it suitable for real-time inference on CPU hardware.

    Args:
        dataset: The Roboflow dataset object returned by download_dataset().
                 Its .location attribute is used to find the data.yaml config.

    Returns:
        ultralytics.engine.results.Results: The YOLO training results object.
    """
    print("Loading YOLOv8 Nano base model...")
    model = YOLO("yolov8n.pt")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"Starting fine-tuning using dataset config: {data_yaml}")
    print("Training on CPU — this may take several minutes per epoch.")

    results = model.train(
        data=data_yaml,
        epochs=10,        # Increase for better accuracy; 10 is a fast baseline
        imgsz=640,        # Standard YOLOv8 input resolution
        device="cpu",     # Remove this line if a CUDA GPU is available
        batch=4,          # Small batch size to reduce CPU/RAM load
        project="runs/detect",
        name="train",
        exist_ok=True,
    )

    best_weights = os.path.join("runs", "detect", "train", "weights", "best.pt")
    print(f"\nTraining complete! Best weights saved to: {best_weights}")
    print("Run 'python app.py' — it will automatically use these fine-tuned weights.")
    return results


def main():
    """Entry point: loads config, downloads dataset, and trains the model."""
    config = load_config()
    dataset = download_dataset(config)
    train_model(dataset)


if __name__ == "__main__":
    main()
