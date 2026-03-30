"""
Weapon Detection Fine-tuning Script

This script downloads a weapon detection dataset (e.g., knives, guns) from 
Roboflow Universe and fine-tunes a pre-trained YOLOv8 Nano model on it. 

Before running this script:
1. Ensure your specific dataset workspace and project names are correct.
2. Ensure you have the 'roboflow' and 'ultralytics' packages installed.

The script will save the best trained weights to 'runs/detect/train/weights/best.pt', 
which you can then use in 'app.py' for real-time inference via webcam.
"""

import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # Initialize the Roboflow client with your private API key
    # Note: Keep your API keys secure especially if pushing to a public repository
    rf = Roboflow(api_key="mCMBhfrder6POXLGvIll")
    
    print("Downloading the weapon detection dataset from Roboflow...")
    
    # IMPORTANT: 
    # Replace 'roboflow-universe-projects' and 'weapon-detection-xxx' 
    # with the exact dataset details from the Roboflow Universe dataset you wish to use.
    # E.g., rf.workspace("roboflow-universe-projects").project("weapon-detection-p5255")
    workspace_name = "your-workspace-name"
    project_name = "your-project-name"
    
    try:
        project = rf.workspace(workspace_name).project(project_name)
        version = project.version(1)  # Change version if needed
        dataset = version.download("yolov8")
        print(f"Dataset downloaded successfully to: {dataset.location}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you update the 'workspace_name' and 'project_name' variables in this script with a valid Roboflow Universe project string.")
        return

    print("Starting fine-tuning of YOLOv8 Nano model...")
    # Load the base YOLOv8 nano model (fastest model, great for real-time CPU performance)
    model = YOLO("yolov8n.pt")
    
    # Fine-tune the model using the downloaded dataset's data.yaml file
    # Adjust 'epochs' and 'imgsz' based on your available computational resources
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=10,       # Start with 10 epochs for a quick execution
        imgsz=640,       # Standard image size for YOLOv8
        device="cpu"     # Force CPU training (you can remove this if a CUDA GPU is available)
    )
    
    print(f"Training complete! The best model weights are saved at runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
