# pip install ultralytics opencv-python roboflow python-dotenv lapx

import os
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Configuration from .env
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))


# Load the fine-tuned model if available, else load base nano model
fine_tuned_path = 'runs/detect/train/weights/best.pt'
yolov8n_path = 'yolov8n.pt' # Base nano model that works well out-of-the-box on CPU

if os.path.exists(fine_tuned_path):
    print(f"Loading fine-tuned weapon detection model from {fine_tuned_path}")
    model_path = fine_tuned_path
else:
    print(f"Loading base YOLOv8 nano model from {yolov8n_path}")
    model_path = yolov8n_path

model = YOLO(model_path)
print(model.names)


def open_camera(preferred_index):
    """
    Try to open a working camera using DirectShow.
    Starts with the preferred index, then iterates through 0-4.
    Warms up the camera by discarding initial black/empty frames.
    Returns an opened VideoCapture or raises SystemExit.
    """
    import numpy as np
    candidates = [preferred_index] + [i for i in range(5) if i != preferred_index]
    for idx in candidates:
        print(f"Trying camera index {idx} (CAP_DSHOW)...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  → Index {idx}: could not open.")
            cap.release()
            continue
        # Warm up: read up to 30 frames and check for non-black content
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and np.mean(frame) > 1.0:
                print(f"  → Index {idx}: OK (shape={frame.shape}). Using this camera.")
                return cap
        print(f"  → Index {idx}: opened but no usable frames (IR/virtual camera).")
        cap.release()
    print("[ERROR] No usable camera found. Check that your webcam is connected and not in use by another app.")
    raise SystemExit(1)


webcamera = open_camera(CAMERA_INDEX)

# Create window up front so it appears on top
cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Camera", cv2.WND_PROP_TOPMOST, 1)
print("Camera ready. Press 'q' inside the Live Camera window to quit.")

while True:
    success, frame = webcamera.read()

    if not success or frame is None:
        print("[WARN] Failed to grab frame, skipping...")
        continue

    results = model.track(frame, classes=0, conf=CONFIDENCE_THRESHOLD, imgsz=480)
    cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()


# For Realsense camera
   # def initialize_realsense():
    #    import pyrealsense2 as rs
    #    pipeline = rs.pipeline()
     #   camera_aconfig = rs.config()
      #  camera_aconfig.enable_stream(rs.stream.depth, *config.DEPTH_CAMERA_RESOLUTION, rs.format.z16, config.DEPTH_CAMERA_FPS)
     #   camera_aconfig.enable_stream(rs.stream.color, *config.COLOR_CAMERA_RESOLUTION, rs.format.bgr8, config.COLOR_CAMERA_FPS)
     #   pipeline.start(camera_aconfig)
      #  return pipeline
# try:
#     # Try to initialize RealSense Camera
#     camera = initialize_realsense()
#     get_frame = get_frame_realsense
# except Exception as e:
#     print("RealSense camera not found, using default webcam.")
#     camera = initialize_webcam()
#     get_frame = get_frame_webcam

# Function to get frames from RealSense
# def get_frame_realsense(pipeline):
#     import pyrealsense2 as rs
#     frames = pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
#     if not depth_frame or not color_frame:
#         return None, None
#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#     return depth_image, color_image

# # Function to get frame from webcam
# def get_frame_webcam(cap):
#     ret, frame = cap.read()
#     return None, frame if ret else None
