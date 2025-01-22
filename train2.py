API = "IZiSYgNQKAQpV47qj08K"
from roboflow import Roboflow
rf = Roboflow(api_key=API)
project = rf.workspace("mohamed-traore-2ekkp").project("gtsdb---german-traffic-sign-detection-benchmark")
version = project.version(3)
dataset = version.download("yolov11")

from pathlib import Path
data_yaml = Path("GTSDB---German-Traffic-Sign-Detection-Benchmark-3/data.yaml")
print(data_yaml)

YOLO_MODEL = "yolov8n.pt"
from ultralytics import YOLO
model = YOLO(YOLO_MODEL)

results = model.train(
        data=str(data_yaml),
        epochs=300,
        patience=30,
        imgsz=640,
        device='cuda',  # Use 'cuda' if GPU is available
        batch=-1,       # Auto batch size
        amp=True,       # Automatic Mixed Precision
        cache='disk',   # Cache images on disk
        save=True,      # Save weights
        pretrained=True, # Load pretrained weights
        project="./yolo-training-gtsdb2",  # Directory for saving training results
        name="yolo_traffic_sign_recognition", # Name of the project
        exist_ok=True,  # Overwrite existing project if it exists
    )