import kagglehub
from pathlib import Path
import os

# Download the latest version of the dataset from Kaggle
try:
    path = kagglehub.dataset_download("doganozcan/traffic-sign-gtrb")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit(1)

# Convert the path to a Path object
path = Path(path)

# Create a Path object for 'data.yaml'
data_yaml = path / "data.yaml"

print("Path to 'data.yaml':", data_yaml)

# Check if the file exists before proceeding
if not data_yaml.exists():
    print(f"Error: data.yaml not found at {data_yaml}")
    exit(1)

# Read and modify the data.yaml file
try:
    with open(data_yaml, "r") as file:
        data = file.read()

    # Replace paths in data.yaml to match your VM's directory structure
    data = data.replace("train: /kaggle/input/traffic-sign/train/images", "train: ./train")
    data = data.replace("val: /kaggle/input/traffic-sign/test/images", "val: ./test")
    
    # Write the modified content back to the data.yaml file
    with open(data_yaml, "w") as file:
        file.write(data)

    print("Updated data.yaml:\n", data)

except Exception as e:
    print(f"Error processing data.yaml: {e}")
    exit(1)

# Import YOLO from ultralytics after handling the dataset
from ultralytics import YOLO

# Define the path to the YOLO model (you should adjust this based on your model file)
YOLO_MODEL = "yolo11m.pt"

# Ensure the model file exists
if not os.path.exists(YOLO_MODEL):
    print(f"Error: Model file {YOLO_MODEL} not found.")
    exit(1)

# Load the YOLO model
model = YOLO(YOLO_MODEL)

# Start training the model
try:
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        device='cuda',  # Use 'cuda' if GPU is available
        batch=-1,       # Auto batch size
        amp=True,       # Automatic Mixed Precision
        cache=True,   # Cache images on disk
        save=True,      # Save weights
        pretrained=True, # Load pretrained weights
        project="./yolo-training",  # Directory for saving training results
        name="yolo_traffic_sign_recognition", # Name of the project
        exist_ok=True,  # Overwrite existing project if it exists
    )

    print("Training Complete")
    
    # Save the trained model
    model.save("Final.pt")

except Exception as e:
    print(f"Error during training: {e}")
    exit(1)
