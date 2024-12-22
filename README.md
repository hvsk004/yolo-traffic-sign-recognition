# YOLO Traffic Sign Recognition

## Overview
This repository implements a traffic sign recognition system using the YOLO (You Only Look Once) object detection framework. The project leverages the `ultralytics` library to fine-tune a YOLO model for detecting and classifying traffic signs based on the GTSRB dataset.

## Features
- Fine-tuned YOLO model for traffic sign recognition.
- Integration with Kaggle datasets for easy setup.
- Support for GPU-based training and inference.
- Configurable training parameters such as batch size, epochs, image size, and more.

## Dataset
The model is trained using the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) dataset, downloaded via Kaggle.

### Dataset Setup
1. Ensure you have the Kaggle API installed and set up.
2. Use the following code snippet to download the dataset:
   ```python
   import kagglehub
   from pathlib import Path

   # Download the latest version
   path = kagglehub.dataset_download("doganozcan/traffic-sign-gtrb")
   data_yaml = Path(path) / "data.yaml"
   print("Path to dataset YAML file:", data_yaml)
   ```

## Training the Model
The project uses YOLOv8 from the `ultralytics` library for training. Below is an example of how to start training:

### Training Command
```python
from ultralytics import YOLO

YOLO_MODEL = "yolo11n.pt"  # Replace with your model path
model = YOLO(YOLO_MODEL)

results = model.train(
    data=str(data_yaml),
    epochs=50,
    imgsz=640,
    device='cuda',
    batch=-1,  # Auto batch size
    amp=True,
    cache='disk',
    save=True,
    pretrained=True,
    project="./yolo-training",  # Save training outputs here
    name="yolo_traffic_sign_recognition",
    exist_ok=True,
)

print("Training Complete")
model.save("Final.pt")
```

### Key Parameters
- **`data`**: Path to the dataset YAML file.
- **`epochs`**: Number of epochs for training (default: 50).
- **`imgsz`**: Image size for resizing input images (default: 640).
- **`device`**: Device for training (`cuda` for GPU, `cpu` for CPU).
- **`batch`**: Batch size for training (set to `-1` for auto-detection).

## Results and Validation
After training, validate the model:
```python
metrics = model.val()
print("Validation Metrics:", metrics)
```

## Requirements
- Python 3.12+
- PyTorch 2.5.1+
- CUDA 12.4 (for GPU training)
- Required Python libraries:
  - `ultralytics`
  - `kagglehub`

Install the required dependencies:```bash
pip install ultralytics kagglehub
```


## Future Work
- Optimize model hyperparameters for better accuracy.
- Expand dataset with additional traffic signs from diverse regions.
- Deploy the model as a real-time inference system using Flask or FastAPI.

## Contributions
Feel free to fork this repository, submit issues, or create pull requests to improve the project.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [YOLO by Ultralytics](https://ultralytics.com/)
- [GTSRB Dataset](https://benchmark.ini.rub.de/)
- Special thanks to the Kaggle community for providing datasets and tools.


