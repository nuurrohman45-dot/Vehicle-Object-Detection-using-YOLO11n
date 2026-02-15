# Vehicle Object Detection using YOLO11n

This project implements an object detection model using YOLO11n (You Only Look Once) to detect and classify vehicles into three categories: bus, car, and truck.

## Project Overview

- **Model**: YOLO11n (nano version)
- **Dataset**: Custom vehicle dataset with 3 classes
- **Classes**: Bus, Car, Truck
- **Training Epochs**: 50
- **Image Size**: 640x640
- **Batch Size**: 4
- **Device**: CPU

## Dataset Structure

```
datasets/
├── data.yaml          # Dataset configuration
├── images/
│   ├── train/         # Training images (80 images)
│   └── val/           # Validation images (20 images)
└── labels/
    ├── train/         # Training labels
    └── val/           # Validation labels
```

## Project Files

- `object_detection.ipynb` - Main Jupyter notebook with complete workflow
- `yolo11n.pt` - Pretrained YOLO11n model weights
- `datasets/` - Dataset directory
- `runs/detect/train2/weights/best.pt` - Best trained model weights

## Requirements

- Python 3.x
- Ultralytics library
- OpenCV
- PyTorch
- Matplotlib

Install dependencies:
```
bash
pip install ultralytics opencv-python matplotlib torch
```

## Usage

### 1. Load Pretrained Model
```
python
from ultralytics import YOLO
pre_model = YOLO("yolo11n.pt")
```

### 2. Train the Model
```
python
model = YOLO("yolo11n.pt")
results = model.train(
    data=r"datasets/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu"
)
```

### 3. Validate the Model
```
python
best_model = YOLO("runs/detect/train2/weights/best.pt")
best_model.val(data=r"datasets/data.yaml")
```

### 4. Make Predictions
```
python
img_path = r"datasets\images\val\17d0db0b-val_img_5.jpg"
results = best_model.predict(source=img_path, conf=0.5, verbose=False)
```

## Results

The trained model shows improved detection performance compared to the pretrained model. The model is capable of detecting buses, cars, and trucks in images.

## Potential Improvements

1. **Data Quality**: Use higher resolution images and more diverse training data
2. **Hardware**: Use GPU for faster training
3. **Hyperparameter Tuning**: Optimize learning rate, batch size, and other parameters
4. **Model Size**: Try larger YOLO models (YOLO11m, YOLO11l) for better accuracy
5. **Data Augmentation**: Apply more data augmentation techniques

## Workflow Summary

1. **Data Preparation**: Organize images and labels in YOLO format
2. **Configuration**: Set up data.yaml with class names and paths
3. **Training**: Train YOLO11n model on the vehicle dataset
4. **Validation**: Evaluate model performance on validation set
5. **Prediction**: Use trained model for inference on new images

## Notes

- Training was performed on CPU, which takes longer compared to GPU training
- The model uses a confidence threshold of 0.5 for predictions
- Best model weights are saved in `runs/detect/train2/weights/best.pt`

## License

This project uses YOLOv11 from Ultralytics, licensed under AGPL-3.0.
