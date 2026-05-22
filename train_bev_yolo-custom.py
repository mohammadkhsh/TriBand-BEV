# train_bev_yolo-custom.py
from ultralytics import YOLO
import sys
import os


# Make sure multiprocessing is handled properly on Windows
if __name__ == '__main__':
    # Load the new 4-channel model configuration
    model = YOLO("ds-yolo-3obj-3h2-lowZwideJitterAug-amp\yolo12-obb-modified-m.yaml")

    # Train the model
    results = model.train(
        data='ds-yolo-3obj-3h2-lowZwideJitterAug-amp/data.yaml',
        epochs=400,
        imgsz=800,
        batch=32,
        optimizer='auto',
        device=0,
        weight_decay=0.0005,
        pretrained=False, # Set to False since pre-trained weights are 3-channel
        save=True,
        val=True,
        workers=16,
        amp=True,
        name='TEST_ds-yolo-3obj-3h2-lowZwideJitterAug-amp',
        iou=0.7,
        patience=75,
        scale=0.1
        )