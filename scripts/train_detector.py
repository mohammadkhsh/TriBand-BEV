from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("configs/triband_obb_detector_m.yaml")
    model.train(
        data="configs/kitti_triband.yaml",
        epochs=400,
        imgsz=800,
        batch=32,
        optimizer="auto",
        device=0,
        weight_decay=0.0005,
        pretrained=False,
        save=True,
        val=True,
        workers=16,
        amp=True,
        name="triband_bev_kitti",
        iou=0.7,
        patience=75,
        scale=0.1,
    )
