from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # load an official model

results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
