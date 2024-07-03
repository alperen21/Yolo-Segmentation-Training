from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # load an official model

results = model.train(data="train.yaml", epochs=1, imgsz=640)
