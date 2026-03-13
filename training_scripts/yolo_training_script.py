from ultralytics import YOLO
import os

data_path = "C:/Users/medbe/OneDrive/Bureau/PFA2026/yolo_dataset"
yaml_file = os.path.join(data_path, "dataset.yaml")
model = YOLO("yolo26m.pt")
# if you have i5 14th generation or more 6 workers is good for training,
# if you have less than that you can set it to 0 or 2
model.train(data=yaml_file, epochs=100, batch=8, imgsz=640, device="cuda", workers=6)
