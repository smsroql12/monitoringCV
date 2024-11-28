from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load an official model

# Export the model
model.export(format="onnx", opset=12, imgsz=(480,640))