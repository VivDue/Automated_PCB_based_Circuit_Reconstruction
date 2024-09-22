from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO("yolov8s-seg.pt")  

# start training 
results = model.train(data="data.yaml", epochs=100, imgsz=768, workers=0)


    