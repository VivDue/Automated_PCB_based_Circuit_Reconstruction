from ultralytics import YOLO
import glob
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


files = glob.glob("data/*")

# Load a model
model = YOLO("yolov8n-seg.pt")  #load standard yolo seg model

# Run batched inference on a list of images
results = model("https://ultralytics.com/images/bus.jpg", save=True)  # return a list of Results objects

"""# Process results list
for id, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"res/result{id}.jpg")  # save to disk"""
    