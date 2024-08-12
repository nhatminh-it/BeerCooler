import numpy as np
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
import streamlit as st
from pathlib import Path

def get_obj_det_model(model_name='yolov8n.pt', local=False):
    # Load YOLOv8 model from Ultralytics
    if local:
        torch.hub.set_dir('.')  # Set directory for local cache if needed
    model = YOLO(model_name)  # You can specify different model variants like 'yolov8n.pt', 'yolov8s.pt', etc.
    return model

def crop_beers(image, model, threshold, GPU=True):
    boxes, classes, labels, preds = find_bottles_and_cans(image, model, detection_threshold=threshold, GPU=GPU)
    if len(boxes) > 0:
        image_cropped = image.crop(tuple(boxes[0]))  # Crop image: select only relevant part of pic
    else:
        image_cropped = image
    return image_cropped, len(boxes)

def find_bottles_and_cans(image, model, detection_threshold=0.3, GPU=True):
    # Convert PIL Image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    device = torch.device('cuda' if GPU and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Run YOLOv8 inference
    results = model(image, conf=detection_threshold, device=device)

    # Process results
    boxes = []
    classes = []
    labels = []
    preds = []

    for r in results:
        for box in r.boxes:
            if box.cls in [39, 41]:  # 39 is the class index for 'bottle', 41 is for 'can'
                if box.cls == 39:
                    classes.append('bottle')
                elif box.cls == 41:
                    classes.append('can')

                boxes.append(box.xyxy[0].cpu().numpy().astype(np.int32))
                labels.append(box.cls.cpu())
                preds.append(box.conf.cpu().numpy())

    return np.array(boxes), classes, labels, np.array(preds)

@st.cache_resource
def get_obj_det_model_DirectDownload(model_name='yolov8n.pt'):
    # This function directly downloads the model if it's not present
    print('Loading or downloading the object detection model')
    model = YOLO(model_name)  # This will download the model if it's not present
    return model
