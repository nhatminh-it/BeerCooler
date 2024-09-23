import os
import torch
from PIL import Image
import yaml
from ultralytics import YOLO
from typing import Union, List, Tuple
import requests

class BeerDetector:
    def __init__(self, object_detection_config: dict, device: torch.device):
        self.device = device
        self.yolo_settings = object_detection_config['settings']
        model_config = object_detection_config['model']

        # Load YOLOv8 model
        model_path = model_config['local_model_dir']
        if model_path and os.path.exists(model_path):
            print(f"Loading YOLOv8 model from local path: {model_path}")
            self.yolo_model = YOLO(model_path)
        else:
            raise FileNotFoundError(f"YOLOv8 model not found at {model_path}")

        self.yolo_model.fuse()
        self.yolo_model.to(self.device)

        # Load data_config to get class names
        data_config_path = model_config.get('data_config', None)
        if data_config_path and os.path.exists(data_config_path):
            with open(data_config_path, 'r') as file:
                data_config = yaml.safe_load(file)
                self.all_class_names = data_config.get('names', [])
                print(f"All class names from data config: {self.all_class_names}")
        else:
            raise FileNotFoundError(f"Data config file not found at {data_config_path}")

        # Set allowed labels to only 'bottle' and 'can'
        self.allowed_labels = ['bottle', 'can']
        print(f"Allowed labels for processing: {self.allowed_labels}")

    def find_objects(self, image: Image.Image) -> dict:
        # Perform inference
        results = self.yolo_model.predict(
            image,
            conf=self.yolo_settings.get('conf', 0.25),
            iou=self.yolo_settings.get('iou', 0.45),
            agnostic_nms=self.yolo_settings.get('agnostic_nms', False),
            max_det=self.yolo_settings.get('max_det', 1000),
            device=str(self.device)
        )

        # Process YOLOv8 output
        bboxes = []
        labels = []
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                bbox = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]

                # Extract class label
                cls_id = int(box.cls.cpu().numpy()[0])
                label = result.names[cls_id]

                # Filter for allowed labels
                if label in self.allowed_labels:
                    bboxes.append(bbox)
                    labels.append(label)
                else:
                    print(f"Ignoring detected object with label '{label}'")

        parsed_results = {
            "<OD>": {
                "bboxes": bboxes,
                "labels": labels
            }
        }
        return parsed_results

    def process_image(self, image_source: Union[str, Image.Image]):
        # Load the image
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
            else:
                image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        else:
            raise ValueError("Invalid image source. Must be a URL, file path, or PIL Image.")

        # Run the object detection
        results = self.find_objects(image)
        bboxes = results['<OD>']['bboxes']
        labels = results['<OD>']['labels']

        # Crop detected objects
        detected_images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = image.crop((x1, y1, x2, y2))
            detected_images.append(cropped_image)

        num_detections = len(detected_images)
        print(f"Number of detections: {num_detections}")

        return detected_images, num_detections, bboxes, labels
