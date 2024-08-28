import os
from typing import Union, List, Tuple
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

class ModelManager:
    def __init__(self, model_id: str, local_model_dir: str, device: torch.device):
        self.model_id = model_id
        self.local_model_dir = local_model_dir
        self.device = device
        self.model = None
        self.processor = None
        self._load_model_and_processor()

    def _fixed_get_imports(self, filename: Union[str, os.PathLike]) -> list[str]:
        imports = get_imports(filename)
        if not torch.cuda.is_available() and "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

    def _load_model_and_processor(self):
        if not os.path.exists(self.local_model_dir):
            with patch("transformers.dynamic_module_utils.get_imports", self._fixed_get_imports):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True
                ).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                self._save_model_and_processor()
        else:
            with patch("transformers.dynamic_module_utils.get_imports", self._fixed_get_imports):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_dir, trust_remote_code=True
                ).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(
                    self.local_model_dir, trust_remote_code=True
                )

    def _save_model_and_processor(self):
        self.model.save_pretrained(self.local_model_dir)
        self.processor.save_pretrained(self.local_model_dir)


class ObjectDetector:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def find_objects(self, image: Image.Image, task_prompt: str = "<OD>", text_input: str = None) -> dict:
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.model_manager.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model_manager.device)

        with torch.no_grad():
            generated_ids = self.model_manager.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self.model_manager.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.model_manager.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer


class ImageProcessor:
    @staticmethod
    def crop_objects(image: Image.Image, results: dict) -> Tuple[List[Image.Image], int]:
        boxes = results.get('<OD>', {}).get('bboxes', [])
        labels = results.get('<OD>', {}).get('labels', [])
        cropped_images = []
        image_width, image_height = image.size

        for box in boxes:
            box = [
                max(0, min(box[0], image_width)),
                max(0, min(box[1], image_height)),
                max(0, min(box[2], image_width)),
                max(0, min(box[3], image_height))
            ]
            if box[0] < box[2] and box[1] < box[3]:
                cropped_image = image.crop((box[0], box[1], box[2], box[3]))
                cropped_images.append(cropped_image)
            else:
                print(f"Invalid bounding box: {box}")

        return cropped_images, len(cropped_images)


class BeerDetector:
    def __init__(self, model_id: str, local_model_dir: str, device: torch.device):
        self.model_manager = ModelManager(model_id, local_model_dir, device)
        self.object_detector = ObjectDetector(self.model_manager)
        self.image_processor = ImageProcessor()

    def process_image(self, image_source: str):
        # Determine if the input is a URL or a file path and load the image accordingly
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_source, stream=True).raw)
            else:
                image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError("Invalid image source. Must be a URL, file path, or PIL Image.")

        # Run the object detection
        results = self.object_detector.find_objects(image)
        bboxes = results['<OD>']['bboxes']
        labels = results['<OD>']['labels']

        # Filter for 'bottle' and 'tin can' only
        allowed_labels = ['bottle', 'tin can']
        filtered_bboxes = []
        filtered_labels = []
        for i, label in enumerate(labels):
            if label in allowed_labels:
                filtered_bboxes.append(bboxes[i])
                filtered_labels.append(label)

        # Process only the allowed objects
        if filtered_bboxes:
            results['<OD>']['bboxes'] = filtered_bboxes
            results['<OD>']['labels'] = filtered_labels
            cropped_images, num_detections = self.image_processor.crop_objects(image, results)
        else:
            cropped_images, num_detections = [], 0

        # Print the number of detections for debugging purposes
        print(f"Number of detections: {num_detections}")

        # Return filtered results
        return cropped_images, num_detections, filtered_bboxes, filtered_labels

# if __name__ == "__main__":
#     # Configuration
#     MODEL_ID = 'microsoft/Florence-2-large'
#     LOCAL_MODEL_DIR = '/Users/leduy/PycharmProjects/BeerClassification/BeerCooler/checkpoints/Florence-2-large'
#     # DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#     DEVICE = 'cpu'
#     # Initialize and run the app
#     app = BeerDetector(MODEL_ID, LOCAL_MODEL_DIR, DEVICE)
#     IMAGE_URL = "https://heineken-vietnam.com.vn/images/2023/bia-cao-cap-quoc-te-cua-chau-a.jpg"
#     app.process_image(IMAGE_URL)
