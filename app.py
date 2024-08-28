import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.models as models
from io import BytesIO
from model.object_detect import ObjectDetector
from model.object_detect import ModelManager  # Import ModelManager

import yaml
from model.classifier import BeerClassifier
from model.object_detect import BeerDetector
from model.classifier import BeerClassifier
from utils.utils import get_classes
# from utils.plot import plot_bbox_with_class, probabilities_to_dataframe, display_heatmap, resize_image

# Load configuration from config.yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Object Detection Configurations
MODEL_ID = config['object_detection']['model']['id']
LOCAL_MODEL_DIR = config['object_detection']['model']['local_model_dir']
DEVICE = torch.device(config['object_detection']['device']['type'])
# IMAGE_URL = config['object_detection']['image']['url']
SETTINGS = config['object_detection']['settings']

# Classification Configurations
CLASS_MODEL_PATH = config['classification']['model']['path']
LOGOS_FOLDER_PATH = config['classification']['logos_folder']['path']
USE_GPU = config['classification']['device']['type']

# Initialize models
beer_detector = BeerDetector(MODEL_ID, LOCAL_MODEL_DIR, DEVICE)
beer_classifier = BeerClassifier(CLASS_MODEL_PATH, LOGOS_FOLDER_PATH, GPU=USE_GPU)
# Get classes name for inference
label_classes = get_classes(LOGOS_FOLDER_PATH)

# Plot bounding boxes with detected classes
def plot_bbox_with_class(image, bboxes, labels, classes):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label, beer_class in zip(bboxes, labels, classes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f'{label}: {beer_class}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    st.pyplot(fig)


# --- Classification (EfficientNet-B7) ---
def load_classification_model(model_path, GPU=False):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    num_classes = state_dict['classifier.1.weight'].size(0)
    model = models.efficientnet_b7(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    model.eval()  # Set to evaluation mode for inference
    return model, num_classes


# Classification prediction
def predict_beer_classification(model, cropped_image, class_names):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = data_transforms(cropped_image).unsqueeze(0)
    image = image.to('cpu')

    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    probabilities = outputs.softmax(dim=1).detach().cpu().numpy()
    predicted_class = class_names[preds.item()]

    return predicted_class, probabilities


# --- Main Integration ---
def integrate_detection_and_classification(image_url, object_detector, model_classification, class_names):
    # Load image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    # Object Detection
    detection_results = object_detector.find_objects(image)  # Updated to use object_detector.find_objects
    bboxes = detection_results['<OD>']['bboxes']
    labels = detection_results['<OD>']['labels']

    # Classify each detected beer can
    classified_beers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integer coordinates
        cropped_image = image.crop((x1, y1, x2, y2))  # Crop detected region
        predicted_class, probabilities = predict_beer_classification(model_classification, cropped_image, class_names)
        classified_beers.append(predicted_class)

    # Plot the results
    plot_bbox_with_class(image, bboxes, labels, classified_beers)


# --- Streamlit UI ---
st.title("Beer Bottle Detection & Classification")

# Input field for image URL
image_url = st.text_input("Enter an image URL for beer bottle detection",
                          value='https://ganhhao.com.vn/wp-content/uploads/2019/05/nong-do-con-cua-cac-loai-bia-o-viet-nam1_800x400-600x400.jpg')

if image_url:
    st.write(f"Using image from: {image_url}")

    # Load detection and classification models
    model_classification_path = 'checkpoints/sabeco-internal-classification_efficientnet_b7.pth'

    # Initialize ModelManager and ObjectDetector
    model_id = 'microsoft/Florence-2-large'
    local_model_dir = 'checkpoints/Florence-2-large'
    device = torch.device('cpu')  # Set to CPU

    # Initialize ModelManager
    model_manager = ModelManager(model_id=model_id, local_model_dir=local_model_dir, device=device)

    # Pass ModelManager to ObjectDetector
    object_detector = ObjectDetector(model_manager)

    # Load the classification model
    model_classification, num_classes = load_classification_model(model_classification_path)

    # Load class names (use actual beer brand names)
    class_names = [f'Class{i + 1}' for i in range(num_classes)]

    # Display original image
    st.image(image_url, caption="Image from URL", use_column_width=True)

    # Perform detection and classification
    st.write("Analyzing the image...")
    integrate_detection_and_classification(image_url, object_detector, model_classification, class_names)

# Initialize ModelManager
model_manager = ModelManager(model_id=model_id, local_model_dir=local_model_dir, device=device)
# Pass ModelManager to ObjectDetector
object_detector = ObjectDetector(model_manager)
detection_results = object_detector.find_objects(image)
