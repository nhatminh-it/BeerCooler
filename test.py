import streamlit as st
import pandas as pd
from PIL import Image
import get_image
import yaml
import torch

from model.object_detect import BeerDetector
from model.classifier import BeerClassifier
from model.utils import get_classes
from util_plot import plot_bbox_with_class, probabilities_to_dataframe, display_heatmap, resize_image

# Load configuration from config.yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Object Detection Configurations
MODEL_ID = config['object_detection']['model']['id']
LOCAL_MODEL_DIR = config['object_detection']['model']['local_model_dir']
DEVICE = torch.device(config['object_detection']['device']['type'])
SETTINGS = config['object_detection']['settings']

# Classification Configurations
CLASS_MODEL_PATH = config['classification']['model']['path']
LOGOS_FOLDER_PATH = config['classification']['logos_folder']['path']
USE_GPU = config['classification']['device']['type']

# Location to save last run
scored_image_location = 'latest_uploaded_photo_scored.jpg'
img_location = 'latest_picture/latest_camera_photo.jpg'

# Streamlit UI Start
st.header('Advanced Beer Analyzing Application')
image = st.file_uploader("Please upload your beer picture here", type=["jpg", "jpeg", "png"])

# Initialize models
beer_detector = BeerDetector(MODEL_ID, LOCAL_MODEL_DIR, DEVICE)
beer_classifier = BeerClassifier(CLASS_MODEL_PATH, LOGOS_FOLDER_PATH, GPU=USE_GPU)
label_classes = get_classes(LOGOS_FOLDER_PATH)

if image is not None:
    image = Image.open(image)
    with st.spinner('Image is being analyzed... Please wait a few seconds...'):
        st.markdown('**Original image:**')
        resized_image = resize_image(image, max_width=400, max_height=600)
        st.image(resized_image)

        # Process image object detection
        detected_beers, n_beers = beer_detector.process_image(image)

        if n_beers > 0:
            for i, beer in enumerate(detected_beers):
                scored_image_location_i = f'latest_picture/{scored_image_location}_{i}.jpg'
                beer.save(scored_image_location_i)
                img_to_classify = scored_image_location_i

                # Perform beer classification
                predicted_class, probabilities, img_heatmap = beer_classifier.predict(
                    image_source=scored_image_location_i)

                st.markdown(f"### Beer Bottle {i + 1}")

                column1, column2, column3, column4 = st.columns(4)

                with column1:
                    st.markdown('**Detected beer bottle:**')
                    st.image(resize_image(beer, max_width=400, max_height=400))

                with column2:
                    st.markdown('**Predicted beer brand:**')
                    logo_location = 'logos/' + str(predicted_class.lower()) + '.png'
                    st.image(resize_image(Image.open(logo_location).convert('RGB'), max_width=300, max_height=300))

                with column3:
                    st.markdown('**Probabilities:**')
                    df = probabilities_to_dataframe(probabilities, label_classes)
                    st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

                with column4:
                    st.markdown(f"**Heatmap (what makes the algorithm think it's {predicted_class}):**")
                    st.image(resize_image(img_heatmap, max_width=400, max_height=400))

        else:
            st.markdown('**No beers detected, using the original image for classification.**')
            img_to_classify = img_location
            image.save(img_to_classify)

            predicted_class, probabilities, img_heatmap = beer_classifier.predict(image_source=img_to_classify)

            st.markdown("### Original Image")

            column1, column2, column3, column4 = st.columns(4)

            with column1:
                st.markdown('**Original image:**')
                st.image(resize_image(image, max_width=400, max_height=400))

            with column2:
                st.markdown('**Predicted beer brand:**')
                logo_location = 'logos/' + str(predicted_class.lower()) + '.png'
                st.image(resize_image(Image.open(logo_location).convert('RGB'), max_width=300, max_height=300))

            with column3:
                st.markdown('**Probabilities:**')
                df = probabilities_to_dataframe(probabilities, label_classes)
                st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

            with column4:
                st.markdown(f"**Heatmap (what makes the algorithm think it's {predicted_class}):**")
                st.image(resize_image(img_heatmap, max_width=400, max_height=400))