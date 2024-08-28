import streamlit as st
import pandas as pd
from PIL import Image
from utils import get_image
import yaml
import torch
from io import BytesIO
from model.object_detect import BeerDetector
from model.classifier import BeerClassifier
from utils.utils import get_classes
from utils.plot import plot_bbox_with_class

# Load configuration from config.yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Object Detection Configurations
MODEL_ID = config['object_detection']['model']['id']
LOCAL_MODEL_DIR = config['object_detection']['model']['local_model_dir']
DEVICE = torch.device(config['object_detection']['device']['type'])

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

# Get classes name for inference
label_classes = get_classes(LOGOS_FOLDER_PATH)

if image is not None:
    image = Image.open(image)

    # Create two columns: one for the original image and one for the results (placeholder)
    col1, col2 = st.columns([1, 1])

    with st.spinner('Image is being analyzed... Please wait a few seconds...'):
        # Display the original image in the left column
        with col1:
            st.markdown('**Original image:**')
            st.image(get_image.resize_image(image=image, max_width=400, max_heigth=600))

        # Set a placeholder for detected beer bottles in the right column
        placeholder = col2.empty()

        # Process image object detection
        detected_beers, n_beers, bboxes, labels_det = beer_detector.process_image(image)

        if n_beers > 0:
            classified_beers_list = []

            # Create a container to update the right column
            with placeholder.container():
                st.markdown("### Detected Beer Bottles")

                for i, beer in enumerate(detected_beers):
                    # Save the scored image and prepare for classification
                    scored_image_location_i = f'latest_picture/{scored_image_location}_{i}.jpg'
                    beer.save(scored_image_location_i)
                    img_to_classify = scored_image_location_i

                    # Perform beer classification
                    predicted_class, probabilities, img_heatmap = beer_classifier.predict(image_source=scored_image_location_i)
                    classified_beers_list.append(predicted_class)

                    # Display classification results for each detected beer in the right column
                    st.markdown(f"### Beer Bottle {i + 1}")

                    # Detected cropped beer bottle
                    st.markdown('**Cropped detected beer bottle:**')
                    st.image(get_image.resize_image(image=beer, max_width=300, max_heigth=300))

                    # Display brand logo and classification results
                    column1, column2 = st.columns([1, 1])

                    with column1:
                        st.markdown('**Predicted beer brand:**')
                        # Display brand logo
                        logo_location = 'logos/' + str(predicted_class.lower()) + '.png'
                        st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=150, max_heigth=150))

                    with column2:
                        st.markdown('**Probabilities:**')
                        # Convert probabilities to a DataFrame
                        probabilities = probabilities.tolist()[0]
                        df = pd.DataFrame([round(num * 100, 1) for num in probabilities], label_classes)
                        df.columns = ['(%)']
                        st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

                    # Heatmap for the detected beer
                    st.markdown(f"**Heatmap (what makes the algorithm think it's {str(predicted_class)}):**")
                    st.image(get_image.resize_image(image=img_heatmap, max_width=300, max_heigth=300))

            # Plot the bounding boxes with detected labels and classes below the original image in the left column
            with col1:
                st.markdown("### Detected Beer Bottles with Bounding Boxes")
                plot_bbox_with_class(image, bboxes, labels_det, classified_beers_list)

        else:
            st.markdown('**No beers detected, using the original image for classification.**')
            img_to_classify = img_location
            image.save(img_to_classify)

            predicted_class, probabilities, img_heatmap = beer_classifier.predict(image_source=img_to_classify)

            with placeholder.container():
                st.markdown("### Original Image")

                st.image(get_image.resize_image(image=image, max_width=300, max_heigth=400))

                column1, column2 = st.columns([1, 1])

                with column1:
                    st.markdown('**Predicted beer brand:**')
                    # Display brand logo
                    logo_location = 'logos/' + str(predicted_class.lower()) + '.png'
                    st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=150, max_heigth=150))

                with column2:
                    st.markdown('**Probabilities:**')
                    # Convert probabilities to a DataFrame
                    probabilities = probabilities.tolist()[0]
                    df = pd.DataFrame([round(num * 100, 1) for num in probabilities], label_classes)
                    df.columns = ['(%)']
                    st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

                st.markdown(f"**Heatmap (what makes the algorithm think it's {str(predicted_class)}):**")
                st.image(get_image.resize_image(image=img_heatmap, max_width=300, max_heigth=300))