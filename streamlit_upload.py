import requests
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
croped_image_name = 'latest_uploaded_photo_detected'
img_location = f'latest_picture/{croped_image_name}.jpg'

# Cache models to avoid reinitialization using st.cache_resource
@st.cache_resource
def load_models():
    beer_detector = BeerDetector(MODEL_ID, LOCAL_MODEL_DIR, DEVICE)
    beer_classifier = BeerClassifier(CLASS_MODEL_PATH, LOGOS_FOLDER_PATH, GPU=USE_GPU)
    label_classes = get_classes(LOGOS_FOLDER_PATH)
    return beer_detector, beer_classifier, label_classes

beer_detector, beer_classifier, label_classes = load_models()

st.header('Advanced Beer Analyzing Application')

# Option to switch between uploading an image or entering a URL
option = st.radio("Select Image Input Method", ('Upload Image', 'Image URL'))

# Initialize variables
image = None
image_url = None

# Option to upload an image or enter a URL
if option == 'Upload Image':
    uploaded_file = st.file_uploader("Please upload your beer picture here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == 'Image URL':
    image_url = st.text_input("Please enter the image URL here")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading image from URL: {e}")

# Defer image processing until after a valid image is provided
if image:
    # Create two columns: one for the original image and one for the results (placeholder)
    col1, col2 = st.columns([1, 1])

    with st.spinner('Image is being analyzed... Please wait a few seconds...'):
        # Display the original image in the left column
        with col1:
            st.markdown('**Original image:**')
            st.image(get_image.resize_image(image=image, max_width=400, max_heigth=600))


        st.markdown(
            """
            <style>
            /* Apply the vertical line with reduced opacity and lighter color */
            div[data-testid="column"] {
                border-left: 1px solid rgba(255, 255, 255, 0.15);  /* Very light gray with 10% opacity for subtlety */
                padding-left: 15px;   /* Reduced padding to tighten content */
                padding-right: 15px;  /* Add small padding on the right to balance the layout */
                height: 100%;  /* Make sure the height fits dynamically */
            }
            /* Adjust the content inside columns to reduce excess space */
            div[data-testid="stVerticalBlock"] > div {
                margin-bottom: 0px !important;  /* Tighten the bottom margin */
                padding-bottom: 0px !important;  /* Remove extra padding */
            }
            /* Reduce the gap between inner elements */
            div[data-testid="stHorizontalBlock"] > div {
                padding-left: 10px !important;
                padding-right: 0px !important;
                margin-right: 0px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Set a placeholder for detected beer bottles in the right column
        placeholder = col2.empty()

        # Process image object detection
        detected_beers, n_beers, bboxes, labels_det = beer_detector.process_image(image)

        if n_beers > 0:
            classified_beers_list = []

            # Create a container to update the right column (col2)
            with placeholder.container():
                st.markdown("### Detected Beer Bottles")

                for i, beer in enumerate(detected_beers):
                    # Save the scored image and prepare for classification
                    croped_image_name_i = f'latest_picture/{croped_image_name}_{i}.jpg'
                    beer.save(croped_image_name_i)
                    img_to_classify = croped_image_name_i

                    # Perform beer classification
                    predicted_class, probabilities, img_heatmap = beer_classifier.predict(
                        image_source=croped_image_name_i)
                    classified_beers_list.append(predicted_class)

                    img_heatmap = img_heatmap.resize(beer.size)

                    # Display classification results for each detected beer in the right column (col2)
                    st.markdown(f"### Beer Bottle {i + 1}")

                    # Detected cropped beer bottle
                    st.markdown('**Cropped detected beer bottle:**')
                    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns([1, 1, 1, 1])

                    with sub_col1:
                        st.image(get_image.resize_image(image=beer, max_width=300, max_heigth=300))

                    with sub_col2:
                        st.markdown('**Predicted beer brand:**')
                        # Display brand logo
                        logo_location = 'logos/' + str(predicted_class.lower()) + '.png'
                        st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=150,
                                                        max_heigth=150))

                    with sub_col3:
                        st.markdown('**Probabilities:**')
                        # Convert probabilities to a DataFrame
                        probabilities = probabilities.tolist()[0]
                        df = pd.DataFrame([round(num * 100, 1) for num in probabilities], label_classes)
                        df.columns = ['(%)']
                        st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

                    with sub_col4:
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