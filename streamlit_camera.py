import streamlit as st
import pandas as pd
from PIL import Image
import get_image
import object_detection
import beer_classification
import GD_download

# Set up the Streamlit app
st.set_page_config(layout="wide")
st.header("Willem's beer bottle classification algorithm")

# Input the IP address within the Streamlit app
ip_address = st.text_input("Enter the IP address of the camera:")

# Ensure the IP address is provided before proceeding
if not ip_address:
    st.warning("Please enter the IP address to continue.")
    st.stop()

model_name = "beerchallenge_resnet50_6vietnambrands.pth"
scored_image_location = 'latest_picture/latest_camera_photo_scored.jpg'
class_names = beer_classification.get_classes()
img_location = 'latest_picture/latest_camera_photo.jpg'

# Download beer classification model from Google Drive (if not already available)
@st.cache_resource
def download_model(model_name):
    GD_download.get_beerclass_model_Drive(modelname=model_name)

download_model(model_name)

# Capture image from the IP camera
image = None  # Initialize the image variable to ensure it's defined

try:
    image = get_image.get_image_IPcamera(IPv4_adress=ip_address, img_location=img_location)
    st.text('Picture captured')
except Exception as e:
    st.error(f"Failed to capture image: {e}")
    st.stop()

# Object detection
@st.cache_resource
def load_obj_det_model():
    return object_detection.get_obj_det_model()

obj_det_model = load_obj_det_model()

# Initialize n_beers to ensure it's always defined
n_beers = 0

if image is not None:
    try:
        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=True)
    except Exception as e:
        st.warning(f"GPU failed with error: {e}. Trying with CPU...")
        try:
            image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=False)
        except Exception as e:
            st.error(f"Failed to detect objects: {e}")
            st.stop()
else:
    st.error("No image available for processing.")
    st.stop()

# Process the detected beers
if n_beers > 0:
    image_scored.save(scored_image_location)
    description_objdet = 'Beer bottle detected'
else:
    description_objdet = 'No beer bottle detected'
    image_scored = image  # Fallback to the original image

# Generate heatmap
if n_beers > 0:
    try:
        img_heatmap, probabilities, label = beer_classification.beer_classification(
            img_location=scored_image_location,
            heatmap_location='.\\latest_picture\\heatmap.jpg'
        )
    except Exception as e:
        st.error(f"Heatmap generation failed: {e}")
        st.stop()

# Define 4 columns
column1, column2, column3, column4 = st.columns(4)

with column1:
    st.image(get_image.resize_image(image, max_width=400, max_heigth=600), caption='Original picture')

with column2:
    st.image(get_image.resize_image(image_scored, max_width=400, max_heigth=600), caption=description_objdet)

with column3:
    if n_beers > 0:
        probabilities = probabilities.tolist()[0]

        df = pd.DataFrame([round(num*100, 1) for num in probabilities], class_names)
        df.columns = ['(%)']
        logo_location = 'logos/' + str(label) + '.png'
        st.image(get_image.resize_image(Image.open(logo_location).convert('RGB'), max_width=400, max_heigth=600))

        st.text('Probabilities:')
        st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))
    else:
        st.text('No beers detected')

with column4:
    if n_beers > 0:
        st.image(get_image.resize_image(img_heatmap, max_width=400, max_heigth=600), caption='Heatmap')
