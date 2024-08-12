import streamlit as st
st.set_page_config(layout="wide", page_title="Beer bottle analyzing application")
import pandas as pd
from PIL import Image
import get_image
import object_detection
import beer_classification
import GD_download

st.header('Beer bottle analyzing application')
st.markdown("Check [Github](https://github.com/willemgeest/BeerCooler/tree/webapp) for more info")


train_folder = 'data/detected/train'
# set parameters
model_name = "beerchallenge_resnet50_7brands.pth"
scored_image_location = 'latest_uploaded_photo_scored.jpg'
class_names = beer_classification.get_classes(train_folder)
img_location = 'latest_camera_photo.jpg'

# download beer classification model from Google Drive (if not already available)
GD_download.get_beerclass_model_Drive(modelname=model_name)

# upload file
image = st.file_uploader("Please upload your beer picture here")

if image is not None:
    image = Image.open(image)

    with st.spinner('Image is being analyzed... Please wait a few seconds...'):
        # define 4 columns
        column1, column2, column3, column4 = st.columns(4)

        with column1:
            # plot original image
            st.markdown('**Original image:**')
            st.image(get_image.resize_image(image=image, max_width=400, max_heigth=600))

        # object detection
        obj_det_model = object_detection.get_obj_det_model_DirectDownload()

        image_scored, n_beers = object_detection.crop_beers(image=image, model=obj_det_model, threshold=0.8, GPU=False)

        if n_beers > 0:
            image_scored.save(scored_image_location)
            img_to_classify = scored_image_location
        else:
            # If no beers are detected, use the original image
            img_to_classify = img_location
            image.save(img_to_classify)

        img_heatmap, probabilities, label = beer_classification.beer_classification(
            img_location=img_to_classify,
            heatmap_location='heatmap_uploaded.jpg')

        with column2:
            if n_beers > 0:
                st.markdown('**Detected beer bottle:**')
                st.image(get_image.resize_image(image=image_scored, max_width=400, max_heigth=400))
            else:
                st.markdown('**No beers detected, using the original image for classification.**')

        with column3:
            st.markdown('**Predicted beer brand:**')
            # logo
            logo_location = 'logos/' + str(label.lower()) + '.png'
            st.image(get_image.resize_image(image=Image.open(logo_location).convert('RGB'), max_width=300, max_heigth=300))

            # create df with probabilities
            probabilities = probabilities.tolist()[0]
            df = pd.DataFrame([round(num * 100, 1) for num in probabilities], class_names)
            df.columns = ['(%)']
            st.markdown('**Probabilities:**')
            st.dataframe(df.style.format('{:7,.1f}').highlight_max(axis=0))

        with column4:
            st.markdown(f"**Heatmap (what makes the algorithm think it's {str(label)})**")
            st.image(get_image.resize_image(image=img_heatmap, max_width=400, max_heigth=400))
