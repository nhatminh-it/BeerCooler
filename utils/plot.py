import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import streamlit as st
# Function to plot bounding boxes and detected class names
# def plot_bbox_with_class(image, bboxes, labels, classes):
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#
#     for bbox, label, beer_class in zip(bboxes, labels, classes):
#         x1, y1, x2, y2 = bbox
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(x1, y1, f'{label}: {beer_class}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
#
#     ax.axis('off')
#     plt.show()


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

# Function to convert probabilities into a DataFrame
def probabilities_to_dataframe(probabilities, label_classes):
    probabilities = probabilities.tolist()[0]
    df = pd.DataFrame([round(num * 100, 1) for num in probabilities], label_classes)
    df.columns = ['(%)']
    return df

# Function to display combined image with heatmap
def display_heatmap(img_heatmap, width=400, height=400):
    plt.imshow(img_heatmap)
    plt.axis('off')
    plt.gcf().set_size_inches(width / 100, height / 100)  # Size in inches
    plt.show()

# Function to resize image
def resize_image(image, max_width, max_height):
    img_ratio = min(max_width / image.width, max_height / image.height)
    new_size = (int(image.width * img_ratio), int(image.height * img_ratio))
    return image.resize(new_size, Image.ANTIALIAS)