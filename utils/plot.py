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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st


def plot_bbox_with_class(image, bboxes, labels, classes, class_colors=None):
    # Define a color map for different beer classes, if not provided
    if class_colors is None:
        class_colors = {
            'saigon_chill': 'blue',
            'saigon_export': 'red',
            'saigon_gold': 'yellow',
            'saigon_lager': 'green',
            'saigon_special': 'purple',
            'lac_viet': 'cyan',
            '333': 'orange'
        }

    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label, beer_class in zip(bboxes, labels, classes):
        x1, y1, x2, y2 = bbox
        color = class_colors.get(beer_class, 'red')  # Assign color for the beer class

        # Draw bounding box around the object
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Display the class in the top-left corner of each bounding box with a smaller annotation box
        plt.text(
            x1, y1, f'{label}: {beer_class}',
            color='white',
            fontsize=6,  # Smaller font size for the annotation
            bbox=dict(facecolor=color, alpha=0.5, pad=1)  # Smaller padding for a smaller box
        )

    # Optional: Display each class in the top-left corner of the entire image with the corresponding color
    y_offset = 10  # Start position for the text
    for beer_class in set(classes):  # Go through each unique class
        color = class_colors.get(beer_class, 'black')  # Get the color for each class
        plt.text(10, y_offset, beer_class, color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.7))
        y_offset += 20  # Move the next text slightly lower for each class

    ax.axis('off')  # Remove axes for a cleaner plot
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