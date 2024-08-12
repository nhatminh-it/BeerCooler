import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Function to load class names
import os
train_folder = 'data/detected/train'
def get_classes(train_folder_path):
    return sorted([d.name for d in os.scandir(train_folder) if d.is_dir()])


# Custom ResNet class for your specific task
class ResNet(nn.Module):
    def __init__(self, model_path, num_classes, device):
        super(ResNet, self).__init__()
        self.train_folder = 'data/detected/train'
        self.resnet = resnet50(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.class_names = get_classes(train_folder_path=self.train_folder)
        # Load the pre-trained model weights
        try:
            self.resnet.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as err:
            print("Error loading model: ", err)

        # Isolate the feature blocks
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )

        # Average pooling layer
        self.avgpool = self.resnet.avgpool

        # Classifier
        self.classifier = self.resnet.fc

        # Gradient placeholder
        self.gradient = None

        # Set the model to evaluation mode
        self.to(device)
        self.eval()

    # Hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # Extract the features
        x = self.features(x)

        # Register the hook
        h = x.register_hook(self.activations_hook)

        # Complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x

# Function to predict the class of an image
def predict_image(image_path, model, class_names, device):
    # Define the preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open the image and apply the transforms
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform the forward pass and get the predicted class
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]
def beer_classification(img_location, heatmap_location, class_int=None):
    # get classes
    train_folder = 'data/detected/train'
    class_names = get_classes(train_folder_path=train_folder)
    # init the resnet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'beerchallenge_resnet50_6vietnambrands.pth'
    resnet = ResNet(model_path=model_path, num_classes=len(class_names), device=device)
    # set the evaluation mode
    _ = resnet.eval()

    #open image
    img = Image.open(img_location)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])  # normalize images for R, G, B (both mean and SD)

    img = test_transforms(img)
    # add 1 dimension to tensor
    img = img.unsqueeze(0)
    # forward pass
    pred = resnet(img)

    # tranfors tensors with results to probabilities
    sm = torch.nn.Softmax(
        dim=1)  # use softmax to convert tensor values to probs (dim = columns (0) or rows (1) have to sum up to 1?)
    probabilities = sm(pred)

    # get the gradient of the output with respect to the parameters of the model
    if class_int is None:
        pred[:, pred.argmax()].backward()  # heatmap of class with highest prob
    else:
        pred[:, class_int].backward()

    # pull the gradients out of the model
    gradients = resnet.get_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = resnet.get_activations(img).detach()
    # len(activations[0])

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # make the heatmap to be a numpy array
    heatmap = heatmap.numpy()

    # interpolate the heatmap
    img = Image.open(img_location)

    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize((img.size[0], img.size[1]))

    # Get the color map by name:
    cm = plt.get_cmap('jet')

    heatmap = np.asarray(heatmap) / 255
    # Apply the colormap like a function to any array:
    heatmap = cm(heatmap)
    heatmap = np.delete(heatmap, 3, 2)

    heatmap = heatmap * 255
    mix = (1.0 - 0.2) * np.asarray(img) + 0.9 * heatmap  # (80% of original picture + 90% of heatmap )

    mix = np.clip(mix, 0, 255).astype(np.uint8)
    # save heatmap
    Image.fromarray(mix).save(heatmap_location)
    return Image.open(heatmap_location), probabilities, class_names[pred.argmax()]

if __name__ == "__main__":
    model_path = "beerchallenge_resnet50_6vietnambrands.pth"
    train_folder = "/path/to/your/train_folder"
    image_path = "/path/to/your/image.jpg"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load class names and the model
    class_names = get_classes(train_folder)
    num_classes = len(class_names)
    model = ResNet(model_path, num_classes, device)

    # Predict the class of the image
    predicted_class = predict_image(image_path, model, class_names, device)
    print(f"Predicted class: {predicted_class}")
