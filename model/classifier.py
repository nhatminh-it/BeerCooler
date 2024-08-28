import torch
import requests
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from utils.utils import get_classes


class BeerClassifier:
    def __init__(self, model_path, logos_folder_path, GPU=True):
        self.device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.class_names = get_classes(logos_folder_path)

    def _load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        num_classes = state_dict['classifier.1.weight'].size(0)
        self.class_names = [f'Class{i + 1}' for i in range(num_classes)]

        model = models.efficientnet_b7(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def _preprocess_image(self, image_source):
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if image_source.startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Load image from file path
            image = Image.open(image_source).convert('RGB')

        image = data_transforms(image)
        image = image.unsqueeze(0)
        return image.to(self.device)

    def _register_hooks(self, image):
        gradients = []
        activations = []

        def hook_function_grad(grad):
            gradients.append(grad)

        def hook_function_act(module, input, output):
            activations.append(output)
            output.register_hook(hook_function_grad)

        self.model.features[-1].register_forward_hook(hook_function_act)

        return gradients, activations

    def _generate_heatmap(self, image, class_int, gradients, activations):
        self.model(image)[:, class_int].backward()

        activations = activations[0].detach()
        gradients = gradients[0]

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = np.maximum(heatmap.cpu().numpy(), 0)

        heatmap /= np.max(heatmap)

        return heatmap

    def _combine_heatmap_with_image(self, heatmap, image_source):
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224))

        heatmap = np.asarray(heatmap_resized) / 255
        cm = plt.get_cmap('jet')
        heatmap_colored = cm(heatmap)
        heatmap_colored = np.delete(heatmap_colored, 3, 2)

        if image_source.startswith(('http://', 'https://')):
            # Load image from URL
            original_image = Image.open(BytesIO(requests.get(image_source).content)).resize((224, 224))
        else:
            # Load image from file path
            original_image = Image.open(image_source).resize((224, 224))

        original_image_np = np.asarray(original_image)
        combined = (1.0 - 0.4) * original_image_np + 0.6 * heatmap_colored * 255
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        return Image.fromarray(combined)

    def predict(self, image_source, class_int=None):
        image = self._preprocess_image(image_source)
        gradients, activations = self._register_hooks(image)

        outputs = self.model(image)
        if class_int is None:
            class_int = torch.argmax(outputs).item()

        heatmap = self._generate_heatmap(image, class_int, gradients, activations)
        combined_image = self._combine_heatmap_with_image(heatmap, image_source)

        predicted_class = self.class_names[class_int]
        probabilities = outputs.softmax(dim=1).detach().cpu().numpy()

        return predicted_class, probabilities, combined_image



# Define model path and logos folder path
model_path = 'checkpoints/sabeco-internal-classification_efficientnet_b7.pth'
logos_folder_path = 'logos'

# Instantiate and use the BeerClassifier
classifier = BeerClassifier(model_path, logos_folder_path, GPU=False)

# Test with an image URL
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfDqis3iKACKqHKCyNlV6m-D6SqmxvBnlI0A&s'
predicted_class, probabilities, combined_image = classifier.predict(image_url)

# Print predicted class and probabilities
print(f'Predicted class: {predicted_class}')
print(f'Probabilities: {probabilities}')

# Display the combined heatmap image using matplotlib
plt.imshow(combined_image)
plt.axis('off')
plt.show()

# Test with a local image file path
image_path = 'data/original/lac_viet/7d9fdc0e9e027a0fa6a6ab5b2be4f05b.jpg'
predicted_class, probabilities, combined_image = classifier.predict(image_path)

# Print predicted class and probabilities
print(f'Predicted class: {predicted_class}')
print(f'Probabilities: {probabilities}')

# Display the combined heatmap image using matplotlib
plt.imshow(combined_image)
plt.axis('off')
plt.show()
