import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path, num_classes, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, class_names, device):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Thêm batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = nn.Softmax(dim=1)(outputs)  # Áp dụng softmax để chuyển đổi thành xác suất
        _, preds = torch.max(probs, 1)

    return class_names[preds[0]], probs[0][preds[0]].item()

def load_class_names(train_folder):
    return sorted([d.name for d in os.scandir(train_folder) if d.is_dir()])

model_path = 'beerchallenge_resnet50_vn_6.pth'
train_folder = 'data/detected/train'
# image_path = '/Users/leduy/PycharmProjects/BeerClassification/BeerCooler/data/detected/train/333/629382.jpg'
# image_path = '/Users/leduy/PycharmProjects/BeerClassification/BeerCooler/data/original/train/saigon_gold/4-2022-21909.jpg'
# image_path = 'data/original/train/saigon_gold/562d039769b76c4d724df8f10409ffab.jpg'
# image_path = 'data/original/train/saigon_gold/400357.jpg'
# image_path = 'data/original/train/lac_viet/4202023449.jpg'
# image_path = 'data/original/train/333/bia-333-export.png'
image_path = 'data/original/train/333/bia-333.jpg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = load_class_names(train_folder)
model = load_model(model_path, len(class_names), device)

predicted_class, confidence_score = predict_image(image_path, model, class_names, device)
print(f'Dự đoán: {predicted_class}, Conf_score: {confidence_score:.4f}')
