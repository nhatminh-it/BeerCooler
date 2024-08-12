import pandas as pd
import sys
import object_detection
import time
import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
import os
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import random


def split_trainval(folder_beers, fraction_train=0.85):
    brands = os.listdir(folder_beers)

    #create train and val folders
    os.makedirs(folder_beers + '/train')
    os.makedirs(folder_beers + '/val')

    for brand in brands:
        # get images
        images = os.listdir(folder_beers + '/' + brand)

        # select random images to train / validate
        n_train = int(round(len(images) * fraction_train, 0))
        images_train = random.sample(images, n_train)
        images_val = [x for x in images if x not in images_train]

        # move images to new folders
        os.makedirs(folder_beers + '/train' + '/' + brand)
        for image in images_train:
            os.rename(src=folder_beers + '/' + brand + '/' + image,
                      dst=folder_beers + '/train' + '/' + brand + '/' + image)

        os.makedirs(folder_beers + '/val' + '/' + brand)
        for image in images_val:
            os.rename(src=folder_beers + '/' + brand + '/' + image,
                      dst=folder_beers + '/val' + '/' + brand + '/' + image)

        #remove original folder brand = 'amstel'
        os.rmdir(folder_beers + '/' + brand)


def crop_beers_to_folder(folder_beers, folder_beers_cropped, GPU=True):
    # Import data
    all_trainval_data = datasets.ImageFolder(root=folder_beers)

    # Get folder structure in folder_beers
    folder_beers_str = [x[0].replace(folder_beers, '') for x in os.walk(folder_beers)]

    # Create folder structure (if it does not already exist)
    for i in folder_beers_str:
        if not os.path.exists(folder_beers_cropped + i):
            os.makedirs(folder_beers_cropped + i)

    # Load object detection model
    obj_det_model = object_detection.get_obj_det_model()
    obj_det_model.eval()
    if GPU:
        obj_det_model.cuda()

    # Save results of cropped files in DataFrame
    cropped_results = pd.DataFrame(columns=['i', 'file', 'n_boxes'])

    # Crop all images
    for i in range(len(all_trainval_data)):
        image_path = all_trainval_data.samples[i][0]
        try:
            image = all_trainval_data[i][0]
            boxes, classes, labels, preds = object_detection.find_bottles(image=image, model=obj_det_model,
                                                                          detection_threshold=.8, GPU=GPU)
            # If there are multiple boxes (beers), make one large box. If there is only one beer, this doesn't change anything
            if len(boxes) > 0:
                x_start = min([x[0] for x in boxes])
                y_start = min([x[1] for x in boxes])
                x_end = max([x[2] for x in boxes])
                y_end = max([x[3] for x in boxes])

                # Crop image
                image_cropped = image.crop((x_start, y_start, x_end, y_end))

                # Save cropped image
                new_location = folder_beers_cropped + image_path.replace(folder_beers, '')
                image_cropped.save(new_location)

            # Add to DataFrame
            cropped_results = pd.concat(
                [cropped_results, pd.DataFrame([{'i': i, 'file': image_path, 'n_boxes': len(boxes)}])],
                ignore_index=True)

        except TypeError as e:
            print(f"TypeError encountered with image {image_path}: {e}")
            continue  # Skip to the next iteration if a TypeError occurs

        finally:
            print('')

        # Print progress every 25 images
        if i % 25 == 0:
            print(f"{i} / {len(all_trainval_data)} ({round(i / len(all_trainval_data) * 100)}%)")

    return cropped_results


def train_beermodel(folder_beers,
                    model_location='./beerchallenge_efficientnet_b7.pth',
                    num_epochs=25,
                    GPU=True):
    # load EfficientNet-b7
    model_ft = models.efficientnet_b7(pretrained=True)

    # set parameters
    since = time.time()
    num_ftrs = model_ft.classifier[1].in_features
    device = torch.device("cuda:0" if GPU else "cpu")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # normalize images for R, G, B (both mean and SD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # load dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(folder_beers, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    #
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))  # determine final (fully connected) layer

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_ft.load_state_dict(best_model_wts)

    torch.save(model_ft.state_dict(), model_location)

    #return model_ft


# split_trainval(folder_beers='data/original')

# crop_beers_to_folder(folder_beers='data/original', folder_beers_cropped='data/detected', GPU=False)

train_beermodel(folder_beers='data/detected/', model_location='checkpoint/beerchallenge_efficientnet_b7.pth', num_epochs=10, GPU=False)



# gg colab have some change
# crop_beers_to_folder(folder_beers='/content/drive/MyDrive/Index/data/original', folder_beers_cropped='/content/drive/MyDrive/Index/data/detected', GPU=True)

# train_beermodel(folder_beers='/content/drive/MyDrive/Index/data/detected', model_location='/content/drive/MyDrive/Index/model/beerchallenge_resnet50_6vietnambrands.pth', num_epochs=50, GPU=True)