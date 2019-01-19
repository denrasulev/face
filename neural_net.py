"""
Build Convolution Neural Net for facial emotion recognition
"""

import os

import torch
from torch import nn
from torchvision import datasets, transforms

# Data directories
DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')

# Define augmentation transformations
NORMALIZE = transforms.normalize([0.5], [0.5])

TRANSFORMATIONS = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE
    ])
}

# Load the data
DIRS = {'train': TRAIN_DIR,
        'valid': VALID_DIR}

MODE = ['train', 'valid']

IMAGES_DATASET = {x: datasets.ImageFolder(DIRS[x], transform=TRANSFORMATIONS[
    x]) for x in MODE}

# See if it works?
print(IMAGES_DATASET['train'].size())

DATA_LOADERS = {x: torch.utils.data.DataLoader(IMAGES_DATASET[x],
                                               batch_size=64, shuffle=True,
                                               num_workers=4) for x in MODE}

DATASET_SIZE = {x: len(IMAGES_DATASET[x]) for x in MODE}
print(DATASET_SIZE)

CLASS_NAMES = IMAGES_DATASET['train'].classes
print(CLASS_NAMES)

# TODO: Define network configuration

# Each image is 48x48 pixels, 2304 in total
# These are greyscale images, so only 1 channel
# There are 7 output classes

model = nn.Sequential(
    # 1st Convolution Layer
    nn.Conv2d(1, 64, 3),
    nn.BatchNorm1d,
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.25),

    # 2nd Convolution layer
    nn.Conv2d(64, 128, 3),
    nn.BatchNorm1d,
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.25),

    # 3rd Convolution layer
    nn.Conv2d(128, 512, 3),
    nn.BatchNorm1d,
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.25),

    # 4th Convolution layer
    nn.Conv2d(512, 512, 3),
    nn.BatchNorm1d,
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.25),

    # Fully connected layer 1st layer
    nn.Linear(256, 128),
    nn.BatchNorm1d,
    nn.ReLU(),
    nn.Dropout(p=0.25),

    # Fully connected layer 2nd layer
    nn.Linear(128, 10),
    nn.BatchNorm1d,
    nn.ReLU(),
    nn.Dropout(p=0.25),

    # Output layer
    nn.LogSoftmax(dim=1))
