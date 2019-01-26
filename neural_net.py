"""
Build Convolution Neural Net for facial emotion recognition
"""

import os

import torch
from torch import nn, optim
from torchsummary import summary
from torchvision import datasets, transforms

# Data directories
DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')

# Define augmentation transformations
NORMALIZE = transforms.Normalize([0.5], [0.5])

TRANSFORMATIONS = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        NORMALIZE
    ]),
    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
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
print("Training dataset size:", len(IMAGES_DATASET['train']))
print("Validation dataset size:", len(IMAGES_DATASET['valid']))

DATA_LOADERS = {x: torch.utils.data.DataLoader(IMAGES_DATASET[x],
                                               batch_size=64, shuffle=True,
                                               num_workers=4) for x in MODE}

DATASET_SIZE = {x: len(IMAGES_DATASET[x]) for x in MODE}
print(DATASET_SIZE)

CLASS_NAMES = IMAGES_DATASET['train'].classes
print(CLASS_NAMES)

# Each image is 48x48 pixels, 2304 in total
# These are greyscale images, so only 1 channel
# There are 7 output classes


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.norm = nn.BatchNorm2d(64)
        self.hidd = nn.Linear(64 * 22 * 22, 128)
        self.drop = nn.Dropout(0.2)
        self.outp = nn.Linear(128, 10)
        self.actv = nn.ReLU()

    def forward(self, inp):
        inp = self.actv(self.conv1(inp))
        inp = self.actv(self.conv2(inp))
        inp = self.pool(inp)
        inp = self.norm(inp)
        inp = inp.view(inp.size(0), -1)
        inp = self.actv(self.hidd(inp))
        inp = self.drop(inp)
        inp = self.outp(inp)
        return inp


model = NeuralNet()

summary(model, (1, 48, 48))

params = list(model.parameters())

conv1Params = list(model.conv1.parameters())

# kernels in conv1Params[0] and biases in conv1Params[1]

print(conv1Params[0].data.mean())

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the network
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(DATA_LOADERS['train'], 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
