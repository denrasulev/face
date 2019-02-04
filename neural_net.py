"""
Build Convolution Neural Net for facial emotion recognition
"""

import copy
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchsummary import summary
from torchvision import datasets, transforms

# Data directories
DATA_DIR = './data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')

# Define augmentation transformations
TRANSFORM = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}

# Load the data
DIRS = {'train': TRAIN_DIR,
        'valid': VALID_DIR}

MODE = ['train', 'valid']

IMAGES_SET = {x: datasets.ImageFolder(DIRS[x], transform=TRANSFORM[x]) for x
              in MODE}

DATA_LOADER = {x: torch.utils.data.DataLoader(IMAGES_SET[x],
                                              batch_size=64, shuffle=True,
                                              num_workers=4) for x in MODE}

DATASET_SIZE = {x: len(IMAGES_SET[x]) for x in MODE}

# Print dataset stats
print("Train dataset has", len(IMAGES_SET['train']), "images")
print("Valid dataset has", len(IMAGES_SET['valid']), "images")
print("Class names are :", IMAGES_SET['train'].classes)


# Define model
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.cnv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cnv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.cnv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.cnv4 = nn.Conv2d(256, 512, kernel_size=3)

        self.fc_1 = nn.Linear(512 * 9 * 9, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.outp = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, inp):
        inp = self.relu(self.cnv1(inp))
        inp = self.relu(self.cnv2(inp))
        inp = self.pool(inp)
        inp = self.drop(inp)

        inp = self.relu(self.cnv3(inp))
        inp = self.relu(self.cnv4(inp))
        inp = self.pool(inp)
        inp = self.drop(inp)

        inp = inp.view(inp.size(0), -1)

        inp = self.relu(self.fc_1(inp))
        inp = self.relu(self.fc_2(inp))
        inp = self.outp(inp)

        return inp


# Training function
def train_model(tf_model, tf_criterion, tf_optimizer, tf_scheduler, tf_epochs):
    """This function trains a model, using provided criterion, optimizer,
    scheduler and a number of epochs.

    Args:
        tf_model (class): name of pre trained model
        tf_criterion (class): criterion for calculating loss
        tf_optimizer (class): what kind of optimizer to use
        tf_scheduler (class): learning rate changer on schedule
        tf_epochs (int): number of epochs to train the model

    Returns:
        class: trained model

    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf_model.to(device)
    time_started = time.time()

    best_model_wts = copy.deepcopy(tf_model.state_dict())
    best_accuracy = 0.0

    for ep in range(tf_epochs):
        print('Epoch {}/{}'.format(ep + 1, tf_epochs))
        print('-' * 10)

        # Each epoch has training and validation MODE
        for tf_mode in ['train', 'valid']:

            if tf_mode == 'train':
                # Set MODEL to training MODE
                tf_scheduler.step()
                tf_model.train()
            else:
                # Set MODEL to evaluation MODE
                tf_model.eval()

            running_loss = 0.0
            running_corr = 0

            # Iterate over data
            for inputs, labels in DATA_LOADER[tf_mode]:

                # Move data and labels to the processing device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                tf_optimizer.zero_grad()

                # Forward and track history only in training MODE
                if tf_mode == 'train':
                    with torch.set_grad_enabled(True):
                        # Get predictions
                        outputs = tf_model(inputs)
                        _, predictions = torch.max(outputs, 1)

                        # Calculate loss
                        loss = tf_criterion(outputs, labels)

                        # Backward and optimize
                        loss.backward()
                        tf_optimizer.step()

                if tf_mode == 'valid':
                    with torch.set_grad_enabled(False):
                        # Get predictions
                        outputs = tf_model(inputs)
                        _, predictions = torch.max(outputs, 1)

                        # Calculate loss
                        loss = tf_criterion(outputs, labels)

                # Calculate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corr += torch.sum(predictions == labels.data)

            # Calculate loss and accuracy for the current epoch
            epoch_loss = running_loss / DATASET_SIZE[tf_mode]
            epoch_accuracy = running_corr.double() / DATASET_SIZE[tf_mode]

            print('{} loss: {:.4f} accuracy: {:.4f}'.format(
                tf_mode, epoch_loss, epoch_accuracy))

            # Deep copy the MODEL
            if tf_mode == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(tf_model.state_dict())

        # Print empty line to separate epochs outputs
        print()

    # Print time taken and best accuracy achieved
    time_elapsed = time.time() - time_started
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:4f}'.format(best_accuracy))

    # Load best MODEL weights
    tf_model.load_state_dict(best_model_wts)

    # Return the MODEL
    return tf_model


# Accuracy calculation
def calc_accuracy(model, test_set_path, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device=device)

    with torch.no_grad():

        batch_accuracy = []

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image_dataset = datasets.ImageFolder(
            test_set_path, transform=data_transform)

        data_loader = torch.utils.data.DataLoader(
            image_dataset, batch_size=batch_size, shuffle=True)

        for _, (inputs, labels) in enumerate(data_loader):

            # if GPU is available move data to GPU
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            # get predictions
            outputs = model.forward(inputs)
            _, predicted = outputs.max(dim=1)
            equals = predicted == labels.data

            efm = equals.float().mean()
            batch_accuracy.append(efm.cpu().numpy())

        mean_acc = np.mean(batch_accuracy)
        print("Mean accuracy: {:4f}".format(mean_acc))

    return mean_acc


# Initialize model
MODEL = NeuralNet()

# Print model's summary
print("\nModel's summary:")
summary(MODEL, (1, 48, 48))

# Define hyper parameters
LR = 0.01
EPOCHS = 50

# Define loss function and optimizer
# CRITERION = nn.CrossEntropyLoss()
# OPTIMIZER = optim.SGD(MODEL.parameters(), lr=LR, momentum=0.9)
CRITERION = nn.NLLLoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LR)

# Decay learning rate by a factor of 0.1 every 5 epochs
SCHEDULER = lr_scheduler.StepLR(OPTIMIZER, step_size=10, gamma=0.1)

# Train the model
MODEL_FT = train_model(MODEL, CRITERION, OPTIMIZER, SCHEDULER, EPOCHS)

# Calculate accuracy
ACC = calc_accuracy(MODEL_FT, './data/test')

# Save model
torch.save(MODEL_FT.state_dict(), 'model_' + str(ACC) + '_acc.pt')
