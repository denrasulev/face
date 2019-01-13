# -*- coding: utf-8 -*-
"""Facial Emotions Recognition project.

Data exploration module.

Code to explore data, prepared for model training.
    - Show samples of data for each category
    - Print and show number of images in each category

Example:
    To show statistics on data, just run:
        $ python3 data_explore.py

This will output some sample images, print statistics on image quantities to
console and also show bar plot with number of images per category.

Attributes:
    BASE_PATH (str): Path to the data directory
    EMOTIONS (list): List of emotions, following folders' names
    CO_PLOT (int): Co-plot number for each graph of sample images
    IMAGES_PER_EMOTION (list): number of images per each emotion
    Y_POS (tuple): Tuple of numbers from 0 to number of EMOTIONS

Todo:
    * Create some tests for the code
"""

import os

import imageio
import matplotlib.pyplot as plt

# Path to the training data folder
BASE_PATH = "./data/train/"
# List of training data folder's subfolders
EMOTIONS = os.listdir(BASE_PATH)

plt.figure(0, figsize=(12, 20))
CO_PLOT = 0

# Show table with several sample images per emotion
for sub_folder in EMOTIONS:
    for i in range(1, 6):
        CO_PLOT += 1
        plt.subplot(7, 5, CO_PLOT)

        # Read first five images from current subfolder
        img = imageio.imread(BASE_PATH + sub_folder + "/"
                             + os.listdir(BASE_PATH + sub_folder)[i])

        plt.imshow(img, cmap="gray")

# Make layout tight, save to file and show
plt.tight_layout()
plt.savefig('samples_per_emotion.jpg')
plt.show()

# Print number of images per emotion to console
print("Number of Images per Emotion :")
print('-' * 30)

IMAGES_PER_EMOTION = []

# Calculate number of files in subfolders and print to console
for sub_folder in EMOTIONS:
    num_images = len(os.listdir(BASE_PATH + sub_folder))
    print('{:<10}'.format(sub_folder) + ': ' + str(num_images))
    IMAGES_PER_EMOTION.append(num_images)

# Show these numbers in bar plot
Y_POS = range(len(EMOTIONS))

# Create the bar plot
plt.bar(Y_POS, IMAGES_PER_EMOTION, alpha=0.8, zorder=2)
plt.xticks(Y_POS, EMOTIONS)
plt.grid(True, axis='y', color='lightgrey', linewidth=.5, zorder=1)
plt.title('Number of Images per Emotion')

# Make layout tight, save to file and show
plt.tight_layout()
plt.savefig('images_per_emotion.jpg')
plt.show()
