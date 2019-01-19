# -*- coding: utf-8 -*-
"""Facial expressions recognition project.

The data comes from the Kaggle competition “Challenges in Representation
Learning: Facial Expression Recognition Challenge”:

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

The data consists of 48x48 pixel grayscale images of faces in the form
of arrays with a greyscale value for each pixel.

Each image  corresponds to a facial expression in one of seven categories:
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.

The dataset contains approximately 36K images.

Attributes:
    RAW_DATA (str): Path to the raw data file (fer2013.csv)
    EMOTION_INDEX (list): Numeric index for each facial expression
    PIXELS_VALUES_STRING (list): List of grayscale pixels values
    CATEGORY (list): Category name to split data into Training, Validation
    and Test sets.
"""

import csv

import imageio
import numpy as np

# Image data pre-processing
RAW_DATA = './raw_data/fer2013.csv'

EMOTION_INDEX = []
PIXELS_VALUES_STRING = []
CATEGORY = []

# Read greyscale image data, then convert, sort and save images
with open(RAW_DATA) as csv_file:
    CSV_DATA = csv.reader(csv_file)

    # Skip the header row
    next(CSV_DATA, None)

    # Counter to add to image names, so they do not overwrite
    COUNTER = 1

    # Csv file structure is as follows:
    # Emotion, Greyscale pixels values, Category
    # 0,70 80 82 72 58 58 60 43 12 46...,Training
    for row in CSV_DATA:

        EMOTION_INDEX = row[0]
        PIXELS_VALUES_STRING = row[1]
        CATEGORY = row[2]
        path = ''
        cate = ''

        # Sort images by folders
        # Create base path according to the Category of an image
        if CATEGORY == 'Training':
            path = './data/train/'
            cate = 'train'
        elif CATEGORY == 'PublicTest':
            path = './data/valid/'
            cate = 'valid'
        elif CATEGORY == 'PrivateTest':
            path = './data/test/'
            cate = 'test'

        # Augment image path according to the emotion label
        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        if EMOTION_INDEX == '0':
            path += 'angry/'
        elif EMOTION_INDEX == '1':
            path += 'disgust/'
        elif EMOTION_INDEX == '2':
            path += 'fear/'
        elif EMOTION_INDEX == '3':
            path += 'happy/'
        elif EMOTION_INDEX == '4':
            path += 'sad/'
        elif EMOTION_INDEX == '5':
            path += 'surprise/'
        elif EMOTION_INDEX == '6':
            path += 'neutral/'

        # Construct complete image path, name and extension
        image_path = path + EMOTION_INDEX + '_' + cate + '_img_' + str(
            COUNTER).zfill(5) + '.jpg'

        # Convert pixels values string to the list of int values
        pixel_values = []
        for pixel in PIXELS_VALUES_STRING.split(' '):
            pixel_values.append(int(pixel))

        # Convert values to the array and reshape it
        # Size 48 x 48 pixels and 1 color channel
        data = np.array(pixel_values)
        data = data.reshape((48, 48, 1)).astype('uint8')

        # Save image
        imageio.imwrite(image_path, data)

        COUNTER += 1
