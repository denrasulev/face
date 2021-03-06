# -*- coding: utf-8 -*-
"""Facial Expressions Recognition project.

Data preparation module.

Data comes from the Kaggle competition “Challenges in Representation
Learning: Facial Expression Recognition Challenge”:

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

Data consists of 48x48 pixel grayscale images of faces in the form of
arrays with a grayscale value for each pixel.

Each image corresponds to a facial expression in one of seven categories:
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.

This module reads raw data, converts numeric values to images and saves
them to according folders - Training, Validation and Test.

Dataset contains approximately 36K images.

Attributes:
    RAW_DATA (str): Path to the raw data file (fer2013.csv)
    EMOTION_INDEX (list): Numeric index for each facial expression
    PIXELS_VALUES (list): List of grayscale pixels values
    CATEGORY_NAME (list): Category name to split data into Training, Validation
    and Test sets.
"""

import csv

import imageio
import numpy as np

# Image data pre-processing
RAW_DATA = './raw_data/fer2013.csv'

EMOTION_INDEX = []
PIXELS_VALUES = []
CATEGORY_NAME = []

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
        PIXELS_VALUES = row[1]
        CATEGORY_NAME = row[2]
        path = ''
        cate = ''

        # Sort images by folders
        # Create base path according to the Category of an image
        if CATEGORY_NAME == 'Training':
            path = './data/train/'
            cate = 'train'
        elif CATEGORY_NAME == 'PublicTest':
            path = './data/valid/'
            cate = 'valid'
        elif CATEGORY_NAME == 'PrivateTest':
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
        for pixel in PIXELS_VALUES.split(' '):
            pixel_values.append(int(pixel))

        # Convert values to the array and reshape it
        # Size 48 x 48 pixels and 1 color channel
        data = np.array(pixel_values)
        data = data.reshape((48, 48, 1)).astype('uint8')

        # Save image
        imageio.imwrite(image_path, data)

        COUNTER += 1
