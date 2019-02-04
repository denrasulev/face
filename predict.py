import io

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


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


def get_model(path):
    gm_model = NeuralNet()
    gm_model.load_state_dict(torch.load(path, map_location='cpu'))
    gm_model.eval()

    return gm_model


def get_tensor(image_bytes):
    my_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(io.BytesIO(image_bytes))

    return my_transform(image).unsqueese(0)


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = get_model('model.pt')
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.forward(get_tensor(roi))

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', fr)

        return jpeg.tobytes()


class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_file):
        # load model from file
        self.prediction = None
        self.loaded_model = get_model(model_file)

    def predict_emotion(self, img):
        self.prediction = self.loaded_model.forward(get_tensor(img))
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.prediction)]
