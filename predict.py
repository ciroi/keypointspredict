from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import os
import cv2
from PIL import Image

from data import get_train_test_set

# 此部分代码针对stage 1中的predict。 是其配套参考代码
# 对于stage3， 唯一的不同在于，需要接收除了pts以外，还有：label与分类loss。

def validPredict(args, trained_model, model, valid_loader):
    model.load_state_dict(torch.load(os.path.join(args.save_directory, trained_model), map_location=torch.device('cpu')))   # , strict=False
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            img = batch['image']
            landmark = batch['landmarks']
            # print('i: ', i)
            # generated
            output_pts, clss = model(img)
            outputs = output_pts.numpy()[0]
            x = list(map(int, outputs[0: len(outputs): 2]))
            y = list(map(int, outputs[1: len(outputs): 2]))
            landmarks_generated = list(zip(x, y))
            # truth
            landmark = landmark.numpy()[0]
            x = list(map(int, landmark[0: len(landmark): 2]))
            y = list(map(int, landmark[1: len(landmark): 2]))
            landmarks_truth = list(zip(x, y))

            img = img.numpy()[0].transpose(1, 2, 0)
            
            img = img * batch['std'][0].numpy() +batch['mean'][0].numpy()
            # 请画出人脸crop以及对应的landmarks
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if clss[0].numpy().argmax() == 1:
                for landmark_truth, landmark_generated in zip(landmarks_truth, landmarks_generated):
                    cv2.circle(img, tuple(landmark_truth), 2, (0, 0, 255), -1)
                    cv2.circle(img, tuple(landmark_generated), 2, (0, 255, 0), -1)

            
            cv2.imshow(str(i), img)
            key = cv2.waitKey()
            if key == 27:
                exit()
            cv2.destroyAllWindows()


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels

def preProcess(img):
    img = cv2.resize(img, (112, 112), 0, 0, cv2.INTER_LINEAR).astype(np.float32)
    imgnorm = channel_norm(img)

    imgnorm = torch.from_numpy(imgnorm).reshape(1, 1, 112, 112)
    return imgnorm, img

def Net(img):
    # face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    face_engine = cv2.CascadeClassifier('D:/opencvlib/opencv/build/etc/haarcascades/'+'haarcascade_frontalface_default.xml')
    faces = face_engine.detectMultiScale(img,scaleFactor=1.3, minSize=(100,100), minNeighbors=5)
    return faces


def predictVideoModel(leftuppt, img, model):
    imgData1, imgData = preProcess(img)
    # generated
    output_pts, clss = model(imgData1)
    outputs = output_pts.numpy()[0]
    x = outputs[0: len(outputs): 2]
    y = outputs[1: len(outputs): 2]

    shape = img.shape
    newshape = imgData.shape
    ratey, ratex = shape[0]/newshape[0], shape[1]/newshape[1]

    x = x*ratex+leftuppt[0]
    y = y*ratey+leftuppt[1]

    x = list(map(int, x))
    y = list(map(int, y))
    landmarks_generated = list(zip(x, y))
    return landmarks_generated


def videoPredict(args, trained_model, model):
    model.load_state_dict(torch.load(os.path.join(args.save_directory, trained_model), map_location=torch.device('cpu')))   # , strict=False
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        # forward pass: compute predicted outputs by passing inputs to the model
        cameraCapture = cv2.VideoCapture(0)
        cv2.namedWindow('Test camera')
        success, frame = cameraCapture.read()
        while success:
            if cv2.waitKey(1) == 27:
                break
            faces = Net(frame)
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            print(len(faces), faces)
            landmarks = []
            if len(faces) == 1:
                faces = faces[0]
                x,y,w,h = faces[0], faces[1], faces[2], faces[3]
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = predictVideoModel((x,y), gray_frame[y:y+h, x:x+w], model)

            if len(landmarks) != 0:
                for landmark_truth in landmarks:
                    cv2.circle(frame, tuple(landmark_truth), 2, (0, 255, 0), -1)

            cv2.imshow("ret", frame)
            success, frame = cameraCapture.read()
        cameraCapture.release()
        cv2.destroyAllWindows()
        

        if cv2.waitKey(1) == 27:
            exit()
