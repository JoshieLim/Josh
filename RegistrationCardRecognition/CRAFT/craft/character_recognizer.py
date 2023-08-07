import cv2
import os
import json
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from stream import weights_from_s3, coder_from_s3
from img_utils import pad_img

import math

WIDTH = 64
HEIGHT = 64
DIM = (WIDTH,HEIGHT)

CONTOUR_THRESHOLD = 160
BLUR_THERSHOLD = 7
BIN_THRESHOLD = 125
ZERO_THRESHOLD = 0.06

class CharacterRecognizer():
    def __init__(self):
        self.decoder = coder_from_s3('character_label_decoder')
        self.target_size = len(self.decoder.keys())
        self.model = self.build_model(self.target_size)
        weights = weights_from_s3('recognizer-models-v2')
        self.model.set_weights(weights)

        self.recognizer = self.model

    def build_model(self, target_size):
        model = Sequential()

        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                        activation ='relu', 
                        input_shape = (WIDTH,HEIGHT,1)))
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                        activation ='relu'))
        model.add(BatchNormalization(momentum=0.15))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                        activation ='relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                        activation ='relu'))
        model.add(BatchNormalization(momentum=0.15))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                        activation ='relu', input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                        activation ='relu'))
        model.add(BatchNormalization(momentum=.15))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.4))
        model.add(Dense(target_size, activation='softmax'))

        model.summary()

        return model


    def predict_character(self, img):
        img = img.reshape(1,WIDTH,HEIGHT,1)
        img = 255-img
        img = img.astype(np.float32)
        pred = self.recognizer.predict(img)
        index = np.argmax(pred)
        score = pred[0][index]
        
        return self.decoder[str(index+1)], score

    def is_background(self, line, zero_threshold = 0.075):
        """
        Check whether the line of array is a background/noise or character
        """
        height = len(line)
        nonzero = len(np.nonzero(line)[0]) 
        res = (nonzero/ height) <= zero_threshold
        return res

    def slice_text(self, img):
        """
        Group character in text
        Parameters:
            img: gray image array in 2d list
        Return:
            characters: list of array containing image character
        """
        ori_img = img.copy()
        blurred = cv2.medianBlur(ori_img,BLUR_THERSHOLD)
        ret, binary_img = cv2.threshold(ori_img, BIN_THRESHOLD,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        h,w = binary_img.shape
        #minimum character pixel length threshoold
        min_cpl = 0
        pad=3
        characters = []
        character =[]
        for i in range(0,w):
            line = binary_img[:,i].tolist()
            if self.is_background(line):
                if len(character) > min_cpl:
                    #left padding
                    start = i-len(character)
                    start_pad =  ori_img[:, max(0,start-pad):start].transpose().tolist()
                    if start > 0:
                        character = start_pad + character

                    # right padding
                    end = min(w,i+pad)
                    end_pad = ori_img[:,i:end].transpose().tolist()
                    if i != end:
                        character = character + end_pad

                    character =[ np.array(col).astype(np.uint8) for col in character]
                    rotated = np.array(character).transpose()
                    characters.append(rotated)

                    #reset
                    character =[]
            else:
                character.append(ori_img[:,i].tolist())
        if len(character) > min_cpl:
            character =[ np.array(col).astype(np.uint8) for col in character]
            rotated = np.array(character).transpose()
            characters.append(rotated)
        return characters

    def predict_text(self, img, threshold = 0.9):
        characters = self.slice_text(img)
        plot_size = len(characters)+2
        pos=1
        text = ''
        for character in characters:
            character = pad_img(character)
            resized_img = cv2.resize(character, DIM, interpolation = cv2.INTER_AREA)
            predicted_letter, score = self.predict_character(resized_img)
            score = float("{:.2f}".format(score))
            if score > threshold:#predicted_letter != 'Other':
                text = text + predicted_letter
                pos = pos +1
        return text

def create_character_recognizer():
    return CharacterRecognizer()