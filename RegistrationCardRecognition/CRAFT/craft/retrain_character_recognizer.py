import keras_ocr
import imgaug
import cv2
import numpy as np
import json
import csv
import os
import time 

import tensorflow as tf

from character_recognizer import create_character_recognizer
from img_utils import pad_img
from filter import filter_corrections
from sklearn.model_selection import train_test_split

from stream import get_corrections, history_to_s3, weights_to_s3, coder_from_s3

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

decoder = coder_from_s3("character_label_decoder")
encoder = coder_from_s3("character_label_encoder")

#DATA SETTINGS
WIDTH = 64
HEIGHT = 64
DIM = (WIDTH,HEIGHT)

#TRAINING SETTINGS
EPOCHS = 50
BATCH_SIZE = 16
def generate_labels(model):
    corrections = get_corrections()
    labels = filter_corrections(corrections, 'gray')
    
    x = []
    y = []

    for img, _, text in labels:
        char_imgs = model.slice_text(img)
        #check len
        if len(text) != len(char_imgs):
            continue

        for char_img, character in zip(char_imgs,text):
            char_img = pad_img(char_img, size=HEIGHT)
            char_img = cv2.resize(char_img, DIM, interpolation = cv2.INTER_AREA)
            label = encoder[character]
            label = int(label)-1
            label_container = np.zeros(len(encoder)).astype(np.float32)
            label_container[label] = 1.0
            label_container = list(label_container)

            x.append(char_img)
            y.append(label_container) #minus 1 to cater 0, which in SSD reserve for bg
    
    x = np.array(x)
    y = np.array(y)
    return x, y

#Initialize OCR Model
def train():
    model = create_character_recognizer()
    x,y = generate_labels(model)
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=60)
    X_train = X_train.reshape(-1,WIDTH,HEIGHT,1)
    X_test = X_test.reshape(-1,WIDTH,HEIGHT,1)

    datagen = ImageDataGenerator(
        rotation_range=0,
        zoom_range = 0,  
        width_shift_range=0.1, 
        height_shift_range=0,
        brightness_range=(0.5, 1.5))
    datagen.fit(X_train)
    
    decayer = ReduceLROnPlateau(monitor='val_loss',
                                patience=5,
                                verbose=1,
                                factor=0.1,
                                min_lr=0.000001)
    earlystopper = EarlyStopping(patience=8, verbose=1)

    model.recognizer.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    train_start = time.time()
    history = model.recognizer.fit(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),
                              epochs = EPOCHS,
                              validation_data = (X_test,y_test),
                              verbose = 1,
                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=[decayer,earlystopper])
    print(f"Training time taken - {time.time()-train_start}")
    print(history.history)
    #store history to S3
    history_to_s3(history.history)
    lowest_loss = min(history.history['val_loss'])
    loss_thresh = 0.5
    if lowest_loss < loss_thresh:    
        #savemodel
        weights_to_s3(model.recognizer, 'recognizer-models-v2')
        print(f"New model saved, current loss - {lowest_loss}")
if __name__ == '__main__':
    start = time.time()
    train()
    print(f"Overall time taken - {time.time() - start}")