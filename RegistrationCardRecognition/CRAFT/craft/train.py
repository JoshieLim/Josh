import keras_ocr
import imgaug
import cv2

import json
import csv
import os

import tensorflow as tf

from network import create_craft
from sklearn.model_selection import train_test_split

with open("../../data/registrationtext/label/character_label_decoder.json") as file:
    decoder = json.load(file)

def generate_labels():
    label_path = '../../data/craft/label/craft_filtered.csv'
    labels = []
    with open(label_path, 'r') as label_file:
            file_reader = csv.DictReader(label_file, delimiter=',')
            for row in file_reader:
                labels.append((f"../../data/registrationtext/img/{row['text_image_name']}",None,row['character']))

    ratios = 0.9
    train_labels = labels[:int(ratios*len(labels))]
    test_labels = labels[int(ratios*len(labels)):]

    return train_labels, test_labels

#Initialize OCR Model
def train():
    craft = create_craft()
    recognizer = craft.recognizer

    batch_size = 32

    augmenter = imgaug.augmenters.Sequential([
        imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
    ])

    train_labels, validation_labels = generate_labels()
    train_labels, validation_labels = train_test_split(train_labels, test_size=0.2, random_state=42)
    (training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
        (
            keras_ocr.datasets.get_recognizer_image_generator(
                labels=labels,
                height=recognizer.model.input_shape[1],
                width=recognizer.model.input_shape[2],
                alphabet=recognizer.alphabet,
                augmenter=augmenter
            ),
            len(labels) // batch_size
        ) for labels, augmenter in [(train_labels, augmenter), (validation_labels, None)]
    ]


    training_gen, validation_gen = [
        recognizer.get_batch_generator(
            image_generator=image_generator,
            batch_size=batch_size
        )
        for image_generator in [training_image_gen, validation_image_gen]
    ]

    #Training

    train_folder = '../../data/craft/train/craft/'
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(f"{train_folder}checkpoints", exist_ok=True)
    os.makedirs(f"{train_folder}logs", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(f'{train_folder}/checkpoints/recognizer_registrationtext.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.CSVLogger(f'{train_folder}recognizer_registrationtext.csv'),
        tf.keras.callbacks.TensorBoard(log_dir=f'{train_folder}logs', write_images=False, profile_batch=100000000)
    ]
    recognizer.training_model.fit_generator(
        generator=training_gen,
        steps_per_epoch=training_steps,
        validation_steps=validation_steps,
        validation_data=validation_gen,
        callbacks=callbacks,
        epochs=1,
    )

if __name__ == '__main__':
    train()