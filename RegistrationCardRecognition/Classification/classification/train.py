from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, \
    TensorBoard
from pathlib import Path
from datetime import datetime
import os
import time
import argparse
import matplotlib.pyplot as plt

from create_dataset import create_dataset


class RCClassification():

    def __init__(self, STORAGE_DIR=None):

        self.ROOT_DIR = Path().resolve().parent.parent

        if STORAGE_DIR is not None:
            self.STORAGE_DIR = STORAGE_DIR
        else:
            self.STORAGE_DIR = os.path.join(self.ROOT_DIR, 'data',
                                            'classifier_data')
        self.MODEL_DIR = os.path.join(self.STORAGE_DIR, 'model')
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, 'classifier_model.h5')

        self.IMG_SIZE = 150

        # create directory if it does not exist
        self.__make_dir(self.STORAGE_DIR)
        self.__make_dir(self.MODEL_DIR)

    def __make_dir(self, dir):
        if not os.path.exists(dir):
            try:
                os.mkdir(dir)
            except OSError:
                print("Directory %s creation failed" % dir)

    def create_model(self):
        '''
        Create model for training

        Args:
            None
        Returns:
            model:  model to be trained
        '''

        base_model = VGG16(weights='imagenet', include_top=False,
                           input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3))

        # freeze vgg layer
        for layer in base_model.layers:
            layer.trainable = False

        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        add_model.add(Dense(256, activation='relu'))
        add_model.add(Dense(1, activation='sigmoid'))

        model = Model(inputs=base_model.input,
                      outputs=add_model(base_model.output))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-4),
                      metrics=['accuracy'])

        print(model.summary())

        return model

    def train_model(self, epochs=3, val_step=5, batch=32, plot=False):
        '''
        Train model for RC classification, save best model in data folder

        Args:
            epochs: number of epoch for training
            val_step: number of steps for validation
            batch: number of images per batch
            plot: plot model log after training finished
        Returns:
            None
        '''
        # load train and val data generator
        train_generator, val_generator, _ = create_dataset(batch=32, size=150,
                                                           shift=0.1,
                                                           rotation=5)

        # load model
        model = self.create_model()

        checkpoint = ModelCheckpoint(self.MODEL_PATH, monitor='val_accuracy',
                                     verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto',
                                     save_freq=1)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0,
                              patience=20, verbose=1, mode='auto')
        logdir = os.path.join(self.MODEL_DIR, 'logs',
                              datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(log_dir=logdir,
                                  histogram_freq=0, write_graph=True,
                                  write_images=False)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator.classes) // batch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_step,
            callbacks=[checkpoint, early, tensorboard])

        model.save(self.MODEL_PATH)
        print(f'Model saved! : {self.MODEL_PATH}')

        if plot:
            plt.figure(figsize=(16, 6))
            plt.subplot(1, 2, 1)
            plt.plot(history.history["accuracy"], label='train')
            plt.plot(history.history['val_accuracy'], label='test')
            plt.title("model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.title("loss")
            plt.ylabel("loss")
            plt.xlabel("Epoch")
            plt.ylim(0, 1)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='Number of epochs for training',
                        default=1, type=int)
    parser.add_argument('-v', '--val-step', help='Number of validation steps',
                        default=2, type=int)
    parser.add_argument('-b', '--batch', help='Image per batch in training',
                        default=32, type=int)
    parser.add_argument('-p', '--plot', help='Show training logs',
                        action='store_true')

    args = parser.parse_args()

    start = time.time()
    rc_class = RCClassification()
    rc_class.train_model(epochs=args.epochs, val_step=args.val_step,
                         batch=args.batch, plot=args.plot)

    print('Training RC classifier took : {:.2f} seconds'.format(
        time.time() - start))
