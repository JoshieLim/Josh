import os
import time
import argparse
import shutil
from pathlib import Path
from datetime import timedelta

from keras.callbacks import ModelCheckpoint
from mrcnn.model import MaskRCNN
from mrcnn import utils

from config import RCConfig, RCDataset


class RoiDetector():

    def __init__(self, STORAGE_DIR=None):
        self.ROOT_DIR = Path().resolve().parent.parent
        # self.ROOT_DIR = Path().resolve()

        if STORAGE_DIR is not None:
            self.STORAGE_DIR = STORAGE_DIR
        else:
            self.STORAGE_DIR = os.path.join(self.ROOT_DIR, 'data',
                                            'registrationarea')

        self.MODEL_DIR = os.path.join(self.STORAGE_DIR, 'model')
        self.CHECKPOINT = os.path.join(self.STORAGE_DIR, 'checkpoint')
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, 'mask_rcnn_latest.h5')

        # create directory if it does not exist
        self.__make_dir(self.STORAGE_DIR)
        self.__make_dir(self.MODEL_DIR)

    def __make_dir(self, dir):
        if not os.path.exists(dir):
            try:
                os.mkdir(dir)
            except OSError:
                print("Creation of the directory %s failed" % dir)

    def confirmation(self, question):
        reply = str(input(question + ' (Y/n): ')).strip()
        try:
            if reply[0] == 'Y':
                return True
            elif reply[0] == 'n':
                return False
            else:
                print('Invalid Input')
                return self.confirmation(question)
        except Exception as error:
            print("Please enter valid inputs")
            print(error)

    def train_roi_model(self, epochs=5, step_per_epoch=3):
        '''
        Train model for detecting region of interest (ROI) for rc card, save best model in data folder

        Args:
            epochs: number of epochs for training
            step_per_epoch: number of steps for per epoch
        Returns:
            None
        '''
        # load dataset
        question = 'Delete checkpoint after training?'
        proceed = self.confirmation(question)

        train_set = RCDataset()
        train_set.load_dataset(self.STORAGE_DIR, is_train=True)
        train_set.prepare()
        print('Train dataset size: %d' % len(train_set.image_ids))

        test_set = RCDataset()
        test_set.load_dataset(self.STORAGE_DIR, is_train=False)
        test_set.prepare()
        print('Test dataset size: %d' % len(test_set.image_ids))

        # modify config
        config = RCConfig()

        config.STEPS_PER_EPOCH = step_per_epoch
        config.display()

        # model_dir = save location
        model = MaskRCNN(mode='training',
                         model_dir=os.path.join(self.CHECKPOINT),
                         config=config)

        # check existence of latest model
        latest_model = self.MODEL_PATH

        if not os.path.exists(self.MODEL_PATH):
            print('No previous training found! Using coco weights')
            # Local path to trained weights file
            COCO_MODEL_PATH = os.path.join(self.MODEL_DIR, "mask_rcnn_coco.h5")
            # Download COCO trained weights from Releases if needed
            if not os.path.exists(COCO_MODEL_PATH):
                print('Coco weight not found! Downloading...')
                utils.download_trained_weights(COCO_MODEL_PATH)
            latest_model = COCO_MODEL_PATH

            print('*' * 15, ' Using Coco weight! ', '*' * 15)
            model.load_weights(latest_model, by_name=True,
                               exclude=["mrcnn_class_logits",
                                        "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        else:
            print('*' * 15, f' Using latest model! ', '*' * 15)
            # model.find_last()
            model.load_weights(latest_model, by_name=True,
                               exclude=["mrcnn_class_logits",
                                        "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])

        # add custom callback here, in a list
        save_best = [ModelCheckpoint(
            filepath=self.MODEL_PATH,
            save_best_only=True)]

        # train model
        model.train(train_set, test_set,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs, layers='heads',
                    custom_callbacks=save_best)

        model.keras_model.save_weights(self.MODEL_PATH)
        print(f'Model saved at {self.MODEL_PATH}')

        if proceed:
            ckpt_dir = os.path.join(self.MODEL_DIR, 'checkpoint')
            print(f'deleting {ckpt_dir}')
            shutil.rmtree(ckpt_dir)
        else:
            pass


if __name__ == '__main__':
    start = time.time()
    roi = RoiDetector()

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='Number of epochs for training')
    parser.add_argument('-s', '--step-per-epoch', default=3, type=int,
                        help='Number of step per epoch for training')

    args = parser.parse_args()

    roi.train_roi_model(epochs=args.epochs,
                        step_per_epoch=args.step_per_epoch, )

    duration = time.time() - start
    print(f'Training MaskRCNN took : {timedelta(seconds=duration)} h:m:s')