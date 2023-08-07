import os
import cv2
import time
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mrcnn.model import MaskRCNN
from mrcnn.config import Config

from utils import pad_image
from train import RoiDetector


class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "pred_cfg"
    # number of classes (background + rc)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def reshape_image(image):
    # reshape image for ML model -> (1, SIZE, SIZE, 1)
    padded_img = pad_image(image)
    return np.expand_dims(padded_img, 0)


def predict(image):
    '''
    Load model and make prediction on region of interest (ROI)

    Args:
        image: Image to be predicted, full size
    Returns:
        pred: Dictionary containing ROI, class_ids, scores, and mask
    '''
    # load model and make roi prediction
    cfg = PredictionConfig()
    roi = RoiDetector()

    # TODO use model.find_last() to use latest model in load weights
    model = MaskRCNN(mode='inference', model_dir=roi.MODEL_DIR,
                     config=cfg)
    model.load_weights(roi.MODEL_PATH, by_name=True)

    # make prediction
    reshaped_image = reshape_image(image)
    pred = model.detect(reshaped_image, verbose=1)[0]

    # print(pred)
    return pred


def crop_roi(orig_image, new_image, pred):
    '''
    Crop the raw image, based on prediction on resized image.
    Resize the ROI to ful size.

    Args:
        orig_image: Raw image, full size
        new_image: Resized image, smaller size
        pred: prediction of roi coordinates (dictionary)
    Returns:
        crop_img: Cropped image, to be consumed by the next model.
    '''
    scaled_size = 640
    old_size = orig_image.shape[:2]  # old_size is in (height, width) format
    ratio = max(old_size) / float(scaled_size)

    print('Ratio reshape to raw file: {}'.format(ratio))

    # take the first prediction of the roi only
    y1, x1, y2, x2 = np.rint(pred['rois'][0] * ratio).astype(int)
    # plt.imshow(orig_image)
    # plt.imshow(new_image)
    print('max shape: {}'.format(max(orig_image.shape)))
    crop_img = pad_image(orig_image, max(orig_image.shape))[y1:y2,
               x1:x2]
    return crop_img


def get_roi(image_path, plot=False):
    '''
    Crop the raw image, based on prediction on resized image.
    Resize the ROI to ful size.

    Args:
        orig_image: Raw image, full size
        new_image: Resized image, smaller size
        pred: prediction of roi coordinates (dictionary)
    Returns:
        crop_img: Cropped image, to be consumed by the next model.
    '''
    start = time.time()
    raw_image = cv2.imread(image_path)

    if raw_image is None:
        print(f'Image file {image_path} found! Check image path')
        sys.exit()
    else:
        resized = pad_image(raw_image)
        pred = predict(raw_image)

        cropped = crop_roi(raw_image, resized, pred)
        print('Prediction took : {:.2f} seconds'.format(time.time() - start))

        # print(cropped) # array
        if plot == True:
            plt.imshow(cropped)
            plt.show()

        return cropped


if __name__ == '__main__':

    root_dir = Path().resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='Path to image to get ROI',
                        default='data/registrationarea/sample_image2.jpg')
    parser.add_argument('-p', '--plot',
                        help='Plot cropped image of predicted ROI',
                        action='store_true')
    parser.add_argument('-s', '--save-image', help='Save roi image',
                        action='store_true')

    args = parser.parse_args()

    crop = get_roi(image_path=os.path.join(root_dir, args.image),
                   plot=args.plot)
    # save cropped image
    if args.save_image:
        temp_image = os.path.join(root_dir, 'data', 'registrationarea',
                                  'temp_result.jpg')
        cv2.imwrite(temp_image, crop)
