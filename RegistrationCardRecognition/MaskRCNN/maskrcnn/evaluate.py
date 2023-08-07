import os
import time
from datetime import timedelta

import numpy as np
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.utils import compute_ap
from mrcnn.config import Config

from config import  RCDataset
from train import RoiDetector

class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "pred_cfg"
    # number of classes (background + rc)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def evaluate_model(dataset, model, cfg):
    '''
    Evaluate model performance on test data.
    Check formula here:
    https://medium.com/@jonathan_hui/map-mean-average-precision-\for-object-detection-45c121a31173

    Args:
        dataset: Dataset to be tested. Train or test
        model: Machine learning model (.h5)
        cfg: prediction config
    Returns:
        mAP, mean average precision [0-1].
    '''
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        pred = model.detect(sample, verbose=0)
        # extract results for first sample
        r = pred[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                 r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = np.mean(APs)
    print('mAP for the test data set: {:.2f}'.format(mAP))
    return mAP


if __name__ == '__main__':
    start = time.time()
    # load config for prediction
    cfg = PredictionConfig()
    roi = RoiDetector()

    # load model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model.load_weights(roi.MODEL_PATH, by_name=True)

    # prepare dataset
    test_set = RCDataset()
    test_set.load_dataset(roi.STORAGE_DIR, is_train=False)
    test_set.prepare()

    test_mAP = evaluate_model(test_set, model, cfg)
    duration = time.time() - start
    print(f'Evaluation took : {timedelta(seconds=duration)} h:m:s')
