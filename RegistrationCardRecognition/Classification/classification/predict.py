import cv2
import os
import time
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model


def classify(model_path, image_path):
    '''
    Make prediction whether an image is RC or not

    Args:
        model_path: model to be used for prediction
        image_path: path for image to be predicted
    Returns:
        True/False
    '''
    IMG_SIZE = 150
    print(f'Loading model from {model_path}')
    model = load_model(model_path)

    print(f'Loading image from {image_path}')
    image = cv2.imread(image_path)

    # reshape image
    image_reshaped = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    predict_image = image_reshaped.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    result = model.predict(predict_image, steps=1)
    print(f'RC Probability: {1-result[0][0]:.2f}')

    if result >= 0.5:
        print('not rc')
        return False
    else:
        print('RC')
        return True


if __name__ == "__main__":
    root_dir = Path().resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='Path to image to get ROI',
                        default='data/classifier_data/sample_rc1.jpg')
    parser.add_argument('-m', '--model', help='Path to model',
                        default='data/classifier_data/model/classifier_model.h5')

    args = parser.parse_args()
    start = time.time()
    classify(image_path=os.path.join(root_dir, args.image),
             model_path=os.path.join(root_dir, args.model))
    print('Prediction took : {:.2f} seconds'.format(time.time() - start))
