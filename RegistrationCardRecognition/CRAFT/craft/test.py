import cv2
import os
import time
import json
import random

from network import create_craft
from filter import get_reg_info, remove_noise, get_rc_info_textbox, sort_prediction, sort_detection, get_xy_minmax
from character_recognizer import create_character_recognizer

temp_folder = '../../data/sample/'

def test():
    """
    Run test prediction on images stored in EC2
    """
    IMAGE_NUM = 5

    img_path = '../../data/registrationcard/train/img/'
    image_filenames = os.listdir(img_path)

    start = IMAGE_NUM
    end = len(image_filenames)

    end_idx = random.randint(start,end)
    start_idx= end_idx-IMAGE_NUM

    images = [ cv2.imread(f"{img_path}{image_name}") for image_name in image_filenames[start_idx:end_idx] ]
    craft = create_craft()
    #Test Pipeline V1
    # model = craft.pipeline
    # all_time = time.time()
    # for image in images:
    #     start_time = time.time()
    #     predictions = model.recognize([image])
    #     # print(predictions)
    #     pred_texts = [' '.join([character for character,_ in sort_prediction(prediction, image)]) for prediction, image in zip(predictions, [image]) ]
    #     reg_infos = [{'text':remove_noise(pred_text.upper()) , **get_reg_info(pred_text)} for pred_text in pred_texts]
    #     print(json.dumps(reg_infos, indent=2))
    #     textboxes = [get_rc_info_textbox(reg_info, prediction) for reg_info, prediction in zip(reg_infos,predictions)]
    #     print(textboxes)
    #     print("---  %s seconds  ---" % ((time.time() - start_time)))
    # print("--- %s images execution : %s seconds  ---" % (IMAGE_NUM, (time.time() - all_time)) )

    # Test Pipeline V2
    model = craft.detector
    detection_groups = model.detect(images)
    character_recognizer = create_character_recognizer()
    for image, detections in zip(images, detection_groups):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sorted_detections = sort_detection(detections, image)
        texts = []
        for detection in sorted_detections:
            minmax = get_xy_minmax(detection)
            test_img = image[minmax['y1']:minmax['y2'],minmax['x1']:minmax['x2']]
            if test_img is not None:
                # print(minmax)
                # print(test_img)
                text = character_recognizer.predict_text(test_img)
                texts.append(text)
        print(' '.join(texts))

if __name__ == '__main__':
    test()
