import os
from xml.etree import ElementTree

import numpy as np
from mrcnn.config import Config
from mrcnn.utils import Dataset


class RCDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "rc")

        print('#' * 35)
        print('Loading data sets...')

        if is_train:
            # define data locations
            images_dir = os.path.join(dataset_dir, 'train', 'image/')
            annotations_dir = os.path.join(dataset_dir, 'train', 'annot/')

        if not is_train:
            # define data locations
            images_dir = os.path.join(dataset_dir, 'test', 'image/')
            annotations_dir = os.path.join(dataset_dir, 'test', 'annot/')

        image_count = len(os.listdir(images_dir))
        print('Number of image found: {}'.format(image_count))

        # find all images, split for train and test
        for filename, i in zip(os.listdir(images_dir),
                               range(len(os.listdir(images_dir)))):
            # extract image id
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path,
                           annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('rc'))
        return masks, np.asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class RCConfig(Config):
    # define the name of the configuration
    NAME = "rc_cfg"
    # number of classes (background + rc)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch/number of photos in the dataset
    STEPS_PER_EPOCH = 3
    # batch size
    IMAGES_PER_GPU = 2
