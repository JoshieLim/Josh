import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_dataset(batch=32, size=299, shift=0.1, rotation=5):
    '''
    Create dataset generator for train, validataion and test

    Args:
        batch: number of image per batch
        size: image size for model (vgg=150, inception=299)
        shift: vertical and horizontal image shift for augmentation
        rotation: image rotation for augmentation
    Returns:
        train_generator: generator object for training data
        val_generator: generator object for validation data
        test_generator: generator object for test data
    '''
    ROOT_DIR = Path().resolve().parent.parent
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'registrationcard')

    TRAINDIR = os.path.join(DATA_DIR, 'train')
    TESTDIR = os.path.join(DATA_DIR, 'test')

    print(TRAINDIR)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=rotation,
                                       width_shift_range=shift,
                                       height_shift_range=shift,
                                       fill_mode="constant",
                                       validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAINDIR, target_size=(size, size),
        class_mode='binary', color_mode='rgb', batch_size=batch,
        subset='training', classes=['img', 'NonRC'])

    val_generator = train_datagen.flow_from_directory(
        TRAINDIR, target_size=(size, size),
        class_mode='binary', color_mode='rgb', batch_size=batch,
        subset='validation', classes=['img', 'NonRC'])

    test_generator = test_datagen.flow_from_directory(
        TESTDIR, target_size=(size, size),
        class_mode='binary', color_mode='rgb', batch_size=batch,
        classes=['img', 'NonRC'])

    return train_generator, val_generator, test_generator
