import cv2
import numpy as np
import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load FER-2013 dataset using image folders with augmentation
def load_fer2013_dataset():
    # Training data augmentation
    training_data_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Testing/Validation data - only rescaling
    validation_data_generator = ImageDataGenerator(rescale=1./255)

    # Load training images
    training_generator = training_data_generator.flow_from_directory(
        'data/FER-2013/train',
        target_size=config.IMG_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # Load validation images
    validation_generator = validation_data_generator.flow_from_directory(
        'data/FER-2013/test',
        target_size=config.IMG_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    return training_generator, validation_generator

# ✅ Preprocess a single grayscale face image for real-time inference
def preprocess_single_face(face_image):
    """Resize and normalize a grayscale face image for model inference."""
    resized_face = cv2.resize(face_image, config.IMG_SIZE)
    normalized_face = resized_face.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(normalized_face, -1), axis=0)
