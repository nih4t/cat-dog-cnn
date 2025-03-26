import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

#Image Augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize to 0-1
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Zoom in/out by 20%
    horizontal_flip=True,    # Randomly flip horizontally
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')
