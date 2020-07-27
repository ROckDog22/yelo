# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : 汪逢生
# @FILE     : model.py
# @Time     : 2020/7/27 13:36
# @Software : PyCharm


from keras.backend import clear_session
from keras.optimizers import SGD
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import initializers, regularizers
import train_model.mytrain.callbacks as callbacks
clear_session()
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator


# Config
height = 299
width = height
weights_file = "weights.best_mobilenet" + str(height) + ".hdf5"
NUM_CLASSES = 2
GENERATOR_BATCH_SIZE = 10
TOTAL_EPOCHS = 1
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 10
BASE_DIR = 'C:\\Users\\lenovo\\Desktop\\nsfw_model-master\\mytrain\\data'

# 数据预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should not be modified
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')

def create_generators(height, width):
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=GENERATOR_BATCH_SIZE
    )

    validation_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=GENERATOR_BATCH_SIZE
    )

    return[train_generator, validation_generator]
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(sess)  # set this Tens

# orFlow session as the default session for Keras



# conv_base = MobileNetV2(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(height, width, 3)
# )
conv_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(height, width, 3)
)

conv_base.trainable = False
x = conv_base.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x)
x = Dropout(0.5)(x)
# Essential to have another layer for better accuracy
x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
x = Dropout(0.25)(x)
predictions = Dense(NUM_CLASSES,  kernel_initializer="glorot_uniform", activation='softmax')(x)


model = Model(inputs = conv_base.input, outputs=predictions)

# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

# Get all model callbacks
callbacks_list = callbacks.make_callbacks(weights_file)

opt = SGD(momentum=.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


train_generator, validation_generator = create_generators(height, width)

print('Start training!')
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=TOTAL_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS
)

# Save it for later
print('Saving Model')
model.save("nsfw_mobilenet2." + str(width) + "1111x" + str(height) + ".h5")
