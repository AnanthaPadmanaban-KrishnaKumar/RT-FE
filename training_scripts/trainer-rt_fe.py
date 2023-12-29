"""
Use this script to train the rail feature extraction model.
We load the dataset from aws s3 bucket in tfrecord format.

"""


import io
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, Activation, BatchNormalization, UpSampling2D, Add, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import datetime
import matplotlib.pyplot as plt

from keras import backend as K
K.clear_session()

import boto3
from botocore.exceptions import NoCredentialsError

aws_access_key_id = ''
aws_secret_access_key = ''
region_name = ''  

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

s3_client = session.client('s3')

try:
    response = s3_client.list_buckets()
    print(response)
except NoCredentialsError:
    print("Credentials are not available or invalid.")


def _parse_function(tfrecord_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(tfrecord_proto, feature_description)
    
    image = tf.io.decode_png(parsed_features['image'], channels=3)
    mask = tf.io.decode_png(parsed_features['mask'], channels=1)

    image = tf.image.resize(image, [896, 576], method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, [896, 576], method=tf.image.ResizeMethod.AREA)

    image = tf.cast(image, tf.float32) / 255.0
    
    mask = tf.cast(mask, tf.float32) / 255.0
    threshold = 0.1  
    mask = tf.where(mask > threshold, x=tf.ones_like(mask), y=tf.zeros_like(mask))
    
    return image, mask

def load_dataset_from_tfrecord(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    return dataset.map(_parse_function)
s3_tfrecord_path = 'halfdata.tfrecord'
dataset = load_dataset_from_tfrecord(s3_tfrecord_path)
for images, masks in dataset.take(5):  
    print('Image shape:', images.numpy().shape)
    print('Mask shape:', masks.numpy().shape)

    import matplotlib.pyplot as plt

def visualize_image_and_mask(image, mask):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask[:, :, 0], cmap='gray')  
    plt.axis('off')

    plt.show()
for images, masks in dataset.take(6): 
    visualize_image_and_mask(images.numpy(), masks.numpy())


SHUFFLE_BUFFER_SIZE = 300  
BATCH_SIZE = 4
DATASET_SIZE = 3000  

dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat()

TRAIN_SIZE = int(0.8 * DATASET_SIZE)
VAL_SIZE = DATASET_SIZE - TRAIN_SIZE

train_dataset = dataset.take(TRAIN_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = dataset.skip(TRAIN_SIZE).take(VAL_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

def visualize_image_and_mask(image, mask):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask[:, :, 0], cmap='gray') 
    plt.axis('off')
    plt.show()

for images, masks in dataset.take(5):
    visualize_image_and_mask(images.numpy(), masks.numpy())


print(TRAIN_SIZE)
print(len(train_dataset))
print(len(val_dataset))

for images, masks in train_dataset.take(1):
    print('Images batch shape:', images.shape)
    print('Masks batch shape:', masks.shape)


def weighted_binary_crossentropy(y_true, y_pred):
    w0 = 0.2
    w1 = 0.8
    bce = w1 * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) + \
          w0 * (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return -tf.reduce_mean(bce)

# Focal Loss
def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    focal_loss_value = (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) + \
                       alpha * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return -tf.reduce_mean(focal_loss_value)

# Combined Loss
def combined_loss(y_true, y_pred, alpha=0.8, gamma=2.0, w0=0.2, w1=0.8):
    w_bce = weighted_binary_crossentropy(y_true, y_pred)
    f_loss = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    combined = (w_bce + f_loss) / 2
    return combined


def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    x = concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x

def unet_vgg19(input_shape, num_classes):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg19.trainable = False
    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output
    b1 = vgg19.get_layer("block5_conv4").output
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=vgg19.input, outputs=outputs)
    return model

input_shape = (896, 576, 3)
num_classes = 1
with tf.device('/gpu:0'):
     model = unet_vgg19(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=combined_loss, metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=16, 
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=80,
    validation_data=val_dataset,
    verbose=1,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback]
)
model.save('unet_model_896_576_weighted_loss.h5')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig('plots/accuracy_plot.pdf', format='pdf')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('plots/loss_plot.pdf', format='pdf')

print(history.history.keys())

val_loss, val_accuracy = model.evaluate(val_dataset)
print(val_loss, val_accuracy)

s3_client.upload