import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# preprocessing the images and the corresponding masks
def preprocess_image(path, target_size=(224, 224)):
    img = load_img(path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array) 
    return img_array

def preprocess_mask(path, target_size=(224, 224)):
    mask = load_img(path, target_size=target_size, color_mode="grayscale")
    mask_array = img_to_array(mask)
    mask_array = mask_array[:, :, 0]  # single channel
    mask_array = mask_array / 255.0  # Normalizing between [0, 1]
    mask_array[mask_array > 0] = 1  # Binarizing the mask: Setting all non-zero values to 1
    mask_array = np.expand_dims(mask_array, axis=-1)  # Adding a channel dimension so it can be used with the model
    return mask_array


def load_images_from_directory(directory, preprocess_function, target_size=(224, 224)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  
            file_path = os.path.join(directory, filename)
            processed_image = preprocess_function(file_path, target_size=target_size)
            images.append(processed_image)
    return images

#loding the dataset
image_path = 'dataset/images/'
mask_path = 'dataset/masks/'

images = load_images_from_directory(image_path, preprocess_image)
masks = load_images_from_directory(mask_path, preprocess_mask)


# train and test split
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
# Converting files in lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import TensorBoard
import datetime

# Defining the UNET Architecture with VGG19 Backbone
def conv_block(input_tensor, num_filters):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # second layer
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x



def decoder_block(input_tensor, concat_tensor, num_filters):
    """Function to add one up-sampling layer and one concatenate layer"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    x = layers.concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x


def unet_vgg19(input_shape, num_classes):
    """Build a U-Net with a VGG19 backbone"""
    # Encoder: VGG19
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg19.trainable = False  # Freeze the VGG19 layers

    # Skip connections from the encoder
    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    # Bridge
    b1 = vgg19.get_layer("block5_conv4").output

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=vgg19.input, outputs=outputs)

    return model



input_shape = (224, 224, 3)  # size of the input image to the UNET model
num_classes = 1  # No classes - 1 : rail

unet_model = unet_vgg19(input_shape, num_classes)

unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
unet_model.summary()

# logging tensorboard data
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# early stopping
early_stopper = EarlyStopping(monitor='val_loss', patience=12, verbose=1)
# run the model
history = unet_model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[early_stopper, tensorboard_callback]
)
    
#saving the model
unet_model.save('models/unet_model.h5')

import matplotlib.pyplot as plt

# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig('plots/accuracy_plot.pdf', format='pdf')

# Plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('plots/loss_plot.pdf', format='pdf')
plt.show()
