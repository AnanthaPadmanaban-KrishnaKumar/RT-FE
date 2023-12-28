"""
Defining all the necessary functions here and calling from the main.py
"""

# importing all necessary packages
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# importing all necessary modules
from modules import config
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 

"""
Custom loss function for track segmentation
It calculates two types of losses:
1. weighted_binary_crossentropy
2. focal_loss
3. combines both and gives out a combined loss
"""

"""
weighted_binary_crossentropy
"""
def weighted_binary_crossentropy(y_true, y_pred):
    w0 = 0.2
    w1 = 0.8
    bce = w1 * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) + \
          w0 * (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return -tf.reduce_mean(bce)

"""
focal loss
"""
def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    focal_loss_value = (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) + \
                       alpha * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return -tf.reduce_mean(focal_loss_value)

"""
Combining both weighted binary crossentropy and focal loss
"""
def combined_loss(y_true, y_pred, alpha=0.8, gamma=2.0, w0=0.2, w1=0.8):
    w_bce = weighted_binary_crossentropy(y_true, y_pred)
    f_loss = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    # Combine the losses as per your requirement, for example, an average.
    combined = (w_bce + f_loss) / 2
    return combined

"""
Preprocessing function to preprocess the frame before passing it into Unet seg model
"""
def preprocess_image(frame, target_size=(576, 896)):
    img = Image.fromarray(frame)
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0 
    return img_array

"""
Postprocessing function to postprocess the frame after receiving it from Unet seg model
"""
def postprocess_mask(pred_mask,orginal_size):
    pred_mask = pred_mask[0] 
    pred_mask = pred_mask[:, :, 0] 
    pred_mask = Image.fromarray(pred_mask)
    pred_mask = pred_mask.resize(orginal_size, Image.NEAREST)
    pred_mask = tf.cast(pred_mask, tf.float32) 
    threshold = 0.4
    pred_mask = tf.where(pred_mask > threshold, 1, 0) 
    return np.array(pred_mask)

"""
main function to process all images in the given folder
"""

def process_images(input_folder, binary_path, track_model):
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()
    
    saved_frame_count = 0

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
       
        # Check if the image was loaded successfully
        if frame is None:
            print(f"Error opening image file: {image_file}")
            continue
        
        # Process the image by sending it to the preprocess and postprocess function
        processed_frame = preprocess_image(frame, target_size=(576, 896))
        pred_mask = track_model.predict(processed_frame)
        output_mask = postprocess_mask(pred_mask, (frame.shape[1], frame.shape[0]))

       # Convert mask to 3 channels
        output_mask_3ch = np.repeat(output_mask[:, :, np.newaxis], 3, axis=2)
        output_mask_3ch = (output_mask_3ch * 255).astype(np.uint8)

        output_mask = np.expand_dims(output_mask, axis=-1) * 255
        # save the binary mask
        binary_output_filename = os.path.join(binary_path, f'frame_{saved_frame_count:05d}.png')
        cv2.imwrite(binary_output_filename, output_mask)
        #cv2.imwrite(clustered_image_filename, clustered_image)
        saved_frame_count += 1