"""
Invoke all functions from this file

"""

# Importing all necessary modules
from modules import config
from modules import req

# Importing necessary packages
import tensorflow as tf

# Loading the necessary models
track_model = tf.keras.models.load_model(config.track_model, custom_objects={'combined_loss': req.combined_loss})

# call the function to load the input video and process it
req.process_images(config.input_images, config.binary_output_path, track_model)
