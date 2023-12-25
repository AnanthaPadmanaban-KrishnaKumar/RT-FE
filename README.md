# Rail_Track_Foreign_Object_Detecion - RT-FODS

![Alt text for your image](URL_TO_YOUR_IMAGE)


## Introduction
This project leverages cutting-edge deep learning techniques to detect and categorize anomalies on rail tracks. By integrating binary semantic segmentation and self-supervised learning, our system is uniquely capable of identifying potential hazards without extensive manual labeling of training data.

## Project Description

### Stage 1: Isolation of Rails via Binary Semantic Segmentation
The first stage is focused on the isolation of rails from the surrounding environment. This is achieved through binary semantic segmentation, which sets the stage for the detailed analysis that follows.

### Stage 2: Anomaly Detection with Self-Supervised Autoencoders
The second stage utilizes self-supervised learning with autoencoders to detect anomalies, such as obstructions or damage to the rails. This stage is crucial for identifying irregularities like coal, dirt, and unexpected obstacles on the tracks.

## Key Contributions

- **Advanced Anomaly Detection**: The system identifies rail track anomalies with high precision, ensuring that no false positives or negatives affect the operational efficiency.
- **Reduced Need for Labeled Data**: Our methodology negates the necessity for a vast corpus of labeled anomaly data, which is typically challenging to procure.
- **Technological Progression**: This project draws upon the rich history of semantic segmentation, employing advanced frameworks like U-Net and FRRN to improve the segmentation process significantly.
- **Environmental Robustness**: The autoencoders are designed to adapt to varying environmental conditions, making the system versatile and reliable.

## Background and Motivation
The approach detailed in the paper "Anomaly Detection on the Rail Lines Using Semantic Segmentation and Self-supervised Learning" has shown success in real-world applications, highlighting the efficacy of the method. This project builds upon the technological advancements in semantic segmentation, utilizing the RailSem19 dataset to push the boundaries further in the railway sector.
## Methods

### Data Source

Our project harnesses the RailSem19 dataset, renowned for its extensive coverage of rail and tram environments. This dataset's rich variety in weather, lighting, and geographic scenarios serves as the foundation for training and fine-tuning our anomaly detection algorithms, ensuring their robustness and precision in real-world settings.

**Dataset Details:**
- **Source**: Available at [wilddash.cc](https://wilddash.cc)
- **Composition**: Over 1,000 video sequences uploaded by rail enthusiasts.
- **Variety**: Features 8,500 sequences showcasing a broad spectrum of environmental conditions and geographic locations.
- **Annotations**: Includes novel rail-specific labels, geometric shapes, and pixel-wise semantic labels. These are crafted through a blend of manual and weakly supervised annotations, derived from existing networks in the road domain.
- **Utility**: The dataset is instrumental for semantic scene understanding specific to the rail environment, presenting unparalleled opportunities for research in computer vision.

**Usage in Our Project:**
The JSON files accompanying the RailSem19 dataset images provide detailed rail line annotations. These annotations have been pivotal in generating mask files for the rail tracks in each image through a custom mask generator script.

### Mask Generation
Utilizing the annotations from the RailSem19 dataset, our mask generator script creates precise mask files that are essential for training our deep learning models. This script ensures that our models are attuned to the specific contours and characteristics of rail tracks, which is vital for the accurate detection of anomalies.

## Methodology

Our project employs semantic segmentation using a U-Net architecture enhanced with a VGG backbone. This methodology is chosen for its efficacy in pixel-wise segmentation tasks, particularly suited for the complex task of rail anomaly detection.

### U-Net Architecture with VGG Backbone

#### VGG Backbone
The VGG (Visual Geometry Group) model is a deep convolutional neural network that is recognized for its simplicity and depth. In our U-Net architecture, the VGG model acts as the encoder, sequentially applying convolutional layers with ReLU activations and max-pooling layers to capture hierarchical patterns in the data, from simple textures to complex rail line structures.

#### Convolutional Layers (Conv) + ReLU
Convolutional layers are the core components of our CNN, extracting spatial hierarchies of features. The ReLU activation function introduces non-linearity, enabling the network to learn complex patterns within the rail environment.

#### Max Pooling
Max pooling layers reduce the spatial dimensions of the feature maps, making the model more invariant to input variations and ensuring focus on the most salient features.

#### Upsampling
In the decoder part of the U-Net, upsampling layers are crucial for increasing the resolution of feature maps, allowing the network to project detailed features back onto the original image size for accurate pixel-wise segmentation.

#### Copy and Concatenate
A key feature of the U-Net architecture is the copy and concatenate operation. It merges the downsampled features with the upsampled ones, providing rich contextual information necessary for precise localization in the segmentation task.

#### Convolutional Layers (Conv) + Batch Normalization
Following feature concatenation, the network applies additional convolutional layers coupled with batch normalization. This step is vital for stabilizing the learning process and refining the feature maps, leading to the final segmentation output.

### Fig 3. U-Net Architecture with VGG Backbone
(Here, an illustrative figure of the U-Net architecture with the VGG backbone would be included.)

The final output of the U-Net architecture is a segmentation map that accurately delineates rail lines from the background. By utilizing the rich feature set extracted by the VGG backbone from the diverse RailSem19 images, combined with the precise localization capabilities of U-Net, our system is equipped to detect fine details in rail lines against complex backdrops.

### Technologies Used

List the technologies, libraries, and frameworks used in your project.

- TensorFlow
- Keras
- VGG19
- Autoencoder-decoder

