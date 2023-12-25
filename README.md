# Rail_Track_Foreign_Object_Detection (RT-FODS)

![Rail Track Anomaly Detection Example](URL_TO_YOUR_IMAGE)

## Introduction
RT-FODS harnesses advanced deep learning techniques to detect and categorize foreign objects on rail tracks. Our system is uniquely capable of identifying potential hazards autonomously, significantly reducing the reliance on manual labeling of training data.

## Project Description

Utilizing a two-stage process, RT-FODS first isolates rail tracks through binary semantic segmentation and then employs self-supervised autoencoders for precise anomaly detection.

### Stage 1: Binary Semantic Segmentation
Isolation of rails from their surroundings is achieved with binary semantic segmentation, setting the groundwork for detailed anomaly analysis.

### Stage 2: Self-Supervised Autoencoders for Anomaly Detection
The second stage involves self-supervised autoencoders that identify irregularities like coal, dirt, and unexpected obstacles, critical for rail safety.

## Key Contributions

- **Precision in Anomaly Detection**: High-accuracy identification of foreign objects on rail tracks with minimal false readings.
- **Label-Efficient Learning**: A methodology that drastically reduces the need for a large labeled dataset.
- **Cutting-Edge Technology**: Incorporation of state-of-the-art semantic segmentation frameworks like U-Net enhanced with VGG backbones.
- **Adaptability**: Autoencoders designed to perform consistently across diverse environmental conditions.

## Background and Motivation
Our work is grounded in "Anomaly Detection on the Rail Lines Using Semantic Segmentation and Self-supervised Learning," highlighting the success of deep learning in practical applications. We extend this innovation to rail tracks, leveraging the RailSem19 dataset to advance rail safety technology.

## Methods

### Data Source

The RailSem19 dataset, with its extensive collection of rail scenes, serves as the core for our model training, providing a diversified set of environmental and geographic contexts for robust algorithm development.

**Dataset Highlights**:
- **Availability**: [wilddash.cc](https://wilddash.cc)
- **Diversity**: Over 8,500 sequences with a wide range of conditions.
- **Innovative Annotations**: A combination of manual and weakly supervised labels, enhancing computer vision research specifically for rail environments.

**Implementation in RT-FODS**:
Annotations from RailSem19 are instrumental in generating accurate mask files for rail track images, aiding our deep learning models in learning precise anomaly detection.

### Deep Learning Models

RT-FODS utilizes two main deep learning models, each serving a unique purpose in the detection pipeline.

### 1. U-Net with VGG Backbone
- **VGG Backbone**: The encoder, capturing complex patterns.
- **Convolutional Layers + ReLU**: Feature extraction and non-linearity.
- **Max Pooling**: Focus on dominant features and input robustness.
- **Upsampling**: Restoring resolution for detailed segmentation.
- **Copy and Concatenate**: Merging features for precise localization.
- **Convolutional Layers + Batch Normalization**: Refining the feature maps for final output.
### Fig 1. U-Net Architecture with VGG Backbone
![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/VGG-Unet.png)


### 2. Autoencoder for Anomaly Detection
- **Input Layer**: Starts with multi-channel rail line images.
- **Convolutional Layer (Conv2D)**: Detects features like edges and textures.
- **Flatten Layer**: Prepares features for dense processing.
- **Dense Layers (Latent Space)**: Encodes salient features into compact space.
- **Reshape Layer**: Readies encoded features for upsampling.
- **Conv2D Transpose Layer**: Reconstructs original image dimensions.
- **Output Layer**: Produces a reconstructed image for anomaly comparison.
### Fig 2. U-Net Architecture with VGG Backbone
![Autoencoder Decoder](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/autoencoder.png)

Anomalies are detected by comparing the original input with the reconstructed image, leveraging discrepancies to flag foreign objects on rail tracks.

### Technologies Used

- **Frameworks**: TensorFlow, Keras
- **Models**: VGG19, Custom Autoencoder
- **Data Processing**: Mask Generation Script using JSON annotations

