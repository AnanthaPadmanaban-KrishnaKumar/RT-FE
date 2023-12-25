# Rail Track Feature Extraction (RT-FE)

![Rail Track Feature Extraction Example](URL_TO_YOUR_IMAGE)

## Introduction
RT-FE employs advanced deep learning techniques for feature extraction from rail tracks. The system is designed to autonomously identify key features within rail scenes, which is crucial for subsequent tasks like anomaly detection, reducing the need for manual labeling of training data.

## Project Description

The RT-FE project uses binary semantic segmentation to isolate rail tracks from their environment, providing a clean separation of features for further analysis and processing.

### Binary Semantic Segmentation
We achieve the isolation of rail features through a process known as binary semantic segmentation. This method sets the stage for extracting detailed features critical for maintaining rail safety.

## Key Contributions

- **High Accuracy in Feature Extraction**: Our system provides precise segmentation of rail track features, ensuring high-quality data for further analysis.
- **Reduction in Label Dependency**: The binary semantic segmentation approach significantly reduces the need for a large labeled dataset.
- **State-of-the-Art Frameworks**: We incorporate advanced frameworks like U-Net with a VGG backbone to segment rail track features effectively.
- **Environmental Adaptability**: The segmentation model is capable of handling diverse environmental conditions, demonstrating robust performance.

## Background and Motivation
Rooted in the success of deep learning for practical applications, our initiative extends these advancements to rail track feature extraction. We utilize the RailSem19 dataset to create a foundation for developing technologies that enhance rail safety.

## Methods

### Data Source

The RailSem19 dataset is a pivotal component in our training process, offering a rich array of rail scenes for a robust development of our algorithms.

**Dataset Highlights**:
- **Access**: Available at [wilddash.cc](https://wilddash.cc)
- **Variety**: Over 8,500 sequences that span a wide array of environmental conditions.
- **Unique Annotations**: Manual and weakly supervised annotations that provide a detailed understanding of rail scenes.

**Application in RT-FE**:
Annotations from the RailSem19 dataset are essential for creating precise masks for the rail tracks, enabling effective training of our segmentation model.

## Deep Learning Model

### U-Net with VGG Backbone
- **VGG Backbone**: Serves as the encoder, essential for capturing the intricate patterns within rail scenes.
- **Convolutional Layers + ReLU**: Extracts and processes features through learned filters and non-linear activation.
- **Max Pooling**: Reduces the spatial dimensions, highlighting the most significant features for the task at hand.
- **Upsampling**: Reconstructs the detailed feature map to match the original image's resolution.
- **Copy and Concatenate**: Combines features from different levels of the network to maintain context and enhance localization.
- **Convolutional Layers + Batch Normalization**: Further refines the feature maps to produce the final segmented output.

### Fig 1. U-Net Architecture with VGG Backbone
![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/VGG-Unet.png)

### Fig 2. Segment Model Input and Result 
![Segment Model Input and Result](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/segmentmask.png)

The U-Net architecture is optimized to maintain high-resolution details and merge both low-level and high-level information. This results in an accurate segmentation map that distinctly identifies rail track features.

### Technologies Used

- **Framework**: TensorFlow, Keras
- **Model**: VGG19 enhanced U-Net
- **Data Processing**: Custom script for mask generation using JSON annotations

Contributors and researchers are welcome to utilize our findings and methods to build upon the field of rail track analysis.

