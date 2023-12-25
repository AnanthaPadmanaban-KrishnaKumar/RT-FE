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

We apply semantic segmentation to the complex task of rail anomaly detection using a U-Net architecture with a VGG backbone, chosen for its precision in pixel-wise identification.

### U-Net Architecture with VGG Backbone

- **VGG Backbone**: 
  - Acts as the encoder in our U-Net architecture.
  - Uses deep convolutional layers known for capturing complex image patterns.
  - Includes ReLU activation functions and max-pooling to hierarchically process image features from simple to complex.

- **Convolutional Layers (Conv) + ReLU**: 
  - Serve as the primary feature extractors.
  - ReLU introduces non-linearity, allowing for the capture of complex patterns.

- **Max Pooling**: 
  - Reduces spatial dimensions to focus on prominent features.
  - Aids in the model's robustness to input variations and distortions.

- **Upsampling**: 
  - Increases the spatial resolution during the decoding phase.
  - Essential for translating learned features back to the original image scale.

- **Copy and Concatenate**: 
  - Merges features from the downsampling path with upsampled content.
  - Provides contextual information critical for accurate localization.

- **Convolutional Layers (Conv) + Batch Normalization**: 
  - Stabilizes learning by normalizing the input layers.
  - Refines feature maps post-concatenation, preparing for the final segmentation output.

The U-Net's architecture is particularly adept at maintaining high-resolution details through the upsampling process and merging low-level and high-level feature information. This results in a segmentation map that precisely differentiates rail lines from their surroundings, which is pivotal given the RailSem19 dataset's complexity.

### Fig 3. U-Net Architecture with VGG Backbone
(Insert diagram of the U-Net architecture with the VGG backbone here)

Through this methodology, our model effectively captures the intricate details required for accurate rail line segmentation, even in the diverse conditions presented by the RailSem19 dataset images.

### Technologies Used

List the technologies, libraries, and frameworks used in your project.

- TensorFlow
- Keras
- VGG19
- Autoencoder-decoder

