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

## Deep Learning Models
we used 2 models in this project.
We apply semantic segmentation to the complex task of rail anomaly detection using a U-Net architecture with a VGG backbone, chosen for its precision in pixel-wise identification.

### 1. U-Net Architecture with VGG Backbone

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

## 2. Autoencoder for Anomaly Detection

We use an autoencoder architecture to pinpoint anomalies on rail tracks, which is adept at identifying deviations from the norm within the rail line images.

### Fig 4. Autoencoder Architecture for Anomaly Detection
(Insert diagram of the autoencoder architecture here)

- **Input Layer**: 
  - Begins with a multi-channel input image of rail line patches, for example, 16x16x3 in dimensions.
  - The input layer feeds the image into the autoencoder.

- **Convolutional Layer (Conv2D)**: 
  - Applies filters (e.g., 64 filters of size 3x3) to detect local features like edges and textures within the rail line images.

- **Flatten Layer**: 
  - Converts the multi-dimensional output of the Conv2D layer into a one-dimensional array for processing in dense layers.

- **Dense Layers (Latent Space)**: 
  - Processes the features into a series of fully connected layers, reducing the data dimensionality and encoding it into a compact latent space.

- **Reshape Layer**: 
  - Transforms the data from the latent space back into a multi-dimensional array to prepare for the decoding process.

- **Conv2D Transpose Layer**: 
  - Acts as the inverse of the Conv2D layer, upsampling the data from the latent space back to its original image dimensions.

- **Output Layer (Reconstructed Image I’)**: 
  - Produces a reconstructed version of the input image, aiming to closely match the original in the absence of anomalies.

- **Anomaly Detection**: 
  - Relies on the comparison between the input image (I) and the reconstructed image (I').
  - Utilizes the reconstruction error to detect anomalies, with higher errors indicating potential anomalies.

The training of the autoencoder focuses on normal rail line images, enabling the model to reconstruct these effectively. When faced with anomalies, the reconstruction error spikes, which our system uses as a signal to flag potential issues. By setting a threshold for this error, we can reliably identify and alert to anomalies in the rail lines.

### Technologies Used

List the technologies, libraries, and frameworks used in your project.

- TensorFlow
- Keras
- VGG19
- Autoencoder-decoder





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

### 2. Autoencoder for Anomaly Detection
- **Input Layer**: Starts with multi-channel rail line images.
- **Convolutional Layer (Conv2D)**: Detects features like edges and textures.
- **Flatten Layer**: Prepares features for dense processing.
- **Dense Layers (Latent Space)**: Encodes salient features into compact space.
- **Reshape Layer**: Readies encoded features for upsampling.
- **Conv2D Transpose Layer**: Reconstructs original image dimensions.
- **Output Layer**: Produces a reconstructed image for anomaly comparison.

Anomalies are detected by comparing the original input with the reconstructed image, leveraging discrepancies to flag foreign objects on rail tracks.

### Fig 3. U-Net Architecture with VGG Backbone
(Insert diagram here)

### Fig 4. Autoencoder Architecture for Anomaly Detection
(Insert diagram here)

### Technologies Used

- **Frameworks**: TensorFlow, Keras
- **Models**: VGG19, Custom Autoencoder
- **Data Processing**: Mask Generation Script using JSON annotations

