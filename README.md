# Rail Track Feature Extraction (RT-FE)

![Rail Track Feature Extraction Example](URL_TO_YOUR_IMAGE)

## Introduction
RT-FE harnesses advanced deep learning to precisely extract features from rail tracks. This autonomous system is pivotal for tasks such as anomaly detection, significantly diminishing the need for manual data labeling and enhancing the efficiency and accuracy of rail track analysis.

## Project Description
RT-FE employs binary semantic segmentation to isolate rail tracks from their surroundings. This clean separation of features sets the stage for in-depth analysis and is integral to ensuring rail track safety and integrity.

### Binary Semantic Segmentation
Through binary semantic segmentation, RT-FE accurately isolates rail features, enabling detailed examination and feature extraction. This process is essential for subsequent analysis, ensuring that crucial features are not overlooked.

## Key Contributions

- **Enhanced Feature Extraction Accuracy**: Our system guarantees precise segmentation of rail track features, providing high-quality data for downstream analyses.
- **Reduced Reliance on Labeled Data**: Leveraging binary semantic segmentation significantly minimizes the need for extensive labeled datasets.
- **Advanced Technological Integration**: The project incorporates leading-edge technologies, specifically the U-Net architecture with a VGG backbone, to ensure effective rail track segmentation.
- **Adaptability Across Environments**: The model demonstrates exceptional performance in various environmental conditions, showcasing its robustness and reliability.

## Background and Motivation
Drawing inspiration from the proven efficacy of deep learning in practical scenarios, RT-FE extends these advancements to the realm of rail track feature extraction. Utilizing the comprehensive RailSem19 dataset, our project lays the groundwork for developing next-generation rail safety technologies.

## Methods

### Data Source
The RailSem19 dataset is instrumental in our model’s development, offering an extensive collection of diverse rail scenes. This variety enriches our algorithm's training, ensuring its applicability in a multitude of real-world scenarios.

**Dataset Highlights**:
- **Accessibility**: Available at [wilddash.cc](https://wilddash.cc).
- **Diversity**: Encompasses over 8,500 sequences across various environmental conditions.
- **Innovative Annotations**: The dataset combines manual and weakly supervised annotations, providing an in-depth view of rail environments.

**Implementation in RT-FE**:
The dataset's detailed annotations are crucial for generating precise rail track masks, fundamental to the training of our segmentation model.

## Deep Learning Model

### U-Net with VGG Backbone
- **VGG Backbone**: Captures intricate patterns as the encoder.
- **Convolutional Layers + ReLU**: Facilitates feature extraction and processing.
- **Max Pooling**: Highlights critical features by reducing spatial dimensions.
- **Upsampling**: Restores resolution for detailed feature mapping.
- **Copy and Concatenate**: Merges multi-level features for improved localization.
- **Convolutional Layers + Batch Normalization**: Refines the feature maps to deliver the final segmented output.

### Fig 1. U-Net Architecture with VGG Backbone
![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/VGG-Unet.png)

### Fig 2. Segment Model Input and Result 
![Segment Model Input and Result](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/segmentmask.png)

This architecture ensures the production of a highly accurate segmentation map, distinctly delineating rail track features, critical for subsequent analysis and safety measures.

### Technologies Used

- **Framework**: TensorFlow, Keras
- **Model**: U-Net enhanced with VGG19
- **Data Processing**: Custom script for efficient mask generation using JSON annotations from RailSem19

We encourage contributions and further research using RT-FE's methodologies and findings to advance the field of rail track analysis.
