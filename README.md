# Rail Track Feature Extraction (RT-FE)

<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/input_video.gif" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/output_video.gif" width="49%" />
</p>



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

## Project workflow

![Workflow](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE/blob/main/assets/RT-FE%20flowchart.png)

- **Video Ingestion**: The process begins with the input video, which is the raw footage needing analysis.
- **Frame Extraction**: The input video is broken down into individual frames to allow for detailed processing of each moment in the footage.
- **Image Preprocessing**: Each extracted frame undergoes preprocessing, which may include normalization, resizing, and augmentation to prepare for effective segmentation.
- **Segmentation Model**: A specialized segmentation model, UNET with a VGG19 architecture, is employed to analyze each preprocessed frame, identifying and isolating the features of interest.
- **Post-Processing**: The output from the segmentation model is then refined through post-processing to enhance the accuracy of the segmentation, remove noise, and improve the clarity of the resultant masks.
- **Mask Generation**: The post-processed frames are combined to produce output masks that highlight the features extracted from the original video, completing the feature extraction process.

## Deep Learning Model

### U-Net with VGG Backbone
- **VGG Backbone**: Captures intricate patterns as the encoder.
- **Convolutional Layers + ReLU**: Facilitates feature extraction and processing.
- **Max Pooling**: Highlights critical features by reducing spatial dimensions.
- **Upsampling**: Restores resolution for detailed feature mapping.
- **Copy and Concatenate**: Merges multi-level features for improved localization.
- **Convolutional Layers + Batch Normalization**: Refines the feature maps to deliver the final segmented output.

![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/VGG-Unet.png)

## Dataset Composition

- **Training Set**: 4,000 images with corresponding segmentation masks.
- **Testing Set**: 1,500 images with associated masks for model accuracy evaluation.

## Preprocessing Techniques

- **Resizing**: Uniformly resized images and masks to 857 x 572 x 3 to standardize the input data.
- **Normalization**: Applied normalization to standardize pixel values across all images and eliminate outliers.

## Training Infrastructure

- Conducted on **Amazon SageMaker** with an NVIDIA Tesla T4 GPU (ml.g5.2xlarge instance).

## Training Hyperparameters

- **Epochs**: 100 epochs to balance learning and prevent overfitting.
- **Batch Size**: A batch size of 4, optimizing memory usage and model performance.
- **Learning Rate**: Set to 0.0001 for steady convergence without overshooting minima.
- **Custom Loss Function**: Weighted Binary Crossentropy with Focal Loss to address class imbalance and focus on challenging examples.
- **Primary Metric**: Accuracy was used to gauge predictive performance.
- **Callbacks**: Early Stopping with a patience of 16 epochs and model checkpointing to save the best-performing model iteration.
- **TensorBoard**: Utilized for real-time monitoring of training metrics and performance analysis.

## Loss Function

### Custom Loss Function for Enhanced Learning

To extract rail line features using a U-Net with VGG19 as the backbone, we implement a custom loss function that significantly enhances the model's learning from the complex data distribution in rail scenes.

#### Weighted Binary Crossentropy

Our custom loss function begins with weighted binary crossentropy, addressing the class imbalance issue common in rail image segmentation. The weight parameters `w0` and `w1` are assigned to the background and rail line classes, respectively, to focus the model's learning on the minority class, which is critical for the detection task.
![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/weighted.png)

#### Focal Loss

The focal loss component reshapes the loss function to prioritize challenging, misclassified examples. It introduces hyperparameters `alpha` and `gamma` to control the contribution of each example to the loss, enabling the model to focus on difficult cases crucial for accurate feature extraction.

![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/Focal.png)

#### Combined Loss

We average weighted binary crossentropy and focal loss to create a robust loss landscape:

![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/combined.png)

Here is the loss graph from our training process, showing the model's learning progress:
### Loss Graph
![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/lossgraph.png)

## Accuracy Results

The following table shows the UNET model training results with different batch sizes and loss functions:

| Batch Size | Loss Function          | mIOU |
|------------|------------------------|------|
| 4          | BCE                    | 53.4 |
| 4          | MSE                    | 57.1 |
| **4**      | **Combined Loss (WBCE+FL)** | **63.2** |
| 8          | BCE                    | 52.8 |
| 8          | MSE                    | 55.9 |
| 8          | Combined Loss (WBCE+FL)| 58.4 |
| 16         | BCE                    | 51.6 |
| 16         | MSE                    | 51.3 |
| 16         | Combined Loss (WBCE+FL)| 54.7 |

The best-performing model uses a batch size of 4 and a combined loss function consisting of Weighted Binary Cross Entropy (WBCE) and Focal Loss (FL), achieving a mean Intersection over Union (mIOU) of 63.2.

Abbreviations:
- **WBCE**: Weighted Binary Cross Entropy
- **FL**: Focal Loss
- **BCE**: Binary Cross Entropy
- **MSE**: Mean Squared Error
  
### Results

The results demonstrate the effectiveness of the U-Net model with VGG19 backbone in segmenting rail track features:

<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/rs00019_image.jpg" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/rs00019.jpg" width="49%" />
</p>
<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/rs00025_image.jpg" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/rs00025.jpg" width="49%" />
</p>
<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/rs00952_image.jpg" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS/blob/main/assets/rs00952.jpg" width="49%" />
</p>

This architecture ensures the production of a highly accurate segmentation map, distinctly delineating rail track features, critical for subsequent analysis and safety measures.

### Technologies Used

- **Framework**: TensorFlow, Keras
- **Model**: U-Net enhanced with VGG19
- **Data Processing**: Custom script for efficient mask generation using JSON annotations from RailSem19

## Conclusion

The Rail Track Feature Extraction (RT-FE) project represents a significant advancement in the application of deep learning for rail infrastructure analysis. By employing the U-Net architecture with a VGG19 backbone, coupled with innovative binary semantic segmentation techniques, we have established a robust framework for extracting rail track features with high precision.

Our custom-designed loss function, which combines weighted binary crossentropy and focal loss, specifically addresses the unique challenges of class imbalance and the intricate nature of rail track data. The successful implementation of this loss function is evident in the model's enhanced ability to accurately identify and segment rail features, a critical step towards automating the monitoring and maintenance of rail infrastructure.

The RT-FE project not only demonstrates the effectiveness of applying cutting-edge machine learning techniques to real-world problems but also paves the way for future innovations in this domain. We have shown that with the right tools and methodologies, deep learning can be a powerful ally in improving the safety and reliability of critical transportation infrastructure.

We believe that the methods and findings of the RT-FE project will serve as a valuable resource for researchers and practitioners in the field. We look forward to seeing how the community will build upon our work to further enhance the state of rail safety technology and welcome contributions that will help to realize this vision.

We invite the research and development community to participate in improving and extending the RT-FE project. By collaborating, we can further the evolution of smart transportation infrastructure and safety.

For more information, issues, or questions about the project, please open an issue in the GitHub repository or submit a pull request with your suggestions and improvements.

## Getting Started

To clone and run the RT-FE project on your local machine, follow these steps:

### Prerequisites

Make sure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)
- Git

### Installation
1. git clone [https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FE.git](https://github.com/AnanthaPadmanaban-KrishnaKumar/RT-FODS)
2. cd RT-FE
3. pip install -r requirements.txt

### Training the Segment Model
1. cd training_scripts
2. use the tfrecorder.ipynb to convert the images and mask to tfrecord format to efficiently load the data into the model
3. run python trainer_rt-fe.py
   
### Running the Project
1. cd pipeline
2. python main.py

Thank you for your interest in RT-FE, and we look forward to your contributions to this vital field of research and development.
