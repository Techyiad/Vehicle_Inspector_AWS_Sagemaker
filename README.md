# Vehicle Damage Classification and Detection using Mask R-CNN

## Overview
This repository contains code for building a vehicle damage classification and detection model using Mask R-CNN on AWS SageMaker. The model is trained to classify and segment different parts of a vehicle in images, enabling accurate detection and localization of damages.

## Project Structure
- `data_processing/`: Scripts for data preprocessing and labeling
- `model_training/`: Code for training the Mask R-CNN model
- `inference/`: Code for performing inference on new images
- `deployment/`: Scripts and configuration files for deploying the model on AWS SageMaker endpoint

## Data Collection and Preprocessing
The dataset for this project was collected from Flickr.com. The images were downloaded and preprocessed to ensure consistent formatting and quality. The data preprocessing pipeline was implemented using Google Cloud Dataflow, allowing for efficient processing and storage of the images in Google Cloud Storage.

## Data Labeling
To create a labeled dataset for training the model, the images were manually annotated using CVAT (Computer Vision Annotation Tool). Each part of the vehicles in the images was carefully labeled for training and segmentation. The annotations provide precise information about the location and type of damage on each vehicle.

## Model Training
The Mask R-CNN model was implemented using TensorFlow. The training code, located in the `model_training/` directory, includes all the necessary scripts and configurations for training the model on the annotated dataset. The model was trained to classify different vehicle parts and generate accurate segmentation masks for damage detection.

## Inference and Evaluation
The `inference/` directory contains code for performing inference on new images using the trained model. The model can detect and classify vehicle damages in real-time, providing valuable insights for analysis and decision-making. Evaluation metrics, such as precision, recall, and IoU (Intersection over Union), can be calculated to assess the model's performance.

## Deployment on AWS SageMaker
The `deployment/` directory includes scripts and configuration files for deploying the trained model on an AWS SageMaker endpoint. This allows for easy integration with other applications or services for real-time inference. Instructions and guidelines for deploying the model are provided in the respective directory.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Google Cloud SDK (for data preprocessing)
- AWS SDK (for deployment on SageMaker)

## Getting Started
To get started with this project, follow the steps below:

1. Install the necessary dependencies mentioned in the 'Requirements' section.
2. Set up the Google Cloud SDK for data preprocessing, if not already done.
3. Collect and preprocess the dataset using the provided scripts in the `data_processing/` directory.
4. Manually label the dataset using CVAT for each part of the vehicles in the images.
5. Train the Mask R-CNN model using the annotated dataset and the training code in the `model_training/` directory.
6. Perform inference on new images using the code in the `inference/` directory.
7. Follow the instructions in the `deployment/` directory to deploy the model on AWS SageMaker endpoint.

## Contribution
Contributions to this project are welcome! If you encounter any issues, have suggestions, or would like to add new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

