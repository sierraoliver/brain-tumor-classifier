# Brain Tumor Classifier
## Description

This project is a deep learning–based image classification system built with PyTorch to classify brain MRI images into four categories (glioma, meningioma, pituitary, no tumor) . It uses a custom convolutional neural network (CNN) trained on labeled medical imaging data and supports training, evaluation, visualization, and real-time inference on user-provided images.

The application is designed as an end-to-end machine learning pipeline, handling data loading, preprocessing, model training, evaluation, persistence, and interactive prediction from the command line.

------------------------------------------------------------------------------------

## Features

- Image Classification with CNN
  - Custom convolutional neural network implemented using PyTorch
  - Classifies MRI images into four brain condition categories

- Data Preprocessing & Augmentation
  - Image resizing and normalization
  - Random horizontal flipping and rotation for improved generalization
  - Separate transforms for training and testing pipelines

- Efficient Data Loading
  - Uses torch.utils.data.DataLoader and ImageFolder for scalable dataset handling
  - Supports GPU acceleration when available

- Model Training & Persistence
  - Trains using AdamW optimizer and cross-entropy loss
  - Automatically saves trained model for reuse
  - Allows loading a previously trained model to skip retraining

- Evaluation & Metrics
  - Computes test loss and classification accuracy
  - Runs inference in evaluation mode with gradient tracking disabled

- Visualization & Debugging
  - Displays random test samples with predicted and ground-truth labels
  - Useful for qualitative inspection of model performance

- Interactive Inference
  - Predicts classes for individual image files provided by the user
  - Robust input validation for file existence

------------------------------------------------------------------------------------
## Dataset Structure

**Link for dataset used:**  
https://data.mendeley.com/datasets/zwr4ntf94j/1

The dataset should follow the directory structure required by `torchvision.datasets.ImageFolder`:

```
dataset/
├── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── notumor/
│ └── pituitary/
└── Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Each class directory should contain corresponding MRI images.

------------------------------------------------------------------------------------

## Installation

1) Clone the repository:
  - git clone https://github.com/yourusername/brain-tumor-classifier.git
  - cd brain-tumor-classifier

2) Install required Python packages:
  - pip install torch torchvision pillow matplotlib

------------------------------------------------------------------------------------

## Usage

Run the main training and inference pipeline:
  - python main.py

During execution, the program allows you to:
  - Load an existing trained model or train a new one
  - Evaluate model performance on the test set
  - Visualize sample predictions
  - Perform predictions on custom image files interactively

------------------------------------------------------------------------------------
