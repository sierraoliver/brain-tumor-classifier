# Brain Tumor Classifier
## Description

This project is a deep learning–based image classification system built with PyTorch to classify brain MRI images into four categories (glioma, meningioma, pituitary, no tumor) . It uses a custom convolutional neural network (CNN) trained on labeled medical imaging data and supports training, evaluation, visualization, and real-time inference on user-provided images.

The application is designed as an end-to-end machine learning pipeline, handling data loading, preprocessing, model training, evaluation, persistence, and interactive prediction from the command line.

------------------------------------------------------------------------------------
## Features
- **Image Classification:** Custom CNN for 4-class brain MRI prediction
- **Data Preprocessing:** Resizing, normalization, and augmentations (flip, rotation)
- **Efficient Training:** GPU support, AdamW optimizer, model checkpointing
- **Evaluation & Visualization:** Test accuracy, loss metrics, and sample predictions
- **Interactive Inference:** Predict on custom MRI images with robust input validation
  
------------------------------------------------------------------------------------
## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, CNN
- **Data Handling:** Torchvision, PIL
- **Visualization:** Matplotlib
- **Deployment:** Command-line interface, interactive inference
  
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
