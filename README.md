# Emotion-Detection-Face-Sentiment-Analysis

Data Source: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

# Emotion Classification Using DenseNet-169

This project focuses on emotion classification from facial images using the DenseNet-169 deep learning architecture. We utilize the FER-2013 dataset and apply transfer learning techniques to achieve robust classification results.

## Project Overview

- **Dataset:** FER-2013 (35,887 grayscale images, 48x48 pixels) with seven emotion categories: anger, disgust, fear, happiness, sadness, surprise, and neutral.
- **Model:** DenseNet-169, pre-trained on ImageNet and fine-tuned for emotion classification.
- **Metrics:** Achieved ~65% accuracy on test data with an ROC AUC score of 0.92, indicating strong classification capability across emotion classes.

## Methodology

### 1. Data Preprocessing and Augmentation
- **Preprocessing:** Standardized input images using TensorFlow’s `ImageDataGenerator` and DenseNet preprocessing function.
- **Augmentation Techniques:**
  - Horizontal Flip
  - Width & Height Shifts
  - Rescaling

### 2. Model Architecture and Training
- **Model Setup:** Used DenseNet-169 as a feature extractor with additional fully connected layers for classification.
  - Three dense layers (256, 1024, 512 units) with ReLU activations and dropout layers to prevent overfitting.
  - Final layer: SoftMax for 7-class emotion classification.
- **Training Strategy:** Two-stage training approach:
  - Initial training with frozen DenseNet layers.
  - Fine-tuning with lower learning rate after unfreezing DenseNet layers.

### 3. Evaluation
- **Performance Metrics:**
  - Accuracy: ~65% on test data.
  - Confusion Matrix: Shows model performance across all emotion classes.
  - ROC AUC: ~0.92, indicating strong class differentiation capability.
  
## Key Results

- **Confusion Matrix:** Provides insights into the model’s performance on each emotion class, identifying areas of misclassification.
- **ROC AUC Curve:** High AUC score demonstrates the model’s effectiveness in distinguishing between different emotions.





