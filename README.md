# Face Recognition and Verification Project

## Overview
This project focuses on building an advanced system for **face recognition** and **face verification** using a pre-trained **InceptionResNetV1** model. The primary goal is to create a robust solution for applications such as **security authentication**, **access control**, and **identity management systems**. The system includes:

- **Face Classifier**: Identifies individuals from a dataset.
- **Face Verifier**: Compares two face embeddings to determine identity match or mismatch.

By exploring various configurations for fine-tuning, including freezing/unfreezing layers and integrating Multi-Layer Perceptrons (MLP), the project achieves high accuracy and robust verification performance.

## Features
- **Face Recognition**: Classifies images into predefined identities with high precision.
- **Face Verification**: Measures cosine similarity and Euclidean distance between embeddings for identity matching.
- **Flexible Training**: Multiple configurations tested for optimal performance.
- **High Accuracy**: Up to **99.66%** in classification and **98.86%** in verification.

## Usage
### Training the Model
1. **Load Pre-trained Model**:
   ```python
   from model import load_model
   model = load_model("InceptionResNetV1")
   ```
2. **Fine-tune and Train**:
   ```python
   from training import train_model
   config = {
       'freeze_layers': 10,  # Customize as needed
       'optimizer': 'Adam',
       'learning_rate': 0.0001
   }
   train_model(model, dataset, config)
   ```

### Verifying Faces
1. Extract embeddings for two faces:
   ```python
   from embedding import extract_embeddings
   embedding1 = extract_embeddings(model, image1)
   embedding2 = extract_embeddings(model, image2)
   ```
2. Compute similarity:
   ```python
   from similarity import cosine_similarity, euclidean_distance
   cosine_score = cosine_similarity(embedding1, embedding2)
   euclidean_score = euclidean_distance(embedding1, embedding2)
   ```
   - If cosine similarity > 0.9 or Euclidean distance < 0.002, faces are considered a match.

## Results
### Classifier Performance
- **Configuration**: Last 10 layers unfrozen, 3-layer MLP with dropout.
   - **Accuracy**: 99.66%
   - **Test Error**: 0.32

### Verifier Performance
- **Similarity Metrics**:
   - **Cosine Similarity**: Achieved **98.86% accuracy**.
   - **Euclidean Distance**: Achieved **97.72% accuracy**.


## Technologies Used
- **Programming Language**: Python
- **Libraries**: PyTorch, NumPy, Matplotlib
- **Model Architecture**: Pre-trained InceptionResNetV1


