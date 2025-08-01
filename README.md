# Cat vs Dog Classifier

This Cat vs Dog Classifier uses deep learning with TensorFlow/Keras, applying transfer learning via MobileNetV2. It processes the "Cats Image (64*64)" (29,843 images) and "Dogs vs. Cats" datasets, training on 5,000 images/class at 64x64 resolution with data augmentation. The Convolutional Neural Network (CNN) model achieves 90-95% accuracy. Includes a training script and Streamlit web app for deployment, optimized for local or cloud use. Developed as of 06:28 PM IST, Aug 01, 2025.

I have used Kaggle dataset "Cats and Dogs Classification" by Mahmudul Haque Shawon for training the model
https://www.kaggle.com/datasets/mahmudulhaqueshawon/catcat?select=train
-Make sure you download the dataset before using the .ipynb
-Update the path in the notebook as well

## Features
- Utilizes CNN with transfer learning (MobileNetV2) for efficient classification.
- Handles 64x64 images with data augmentation for robustness.
- Trains on balanced dataset (5,000 cats, 5,000 dogs).
- Provides a web interface via Streamlit for predictions.
- Saves model and plots for analysis.

## Requirements
- **Python**: 3.7+
- **Dependencies**: `tensorflow`, `matplotlib`, `pandas`, `numpy`, `pillow`, `scikit-learn`, `streamlit`
- **Hardware**: GPU recommended (NVIDIA with CUDA support).

THANK YOU feel free to ask your doubts.....
