import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the trained model
model_path = os.path.join('trained_model', 'cat_dog_classifier.keras')
model = tf.keras.models.load_model(model_path)

# Set page config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Title and description
st.title("Cat vs Dog Classifier")
st.write("Upload an image to classify it as a cat or dog!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    try:
        prediction = model.predict(img_array)
        class_name = "Dog" if prediction[0][0] >= 0.5 else "Cat"
        confidence = prediction[0][0] if class_name == "Dog" else 1 - prediction[0][0]
        st.success(f"Prediction: {class_name} (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Run instructions
st.write("### How to Use:")
st.write("- Upload a JPG, JPEG, or PNG image.")
st.write("- The model will classify it as a cat or dog and display the confidence score.")