import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# Load your trained VGG16 model
model = load_model('lung_cancer_vgg16_model.h5')

# Define class names
classes = ['benign', 'squamous', 'adenocarcinoma']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to VGG16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to classify the input image
def classify_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return classes[predicted_class], predictions[0]

# Streamlit app
st.title("Lung Cancer Detection Using VGG16")

# Upload an image file
uploaded_file = st.file_uploader("Please upload a histopathology image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Classify image on button click
    if st.button('Classify Image'):
        result_class, probabilities = classify_image(img)
        
        # Display results
        st.write(f"### Predicted Class: {result_class}")
        st.write(f"### Class Probabilities:")
        st.write(f"Benign: {probabilities[0]:.4f}")
        st.write(f"Squamous: {probabilities[1]:.4f}")
        st.write(f"Adenocarcinoma: {probabilities[2]:.4f}")
