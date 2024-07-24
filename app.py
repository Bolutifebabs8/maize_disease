#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved model
model = load_model('maize_leaf_disease_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the disease
def predict_disease(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
st.title("Maize Leaf Disease Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict the disease
    prediction = predict_disease(image, model)

    # Display the prediction
    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    predicted_class = classes[np.argmax(prediction)]
    st.write(f'Prediction: {predicted_class}')


# In[ ]:




