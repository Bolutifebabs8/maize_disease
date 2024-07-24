import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown

@st.cache_resource
def download_model():
    url = 'https://drive.google.com/uc?id=17R3lfH8DHW6LKZjTXd0K8tOu3IAzNTu3'
    output = 'maize_leaf_disease_model.h5'
    gdown.download(url, output, quiet=False)
    return load_model(output)

model = download_model()

def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_disease(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.title("Maize Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    prediction = predict_disease(image, model)

    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    predicted_class = classes[np.argmax(prediction)]
    st.write(f'Prediction: {predicted_class}')
