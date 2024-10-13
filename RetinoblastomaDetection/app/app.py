import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import pickle
import numpy as np

load_dotenv()

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def print_predictions(predictions):
    np.set_printoptions(suppress=True)
    for pred in predictions:
        st.write("Predictions:")
        for i, prob in enumerate(pred):
            st.write(f"Class {i + 1}: {prob:.4f}")
st.set_page_config(page_title='RetinoNet', page_icon=':eye:', layout="wide")
st.title('RetinoNet: Retinal Image Analysis :eye:')
st.write('This is a web app to predict the presence of Retinoblastoma in retinal images.')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 
model = load_model(path=os.getenv('MODEL_PATH'))

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image_array = np.array(image)
    normalized_image = image_array / 255.0
    st.write("Classifying...")
    preds = model.predict(np.expand_dims(normalized_image, axis=0))
    st.write(print_predictions(preds))
else:
    st.write("Please upload an image to classify.")
