import streamlit as st
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()


st.set_page_config("Cryptocurrency Address Classifier", page_icon="🔒")

@st.cache_resource(show_spinner=False)
def load_model(model_path, scaler_path, encoder_path, model_encoder):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(model_encoder, 'rb') as f:
        model_encoder = pickle.load(f)
        
    return model, scaler, encoder, model_encoder

def preprocess_input(address):
    address_length = len(address)
    address_start = address[:2]
    address_encoded = encoder.transform([address_start])
    address_scaled = scaler.transform([[address_length, address_encoded[0]]])
    return address_scaled

def classify_address(model, address, encoder):
    input_data = preprocess_input(address)
    prediction = model.predict(input_data)
    # Decode the numerical label to the original class name
    class_name = encoder.inverse_transform([prediction[0]])
    return class_name[0]

encoder_path = os.getenv("ENCODER_PATH")
model_path = os.getenv("MODEL_PATH")
scaler_path = os.getenv("SCALER_PATH")
xg_encoder_path = os.getenv("XG_ENCODER")

model, scaler, encoder, model_encoder = load_model(model_path, scaler_path, encoder_path, xg_encoder_path)


st.title("Cryptocurrency Address Classifier")
st.markdown("Available Cryptos that could be classified:")
st.markdown("1. Bitcoin (BTC) " "2. Ethereum(ETH) " "3. Doge Coin " "4. Dash " "5. Lite Coin " "6. Zilliqa " "7. Polygon")
address_input = st.text_input("Enter a cryptocurrency address:")
if st.button("Classify"):
    if address_input:
        try:
            prediction = classify_address(model, address_input, model_encoder)
            st.write(f"The address {address_input} is classified as: {prediction}")
        except Exception as e:
            st.error(f"Error processing address: {e}")
    else:
        st.error("Please enter a cryptocurrency address.")
