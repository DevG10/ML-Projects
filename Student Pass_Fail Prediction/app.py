import numpy as np
import pickle
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from streamlit_extras.let_it_rain import rain

st.set_page_config('Pass/Fail Predictor', page_icon = ":student:")
def padhai(study_hours) -> str:
    if study_hours > 0 and study_hours <= 2:
        return 'Itna jyaada kyu padh raha hai bhai'
    elif study_hours >= 3 and study_hours <= 5:
        return 'Tum Engineer nahi ban sakte!'
    elif study_hours > 5 and study_hours <= 8:
        return 'Bhai tu hi Engineer banega'
    elif study_hours > 8 and study_hours <= 12:
        return 'Aap IIT aa rahe hai'
    elif study_hours > 12 and study_hours <= 24:
        return 'Padhai ke alawa kuch hai ki nahi zindagi mein?'

def emoji_rain():
    rain(
        emoji = 'STUDY HARD',
        font_size = 50,
        falling_speed = 5,
        animation_length = 0.3
    )
st.markdown("""
    <h1 style='text-align: center; margin-top: -30px; margin-bottom: 30px;'>PassPredictor</h1>
    <h2 style='text-align: center; margin-bottom: 30px;'>A Machine Learning Model<br>for Student Success Prediction</h2>
""", unsafe_allow_html=True)

study_hours = st.slider("Select Study Hours", min_value=0, max_value=24, value=0, step=1)
if study_hours:
    st.write(padhai(int(study_hours)))
    
previous_marks = st.number_input("Enter the marks obtained in previous exams [optional]", min_value = 0, step = 1)


model_path = os.getenv('MODEL_PATH')
scaler_path = os.getenv('SCALER_PATH')

with open(model_path, "rb") as file1:
    model = pickle.load(file1)

with open(scaler_path, "rb") as file2:
    scaler = pickle.load(file2)
    
variables = [[study_hours, previous_marks]]
transformed_data = scaler.transform(variables)

prediction = model.predict_proba(transformed_data)

if st.form_submit_button:
    if prediction[:, 1] >= 0.50 and prediction[:, 1] <= 0.8:
        st.success("Congratulations! Your pass will be approved.")
    elif prediction[:, 1] > 0.8:
        st.success("Bhai tu top maarne waala hai")
        st.balloons()
    elif prediction[:, 1] <= 0.4:
        st.error("Bhai tere se naa ho paayega")
        


st.info("This project was created only for as a part of  project and does not guarantees that student will be able to predict their actual results in the examination. it purely depends on the performance of the student and how well he/she is preparing for their test")
