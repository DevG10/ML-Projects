import streamlit as st
import os
import pickle
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Cardiovascular Disease Sentinel", page_icon=":heart:")

# Write content
st.write("""
# Cardiovascular Disease Sentinel
**This application will determine your risk of getting the CV Diseases**!
""")
st.write('')
st.markdown("""
<style>
    .info-box {
        background-color: red;  
        padding: 10px;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="info-box"><span style="font-size:1.5em;">ℹ</span> Do you know? \n Every year, approximately 17.9 million people die from cardiovascular diseases. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. 4 out of 5 CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years </div>', unsafe_allow_html=True)

st.text('')
st.text('')
st.write("Enter the following details to know your risk of getting CV Diseases")
st.write('----------------------------------------------')

# Collecting user input so that we can proceed to predict the risk

gender = st.radio('Select your Gender', ['Male', 'Female'], horizontal=True)
age = st.slider("What's your Age", 1, 50, 10)
height = st.slider('Height (in cm)', 1, 200, 1)
weight = st.slider('Weight (in kg)', 1, 200, 1)
family_history = st.radio("Do your family members have history with overweight or obesity?", options=[
                          'Yes', 'No'], horizontal=True)
family_history = family_history.lower()
caloric_food = st.radio("Do you frequently eat high caloric food?", options=[
                        'Yes', 'No'], horizontal=True)
caloric_food = caloric_food.lower()
vegetable_consumption = st.radio("How often do you consume vegetables?", options=[
                                 'Rarely', 'Ocassionally', 'Regularly'], horizontal=True)
mapping = {'Rarely': 1, 'Ocassionally': 2, 'Regularly': 3}
vegetable_consumption = mapping[vegetable_consumption]
num_meals = st.slider('How many meals do you take in a day?', 1, 4, 1)
food_in_between_meals = st.radio("Do you often eat food/snacks in between meals?", options=[
                                 'Sometimes', 'Frequently', 'Always', 'No'], horizontal=True)
if food_in_between_meals == 'No':
    food_in_between_meals = food_in_between_meals.lower()
smoke = st.radio("Do you smoke?", options=['Yes', 'No'], horizontal=True)
smoke = smoke.lower()
water_consumption = st.radio("How much water do you consume in a day?", options=[
                             'Less than 1 litre', '1-2 litres', '2-3 litres', 'More than 3 litres'], horizontal=True)
mapping = {'Less than 1 litre': 0, '1-2 litres': 1,
           '2-3 litres': 2, 'More than 3 litres': 3}
water_consumption = mapping[water_consumption]
calories_consumption_monitoring = st.radio(
    "Do you monitor your calories consumption?", options=['Yes', 'No'], horizontal=True)
calories_consumption_monitoring = calories_consumption_monitoring.lower()
physical_activity_frequency = st.radio("How frequently do you engage in physical activities?", options=[
                                       '1 Time a week', '2-3 Times a week', 'More than 3 Times a week'], horizontal=True)
mapping = {'1 Time a week': 0, '2-3 Times a week': 1,
           'More than 3 Times a week': 2}
physical_activity_frequency = mapping[physical_activity_frequency]
time_spent_using_tech_devices = st.radio("How much time do you spend using tech devices in a day?", options=[
                                         'Less than 2 hours', '2-4 hours', '4-6 hours', 'More than 6 hours'], horizontal=True)
mapping = {'Less than 2 hours': 0, '2-4 hours': 1,
           '4-6 hours': 2, 'More than 6 hours': 3}
time_spent_using_tech_devices = mapping[time_spent_using_tech_devices]
alcohol_consumption = st.radio("How often do you consume alcohol?", options=[
                               'Never', 'Sometimes', 'Frequently', 'Always'], horizontal=True)
alcohol_consumption = alcohol_consumption.replace('Never', 'no')
transportation = st.selectbox("What do you prefer for transportation?",   ['Public Transportation', 'Walking', 'Automobile', 'Motorbike',
                                                                           'Bike'])
if transportation == 'Public Transportation':
    transportation = 'Public_Transportation'

# Loading the saved models from the file
model_path = os.getenv('Model_Path')
scaler_path = os.getenv('Scaler_Path')
label_encoder_path = os.getenv('Label_Encoder_Path')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Preprocessing the user input

############### Category Encoding ################
gender_encoded = label_encoder.transform([gender])[0]
family_history_encoded = label_encoder.transform([family_history])[0]
caloric_food_encoded = label_encoder.transform([caloric_food])[0]
food_in_between_meals_encoded = label_encoder.transform(
    [food_in_between_meals])[0]
calories_consumption_monitoring_encoded = label_encoder.transform(
    [calories_consumption_monitoring])[0]
alcohol_consumption_encoded = label_encoder.transform([alcohol_consumption])[0]
transportation_encoded = label_encoder.transform([transportation])[0]
smoke_encoded = label_encoder.transform([smoke])[0]

print("transportation_encoded:", transportation_encoded)
############### Feature Scaling ################
input_data = [[gender_encoded, age, height, weight, family_history_encoded,
               caloric_food_encoded, vegetable_consumption, num_meals,
               food_in_between_meals_encoded, smoke_encoded, water_consumption,
               calories_consumption_monitoring_encoded, physical_activity_frequency,
               time_spent_using_tech_devices, alcohol_consumption_encoded,
               transportation_encoded]]

scaled_data = scaler.transform(input_data)

#################### Prediction ####################
predicted_risk = model.predict(input_data)
# Get the probability of obesity (assuming it's the positive class)
st.write(predicted_risk)
