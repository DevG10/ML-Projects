import streamlit as st
from streamlit_extras.let_it_rain import rain
import os
import pickle
from dotenv import load_dotenv
import time
load_dotenv()

def rain_fun():
    rain(
        emoji="🥲",
        font_size=50,
        falling_speed=6,
        animation_length=3
    )
    
st.set_page_config(
    page_title="Cardiovascular Disease Sentinel", 
    page_icon=":heart:", 
    layout="wide"
    )

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



st.info(
    "Every year, approximately 17.9 million people die from cardiovascular diseases. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. 4 out of 5 CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years", icon='ℹ')

st.text('')
st.text('')
st.write("Enter the following details to know your risk of getting CV Diseases")
st.write('----------------------------------------------')

# Collecting user input so that we can proceed to predict the risk

gender = st.radio('Select your Gender', ['Male', 'Female'], horizontal=True)
age = st.slider("What's your Age", 10, 50, 10)
height = st.slider('Height (in cm)', 150, 200, 150)
height = height/100

weight = st.slider('Weight (in kg)', 30, 200, 30)
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

# Loading the saved models from the env variables
model_path = os.getenv('Model_Path')
scaler_path = os.getenv('Scaler_Path')
gender_encoder_path = os.getenv('gender_encoder')
family_history_encoder_path = os.getenv('family_history_with_overweight_encoder')
caloric_food_encoder_path = os.getenv('caloric_food_encoder')
food_in_between_meals_encoder_path = os.getenv('food_between_meals_encoder')
calory_monitor_encoder = os.getenv('colry_monitoring_encoder')
smoking_encoder_path = os.getenv('smoke_encoder')
mtrans_encoder_path = os.getenv('mtrans_encoder')
alcohol_encoder_path = os.getenv('alcohol_encoder')


with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(gender_encoder_path, 'rb') as file:
    gender_encoder = pickle.load(file)

with open(family_history_encoder_path, 'rb') as file:
    family_history_encoder = pickle.load(file)

with open(caloric_food_encoder_path, 'rb') as file:
    caloric_food_encoder = pickle.load(file)

with open(food_in_between_meals_encoder_path, 'rb') as file:
    food_in_between_meals_encoder = pickle.load(file)

with open(smoking_encoder_path, 'rb') as file:
    smoking_encoder = pickle.load(file)

with open(mtrans_encoder_path, 'rb') as file:
    mtrans_encoder = pickle.load(file)
    
with open(calory_monitor_encoder, 'rb') as file:
    caloric_food_encoder = pickle.load(file)

with open(alcohol_encoder_path, 'rb') as file:
    alcohol_encoder = pickle.load(file)

# Preprocessing the user input

############### Category Encoding ################
gender_encoded = gender_encoder.transform([[gender]])[0]
family_history_encoded = family_history_encoder.transform([[family_history]])[0]
caloric_food_encoded = caloric_food_encoder.transform([[caloric_food]])[0]
food_in_between_meals_encoded = int(food_in_between_meals_encoder.transform(
    [[food_in_between_meals]])[0])
calories_consumption_monitoring_encoded = caloric_food_encoder.transform(
    [[calories_consumption_monitoring]])[0]
alcohol_consumption_encoded = int(alcohol_encoder.transform([[alcohol_consumption]])[0])
transportation_encoded = int(mtrans_encoder.transform([[transportation]])[0])
smoke_encoded = smoking_encoder.transform([[smoke]])[0]
calory_monitor_encoded = caloric_food_encoder.transform([[calories_consumption_monitoring]])[0]

############### Feature Scaling ################

scaled_data = scaler.transform([[gender_encoded, age, height, weight, family_history_encoded, caloric_food_encoded, vegetable_consumption, num_meals, food_in_between_meals_encoded, smoke_encoded, water_consumption, calories_consumption_monitoring_encoded, physical_activity_frequency, time_spent_using_tech_devices, alcohol_consumption_encoded, transportation_encoded]])

############### Messages ################
normal= "Congratulations you are having a normal health status. Keep it up!"
insuficient = "Your health is insuficient. You need to improve your lifestyle to avoid health risks. Eat healthy food, exercise regularly and consult a doctor for further advice."
overweight_1 = "You are having overweight level I. Take care of your health by eating healthy food and exercise regularly."
overweight_2 = "You are having overweight level II. Kindly exercise regularly and consult a doctor for further advice."
obesity_type_1 = "You are having Class I obesity which means your BMI is 30 to <35 kg/m². Avoid high caloric food and consult a doctor for further advice."
obesity_type_2 = "You are having Class II obesity which means your BMI is 35 to <40 kg/m². Please consult a doctor for further advice."
obesity_type_3 = "You are having Class III obesity which means your BMI is 40+ kg/m². This is the most severe form of obesity kindly consult a doctor for further advice."

def stream_words(word):
     for word in word.split(" "):
        yield word + " "
        time.sleep(0.09)

#################### Prediction ####################
if st.button('Get the results', help = 'Predict the results'):
    prediction = model.predict(scaled_data)[0]
    progress = st.progress(0, text='Operation in Progress please wait...')
    for percent_complete in range(1,100):
        time.sleep(0.01)
        progress.progress(percent_complete, text = f"Progress: {percent_complete + 1}%")
    time.sleep(1)
    progress.empty()
    container = st.empty()
    
    if prediction == 'Normal_Weight':
        st.write(stream_words(normal))
        st.balloons()
    elif prediction == 'Insufficient_Weight':
        st.write(stream_words(insuficient))
    elif prediction == 'Overweight_Level_I':
        st.write(stream_words(overweight_1))
    elif prediction == 'Overweight_Level_II':
        st.write(stream_words(overweight_2))
    elif prediction == 'Obesity_Type_I':
        st.write(stream_words(obesity_type_1))
        rain_fun()
    elif prediction == 'Obesity_Type_II':
        st.write(stream_words(obesity_type_2))
    elif prediction == 'Obesity_Type_III':
        st.write(stream_words(obesity_type_3))
        rain_fun()
