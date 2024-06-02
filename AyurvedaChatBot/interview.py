import requests

def get_gender(name):
    response = requests.get(f"https://api.genderize.io?name={name}")
    data = response.json()
    return data['gender']

name = "Sanjay"
gender = get_gender(name)
print(f"The predicted gender for the name {name} is {gender}")
