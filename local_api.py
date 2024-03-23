import json
import main
import requests
import asyncio
# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

#print the status code
print(r.status_code)
# print the welcome message
asyncio.run(main.greetings())
print(asyncio.run(main.greetings()))



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# send a POST using the data above
p = requests.post("http://127.0.0.1:8000",json = data)

# print the status code
print(p.status_code)
# print the result
print(asyncio.run(main.post_inference()))
