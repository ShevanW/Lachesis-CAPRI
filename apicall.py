import requests

location_id = ['@12691', '@8519', '@13046', 'A110542', '@3247', '@3243', 'A110227', '@3244', '@3240', '@13053', '@3242', '@13782', '@13780', 'A490561', '@10141', '@3234', '@8518', '@3236', 'A189871', '@3235', '@3237', 'A416350', 'A481600', '@3238', '@8009', '@4749', '@8010', '@4750', '@3248', '@12685'] 

# ====== CONFIGURE THIS ======
API_KEY = "4fa9ac50f09c5dfc3a41b780793ecfff46af4148"
# ============================
import time

BASE_URL = "https://api.waqi.info/feed/"

all_data = []

# Loop through cities in Victoria
for location in location_id:
    url = f"https://api.waqi.info/feed/{location}/?token={API_KEY}"
    data = requests.get(url).json()

    all_data.append(data)

for data in all_data:
    city = data["data"]["city"]["name"]
    aqi_us = data["data"]["aqi"]
    dominant_pol = data["data"]["dominentpol"]
    print(f"City: {city} | AQI: {aqi_us} | Dominant Pollutant: {dominant_pol}")