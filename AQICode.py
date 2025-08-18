import requests

# ====== CONFIGURE THIS ======
API_KEY = "e8c53900-b9f2-430e-89cf-d2fece3e06d4"
# ============================

url = f"https://api.airvisual.com/v2/nearest_city?key={API_KEY}"

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise error for bad responses
    data = response.json()

    if data.get("status") == "success":
        city = data['data']['city']
        state = data['data']['state']
        country = data['data']['country']

        aqi_us = data['data']['current']['pollution']['aqius']
        main_pollutant = data['data']['current']['pollution']['mainus']

        temp_c = data['data']['current']['weather']['tp']
        humidity = data['data']['current']['weather']['hu']
        wind_speed = data['data']['current']['weather']['ws']

        print(f"Location: {city}, {state}, {country}")
        print(f"Temperature: {temp_c}°C")
        print(f"Humidity: {humidity}%")
        print(f"Wind Speed: {wind_speed} m/s")
        print(f"AQI (US Standard): {aqi_us}")
        print(f"Main Pollutant: {main_pollutant.upper()}")
    else:
        print("❌ API request failed:", data.get("data", "Unknown error"))

except requests.exceptions.RequestException as e:
    print("⚠️ Request error:", e)
    
    
    
import time

BASE_URL = "https://api.airvisual.com/v2"
STATE = "Victoria"  # Target state

# Get all cities in Victoria
cities_url = f"{BASE_URL}/cities?state={STATE}&country=Australia&key={API_KEY}"
cities_data = requests.get(cities_url).json()

if cities_data["status"] != "success":
    print(f"Error getting cities for {STATE}:", cities_data)
    exit()

all_data = []

# Loop through cities in Victoria
for city_info in cities_data["data"]:
    city = city_info["city"]
    print(f"Fetching air quality for city: {city}")

    city_url = f"{BASE_URL}/city?city={city}&state={STATE}&country=Australia&key={API_KEY}"
    city_data = requests.get(city_url).json()

    if city_data["status"] == "success":
        all_data.append(city_data["data"])

    # Delay to avoid rate limit
    time.sleep(1)

# Display results
print("\nAir Quality Data for Victoria, Australia:\n")
for entry in all_data:
    city = entry['city']
    aqi_us = entry['current']['pollution']['aqius']
    main_pollutant = entry['current']['pollution']['mainus']
    temp_c = entry['current']['weather']['tp']
    humidity = entry['current']['weather']['hu']

    print(f"{city} → AQI: {aqi_us}, Main Pollutant: {main_pollutant.upper()}, Temp: {temp_c}°C, Humidity: {humidity}%")