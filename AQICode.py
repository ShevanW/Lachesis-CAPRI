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