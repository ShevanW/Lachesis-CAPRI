import requests, time, pandas as pd
from datetime import datetime

API_KEY = "e8c53900-b9f2-430e-89cf-d2fece3e06d4"
BASE_URL = "https://api.airvisual.com/v2"
STATE = "Victoria"
DELAY_SEC = 1
TIMEOUT = 10
RETRIES = 1

code_map = {
    "p2": "PM2.5",
    "p1": "PM10",
    "o3": "O₃",
    "n2": "NO₂",
    "s2": "SO₂",
    "co": "CO"
}

def get_json(url, timeout=TIMEOUT, retries=RETRIES):
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)  # brief backoff then retry
    raise last_err

# 1) Get cities in Victoria
cities_url = f"{BASE_URL}/cities?state={STATE}&country=Australia&key={API_KEY}"
cities_data = get_json(cities_url)

if cities_data.get("status") != "success":
    print("Failed to fetch city list:", cities_data)
    raise SystemExit

rows, failed = [], []

# 2) Loop cities with delay
for item in cities_data["data"]:
    city = item["city"]
    print(f"Fetching: {city}")
    city_url = f"{BASE_URL}/city?city={city}&state={STATE}&country=Australia&key={API_KEY}"
    try:
        d = get_json(city_url)
        if d.get("status") == "success":
            dat = d["data"]
            pol = dat["current"]["pollution"]
            wea = dat["current"]["weather"]

            main_code = (pol.get("mainus") or "").lower()
            rows.append({
                "city": dat.get("city"),
                "state": dat.get("state"),
                "country": dat.get("country"),
                "aqi_us": pol.get("aqius"),
                "main_pollutant_code": main_code.upper(),
                "main_pollutant": code_map.get(main_code, main_code.upper()),
                "temp_c": wea.get("tp"),
                "humidity_%": wea.get("hu"),
                "wind_mps": wea.get("ws"),
                "time_pollution": pol.get("ts"),
                "time_weather": wea.get("ts"),
                "lat": dat.get("location", {}).get("coordinates", [None, None])[1] if dat.get("location") else None,
                "lon": dat.get("location", {}).get("coordinates", [None, None])[0] if dat.get("location") else None,
            })
        else:
            failed.append((city, d))
    except Exception as e:
        failed.append((city, str(e)))

    time.sleep(DELAY_SEC)

# 3) To DataFrame + sort
df = pd.DataFrame(rows)
df_sorted = df.sort_values("aqi_us", ascending=False, na_position="last")

# 4) Save to Excel
stamp = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"victoria_aqi_{stamp}.xlsx"
with pd.ExcelWriter(fname, engine="xlsxwriter") as xlw:
    df_sorted.to_excel(xlw, index=False, sheet_name="Victoria AQI")

print("\nSaved:", fname)
print(f"Records: {len(df_sorted)} | Failed: {len(failed)}")
if failed:
    print("Some cities failed/skipped (showing up to 10):")
    for c, err in failed[:10]:
        print(" -", c, "→", err)

# 5) Console summary (worst 10 by AQI)
print("\nTop 10 worst AQI (higher is worse):")
print(df_sorted[["city", "aqi_us", "main_pollutant"]].head(10).to_string(index=False))

# And best 10
print("\nTop 10 best AQI:")
print(df_sorted[["city", "aqi_us", "main_pollutant"]].tail(10).to_string(index=False))