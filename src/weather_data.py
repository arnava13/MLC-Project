import numpy as np
import requests
import pandas as pd

# create a grid of latitude and longitude points
# from the Sentinel file to get the range of latitudes and longitudes
# the value of interval is to be adjusted
lats = np.arange(40.75, 40.88, 0.01) 
lons = np.arange(-74.01, -73.86, 0.01)
grid_points = [(lat, lon) for lat in lats for lon in lons]

# time window for weather data
start_date = "2021-06-01"
end_date = "2021-09-01"

def get_openmeteo_weather(lat, lon, start_date, end_date):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=America/New_York"
    )
    res = requests.get(url)
    res.raise_for_status()
    return res.json()["daily"]

# get the weather data for each grid point
weather_records = []
for lat, lon in grid_points:
    try:
        print(f"Fetching weather for ({lat}, {lon})")
        daily = get_openmeteo_weather(lat, lon, start_date, end_date)
        for i, date in enumerate(daily["time"]):
            weather_records.append({
                "lat": lat,
                "lon": lon,
                "date": date,
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "precip": daily["precipitation_sum"][i],
            })
    except Exception as e:
        print(f"Failed at ({lat}, {lon}):", e)

# save the weather data to a CSV file
df = pd.DataFrame(weather_records)
df.to_csv("nyc_weather_grid.csv", index=False)
