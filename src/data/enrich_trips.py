import pandas as pd
import glob
import os
import requests
from io import StringIO
from astral import LocationInfo
from astral.sun import sun
import numpy as np

input_pattern = 'data/interim/2025*-citibike-tripdata_*.parquet'
output_dir = 'data/processed/trips_enriched'
os.makedirs(output_dir, exist_ok=True)

# NYC coordinates
NYC_LAT =  40.7143
NYC_LON = -74.006
NYC_TZ = 'America/New_York'

year = 2025

# ============================================
# FUNCTION 1: WEATHER DATA DOWNLOAD
# ============================================
def get_nyc_weather_history():
    """
    Downloads hourly weather data for NYC for the entire year from Open-Meteo.
    Returns a DataFrame with index aligned to nearest hour.
    """
    print("Downloading weather data for NYC...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "cloud_cover"],
        "timezone": NYC_TZ,
        "timeformat": "iso8601"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to download weather: {response.text}")
    
    # Convert JSON to DataFrame
    data = response.json()
    hourly = data['hourly']

    df_weather = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'temperature_c': hourly['temperature_2m'],
        'precipitation_mm': hourly['precipitation'],
        'wind_speed_kmh': hourly['wind_speed_10m'],
        'relative_humidity_%': hourly['relative_humidity_2m'],
        'cloud_cover_%': hourly['cloud_cover']
    })
    
    df_weather.set_index('time', inplace=True)

    print(f"Weather data ready: {len(df_weather)} hourly records")
    return df_weather

# ============================================
# FUNCTION 2: SUN CYCLES
# ============================================
def get_sun_cycles():
    """
    Pre-calculates sunrise and sunset for every day of the year in NYC.
    """
    print("Calculating solar cycles (Sunrise/Sunset)...")
    city = LocationInfo("NYC", "USA", NYC_TZ, NYC_LAT, NYC_LON)

    # Create a range of dates for the year
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')

    sun_data = []

    for date in dates:
        s = sun(city.observer, date=date, tzinfo=city.timezone)
        sun_data.append({
            'date_key': date.date(),
            'sunrise': pd.to_datetime(s['sunrise']).replace(tzinfo=None),
            'sunset': pd.to_datetime(s['sunset']).replace(tzinfo=None)
        })

    df_sun = pd.DataFrame(sun_data)
    print("Solar cycles calculated")
    return df_sun

# ============================================
# MAIN PROCESS
# ============================================
try:
    df_weather = get_nyc_weather_history()
    df_sun = get_sun_cycles()
except Exception as e:
    print(f"Error preparing enrichment data: {e}")
    exit()

# Identify all .parquet files
all_files = sorted(glob.glob(input_pattern))
if not all_files:
    print("No files found")
    exit()

# Extract unique months present in the files
months = sorted(list(set([os.path.basename(f).split('-')[0] for f in all_files])))

print(f"\nFound files for {len(months)} months. Starting aggregation and enrichment...\n")

for month in months:
    print(f"--- Processing Month: {month} ---")

    # Load all sub-files for this month
    month_files = [f for f in all_files if os.path.basename(f).startswith(month)]
    print(f"   Merging {len(month_files)} sub-files...")
    
    df_month = pd.read_parquet(month_files)

    # Drop redundant station info
    cols_to_drop = [
        'start_station_name', 'start_lat', 'start_lng', 
        'end_station_name', 'end_lat', 'end_lng'
    ]
    df_month.drop(columns=[c for c in cols_to_drop if c in df_month.columns], inplace=True)

    # Enrich with weather
    # Create a temporary column rounded to nearest hour to join with weather
    df_month['temp_time'] = df_month['started_at'].dt.round('h')

    df_month = df_month.merge(df_weather, left_on='temp_time', right_index=True, how='left')

    # Enrich with day/night
    # Create a temporary date key for merging
    df_month['temp_date'] = df_month['started_at'].dt.date

    df_month = df_month.merge(df_sun, left_on='temp_date', right_on='date_key', how='left')

    # Apply logic: Is it Day or Night?
    # Logic: It is day if started_at is between sunrise and sunset
    # 1 day, 0 night
    df_month['is_day'] = (df_month['started_at'] >= df_month['sunrise']) & (df_month['started_at'] <= df_month['sunset'])
    df_month['is_day'] = df_month['is_day'].astype(int)

    # Remove auxiliary columns
    df_month.drop(columns=['temp_time', 'temp_date', 'date_key', 'sunrise', 'sunset'], inplace=True)

    # Weekend flag
    # 1 weekend, 0 weekday
    df_month['is_weekend'] = df_month['started_at'].dt.dayofweek.isin([5,6]).astype(int)

    # Save final Parquet
    output_file = os.path.join(output_dir, f"{month}-citibike-tripdata.parquet")
    df_month.to_parquet(output_file, index=False)
    
    print(f"   Saved: {os.path.basename(output_file)}")
    print(f"   Rows: {len(df_month):,}")

print("\nProcess Completed. Data is enriched and optimized")