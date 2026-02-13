import pandas as pd
import glob
import os

TRIPS_FILES = "data/processed/trips_enriched/2025*-citibike-tripdata.parquet"
STATIONS_FILE = "data/processed/stations.csv"
OUTPUT_FILE = "data/processed/real_routes.parquet"

def main():
    files = glob.glob(TRIPS_FILES)

    if not files:
        print("No trip files found.")
        return
    
    total_pairs = []

    for file in files:
        print(f"Processing file: {file}")
        try:
            df = pd.read_parquet(file, columns=['start_station_id', 'end_station_id'])
            df.dropna()
            unique_month = df.drop_duplicates()
            total_pairs.append(unique_month)

        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    print("Combining unique station pairs from all files...")

    df_global = pd.concat(total_pairs, ignore_index=True)
    df_final = df_global.drop_duplicates().copy()
    print(f"Total unique station pairs: {len(df_final)}")

    df_stations = pd.read_csv(STATIONS_FILE)
    
    df_final['start_station_id'] = df_final['start_station_id'].astype(str)
    df_final['end_station_id'] = df_final['end_station_id'].astype(str)
    df_stations['station_id'] = df_stations['station_id'].astype(str)

    # Merge origin
    df_final = df_final.merge(
        df_stations[['station_id', 'lat', 'lng']], 
        left_on='start_station_id', right_on='station_id',
        how='inner'
    ).rename(columns={'lat': 'lat_origin', 'lng': 'lon_origin'})

    # Merge destination
    df_final = df_final.merge(
        df_stations[['station_id', 'lat', 'lng']], 
        left_on='end_station_id', right_on='station_id',
        how='inner',
        suffixes=('_start', '_end')
    ).rename(columns={'lat': 'lat_destination', 'lng': 'lon_destination'})

    
    columns = ['start_station_id', 'end_station_id', 'lat_origin', 'lon_origin', 'lat_destination', 'lon_destination']
    df_final = df_final[columns]

    df_final.to_parquet(OUTPUT_FILE, index=False)
    print("Route data saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()