import pandas as pd
import glob
import os

csv_files = glob.glob('data/raw/trips/*.csv')
output_trips_directory = 'data/interim'
output_stations_directory = 'data/processed'

total_files = len(csv_files)

if not csv_files:
    print("Error: No CSV files found")
    exit()

print(f"PHASE 1: Building Master Station Dictionary (Scanning {total_files} files)")

stations_list = []
# Colums to identify stations
cols_start = ['start_station_id', 'start_station_name', 'start_lat', 'start_lng']
cols_end   = ['end_station_id', 'end_station_name', 'end_lat', 'end_lng']

for i, file in enumerate(csv_files, 1):
    filename = os.path.basename(file)
    print(f"[{i}/{total_files}] Scanning stations in: {filename}...", end='\r')

    df_temp = pd.read_csv(file, usecols=cols_start + cols_end, low_memory=False)

    # Standardize column names for concatenation
    start_stations = df_temp[cols_start].rename(columns={
        'start_station_id': 'station_id', 'start_station_name': 'station_name', 
        'start_lat': 'lat', 'start_lng': 'lng'
    })
    end_stations = df_temp[cols_end].rename(columns={
        'end_station_id': 'station_id', 'end_station_name': 'station_name', 
        'end_lat': 'lat', 'end_lng': 'lng'
    })

    # Combine and clean
    combined_stations = pd.concat([start_stations, end_stations])

    # Transform ID to numeric and drop rows without a valid ID
    combined_stations['station_id'] = pd.to_numeric(combined_stations['station_id'], errors='coerce')
    combined_stations = combined_stations.dropna(subset=['station_id'])

    # Keep the first occurrence of each ID
    combined_stations = combined_stations.sort_values('station_name').drop_duplicates(subset=['station_id'], keep='first')
    
    stations_list.append(combined_stations)

print(f"\n\n>> Consolidating station data...")

if stations_list:
    df_stations = pd.concat(stations_list)
    # Get the first record per Station ID
    df_stations = df_stations.groupby('station_id', as_index=False).first()

    # Save station master file
    stations_csv = os.path.join(output_stations_directory, 'stations.csv')
    df_stations.to_csv(stations_csv, index=False)
    
    # Create dictionaries for mapping
    map_names = df_stations.set_index('station_id')['station_name'].to_dict()
    map_lat = df_stations.set_index('station_id')['lat'].to_dict()
    map_lng = df_stations.set_index('station_id')['lng'].to_dict()
    
    print(f"Stations list saved with: {len(df_stations)} unique stations")
    
    # Free memory
    del stations_list, df_stations, combined_stations, start_stations, end_stations, df_temp
else:
    print("Error: No CSV files found or empty lists.")
    exit()

print(f"PHASE 2: Processing Trips, Cleaning & Converting to Parquet")

for i, file in enumerate(csv_files, 1):
    filename = os.path.basename(file)
    print(f"\n[{i}/{total_files}] Processing: {filename}")

    df = pd.read_csv(file, low_memory=False)
    initial_rows = len(df)

    # -----------------------------
    # 1. TYPE CASTING AND FORMATTING
    # -----------------------------

    # Cast IDs to string
    df['ride_id'] = df['ride_id'].astype(str)
    
    # Convert low-cardinality columns to 'category'
    df['rideable_type'] = df['rideable_type'].astype('category')
    df['member_casual'] = df['member_casual'].astype('category')

    # Convert timestamps
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

    # Standardize Station IDs to numeric
    df['start_station_id'] = pd.to_numeric(df['start_station_id'], errors='coerce')
    df['end_station_id'] = pd.to_numeric(df['end_station_id'], errors='coerce')

    # -----------------------------
    # 2. FEATURE ENGINEERING
    # -----------------------------

    # Calculate trip duration in seconds
    df['trip_duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds()

    # -----------------------------
    # 3. DATA CLEANING AND FILTERING
    # -----------------------------

    # Drop rows with corrupted dates (NaT)
    df = df.dropna(subset=['started_at', 'ended_at'])

    # Identify invalid trips (Duration < 3min, > 3h, or Same Origin/Dest)
    invalid_mask = (
        (df['trip_duration'] < 180) | 
        (df['trip_duration'] > 10800) | 
        (df['start_station_id'] == df['end_station_id'])
    )

    num_invalid = invalid_mask.sum()

    # Keep only valid trips using the mask
    df = df[~invalid_mask]

    # -----------------------------
    # 4. MISSING DATA IMPUTATION
    # -----------------------------

    # Fill missing Station Names/Coords using the Map from Phase 1
    df['start_station_name'] = df['start_station_name'].fillna(df['start_station_id'].map(map_names))
    df['start_lat'] = df['start_lat'].fillna(df['start_station_id'].map(map_lat))
    df['start_lng'] = df['start_lng'].fillna(df['start_station_id'].map(map_lng))
    
    df['end_station_name'] = df['end_station_name'].fillna(df['end_station_id'].map(map_names))
    df['end_lat'] = df['end_lat'].fillna(df['end_station_id'].map(map_lat))
    df['end_lng'] = df['end_lng'].fillna(df['end_station_id'].map(map_lng))

    # Drop rows where Station ID is still missing (irrecoverable data)
    rows_before_final_drop = len(df)
    df = df.dropna(subset=['start_station_id', 'end_station_id'])
    rows_no_id = rows_before_final_drop - len(df)

    # -----------------------------
    # 5. SAVING OUTPUT
    # -----------------------------

    base_name = os.path.basename(file).replace('.csv', '.parquet')
    new_file_route = os.path.join(output_trips_directory, base_name)
    df.to_parquet(new_file_route, index=False)

    # Statistics logs
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows

    print(f"   -> Original rows: {initial_rows:,}")
    print(f"   -> Invalid trips removed: {num_invalid:,}")
    print(f"   -> Irrecoverable IDs removed: {rows_no_id:,}")
    print(f"   -> Final saved rows: {final_rows:,} (Dropped: {dropped_rows:,})")

    del df

print("Process completed successfully")