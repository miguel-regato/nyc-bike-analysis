import pandas as pd
import numpy as np
import glob
import os
import gc

TRIPS_FOLDER = "data/processed/trips_enriched/"
ROUTES_FOLDER = "data/processed/matrix_routes/"
OUTPUT_FOLDER = "data/processed/trips_final/"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def load_routes_data():
    """
    Load all routes and prepare two structures:
    1. routes_wide: To quickly calculate which is the best profile.
    2. routes_long: To recover the geometry and distance once the profile has been chosen.
    """
    print("Loading routes data...")
    df_routes = pd.read_parquet(ROUTES_FOLDER, columns=['start_station_id', 'end_station_id', 'profile', 'time_min', 'distance_m'])

    df_routes['start_station_id'] = df_routes['start_station_id'].astype(str)
    df_routes['end_station_id'] = df_routes['end_station_id'].astype(str)

    df_routes['duration_sec'] = df_routes['time_min'] * 60

    # Create wide table
    print("     Creating wide table for quick comparisons")
    routes_wide = df_routes.pivot_table(
        index=['start_station_id', 'end_station_id'],
        columns='profile',
        values='duration_sec'
    ).reset_index()

    # Rename columns
    columns_profiles = [c for c in routes_wide.columns if c not in ['start_station_id', 'end_station_id']]
    routes_wide.columns = ['start_station_id', 'end_station_id'] + [f"time_{c}" for c in columns_profiles]

    routes_long = df_routes.set_index(['start_station_id', 'end_station_id', 'profile'])[['distance_m']]

    return routes_wide, routes_long, columns_profiles

def process_month(file_path, routes_wide, routes_long, profile_names):
    filename = os.path.basename(file_path)
    print(f"Processing file: {filename}")

    df_trips = pd.read_parquet(file_path)

    df_trips['start_station_id'] = df_trips['start_station_id'].astype(str)
    df_trips['end_station_id'] = df_trips['end_station_id'].astype(str)

    # Merge wide table with months trips
    df_merged = df_trips.merge(
        routes_wide,
        on=['start_station_id', 'end_station_id'],
        how='left'
    )

    time_cols = [f"time_{p}" for p in profile_names]

    # Calculate the absolute difference
    diffs = pd.DataFrame()
    for p_col in time_cols:
        diffs[f"diff_{p_col}"] = (df_merged['trip_duration'] - df_merged[p_col]).abs()

    # Get the profile with the minimum difference
    best_col = diffs.idxmin(axis=1)
    df_merged['estimated_profile'] = best_col.str.replace('diff_time_', '', regex=False)
    df_merged['estimation_error_sec'] = diffs.min(axis=1)

    # Merge with long table to get geometry and distance
    df_merged = df_merged.join(
        routes_long, 
        on=['start_station_id', 'end_station_id', 'estimated_profile'],
        how='left'
    )
    
    cols_to_drop = time_cols
    df_final = df_merged.drop(columns=cols_to_drop)
    
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    df_final.to_parquet(output_path, index=False)
    
    del df_trips, df_merged, df_final, diffs
    gc.collect()

def main():
    routes_wide, routes_long, profile_names = load_routes_data()
    trip_files = sorted(glob.glob(os.path.join(TRIPS_FOLDER, "*.parquet")))

    if not trip_files:
        print("No trip files were found.")
        return
    
    for f in trip_files:
        try:
            process_month(f, routes_wide, routes_long, profile_names)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("\nAll files processed.")

if __name__ == "__main__":
    main()