import pandas as pd
import glob
import requests
import concurrent.futures
import time
import os
import gc

INPUT_FILE = "data/processed/real_routes.parquet"
OUTPUT_DIR = "data/processed/matrix_routes"
GH_URL = "http://localhost:8989/route"

# All existing profiles
PROFILES = ["lts1_kids", "lts2_safe", "lts3_confident", "lts4_direct"]

MAX_WORKERS = 24
CHUNK_SIZE = 2000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR) 

thread_local = os.sys.modules['threading'].local()

def get_session():
    """
    Get a thread-local session to reuse connections for better performance.
    """
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def get_geometry_wkt(paths_json):
    """
    Convert the 'paths' JSON from GraphHopper into a list of geometries with their associated profile.
    """
    try:
        points = paths_json['points']['coordinates']
        coords_str = ",".join([f"{round(p[0],5)} {round(p[1],5)}" for p in points])
        return f"LINESTRING({coords_str})"
    except:
        return None
    
def calculate_pair(row):
    """
    Calculate the path for a given pair of stations and all profiles.
    """ 
    session = get_session()

    start_id = row['start_station_id']
    end_id = row['end_station_id']
    coord_start = f"{row['lat_origin']},{row['lon_origin']}"
    coord_end = f"{row['lat_destination']},{row['lon_destination']}"

    results = []

    for profile in PROFILES:
        data = {
            "start_station_id": start_id,
            "end_station_id": end_id,
            "profile": profile,
            "time_min": None,
            "distance_m": None,
            "geometry": None
        }

        try:
            params = {
                "point": [coord_start, coord_end],
                "profile": profile,
                "locale": "es",
                "calc_points": True,
                "points_encoded": False,
                "elevation": False
            }

            resp = session.get(GH_URL, params=params, timeout=5)

            if resp.status_code == 200 :
                json_data = resp.json()
                path = json_data['paths'][0]

                data['time_min'] = round(path['time'] / 60000, 3)
                data['distance_m'] = round(path['distance'], 2)
                data['geometry'] = get_geometry_wkt(path)
            else:
                print(f"Error {resp.status_code} for {start_id} -> {end_id} with profile {profile}")
                pass

        except Exception as e:
            print(f"Exception for {start_id} -> {end_id} with profile {profile}: {e}")
            pass

        results.append(data)
    
    return results

def process_chunk(df_chunk, batch_num): 
    """ 
    Process a chunk of station pairs and save the results to a batch file. 
    """
    buffer = []
    path_list = df_chunk.to_dict('records')

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iter = executor.map(calculate_pair, path_list) 
        for result in results_iter: 
            buffer.extend(result) 
    
    if buffer:
        filename = os.path.join(OUTPUT_DIR, f"batch_{batch_num}.parquet")
        pd.DataFrame(buffer).to_parquet(filename, index=False)
        return len(buffer)
    
    return 0

def main():
    print("Initializing route calculations...")

    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Please run the data processing step first.")
        return
    
    stations_df = pd.read_parquet(INPUT_FILE)
    total_paths = len(stations_df)
    print(f"Total paths to process: {total_paths}")


    batches_done = glob.glob(os.path.join(OUTPUT_DIR, "batch_*.parquet"))
    last_batch = -1
    for b in batches_done:
        try:
            n = int(os.path.basename(b).split('_')[1].split('.')[0])
        except: pass

    start_idx = (last_batch + 1) * CHUNK_SIZE

    if start_idx >= total_paths:
        print("Process already done")
        return
    
    print(f"Starting from {start_idx} (Batch {last_batch + 1})")

    start_time_global = time.time()

    for i in range(start_idx, total_paths, CHUNK_SIZE):
        batch_num = i // CHUNK_SIZE

        df_chunk = stations_df.iloc[i : i + CHUNK_SIZE]

        print(f"Processing Batch {batch_num} (Routes {i} to {min(i+CHUNK_SIZE, total_paths)})")

        t0 = time.time()
        n_rows = process_chunk(df_chunk, batch_num)
        t1 = time.time()

        elapsed = t1 - t0
        rutas_processed = len(df_chunk)
        speed = rutas_processed / elapsed

        print(f"Saved batch_{batch_num}.parquet in {elapsed:.1f}s ({speed:.1f} routes/s)")

        del df_chunk
        gc.collect()

    print("Route calculations completed.")

if __name__ == "__main__":
    main()
