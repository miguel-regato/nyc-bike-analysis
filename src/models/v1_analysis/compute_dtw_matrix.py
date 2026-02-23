import pandas as pd
import numpy as np
from tslearn.metrics import cdist_dtw
import os
import time

INPUT_DIR = 'data/processed/clustering'
OUTPUT_DIR = 'data/processed/clustering'

FILES_TO_PROCESS = {
    'weekday': 'signatures_weekday.csv',
    'weekend': 'signatures_weekend.csv'
}

def compute_matrix():
    print("Starting DTW matrix computation...")

    for key, filename in FILES_TO_PROCESS.items():
        input_path = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(input_path):
            print(f"File {input_path} not found. Skipping {key}.")
            continue

        print(f"Processing {key} data from {input_path}...")

        # Load the data
        df = pd.read_csv(input_path, index_col=0)

        # Save the index for later use
        ids = df.index.to_numpy()

        # Prepare data
        X = df.values
        X = X.reshape(X.shape[0], X.shape[1], 1)

        print(f"Loaded data: {X.shape[0]} stations {X.shape[1]} hours.")
        print("Calculating DTW distance matrix...")

        start_time = time.time()

        # Calculate the DTW distance matrix
        dtw_matrix = cdist_dtw(X)

        elapsed_time = time.time() - start_time
        minutes = elapsed_time / 60
        print(f"DTW matrix computed in {minutes:.2f} minutes.")
        print(f"DTW matrix shape: {dtw_matrix.shape}")

        # Save the DTW matrix and IDs
        output_matrix_path = os.path.join(OUTPUT_DIR, f'dtw_matrix_{key}.npy')
        output_ids_path = os.path.join(OUTPUT_DIR, f'station_ids_{key}.npy')

        np.save(output_matrix_path, dtw_matrix)
        np.save(output_ids_path, ids)

if __name__ == "__main__":
    compute_matrix()