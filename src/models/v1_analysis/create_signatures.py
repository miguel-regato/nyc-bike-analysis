import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler

INPUT_PATH = 'data/processed/trips_enriched/*.parquet'
OUTPUT_DIR = 'data/processed/clustering'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_signatures():
    files = glob.glob(INPUT_PATH)
    print(f"Detected {len(files)} parquet files")

    df_list = []
    for f in files:
        df = pd.read_parquet(f, columns=['start_station_id', 'started_at', 'is_weekend'])
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Total trips loaded: {len(df_all)}")

    df_all['hour'] = df_all['started_at'].dt.hour
    df_all = df_all.dropna(subset=['start_station_id'])

    for day_week, is_weekend in [('weekday', 0), ('weekend', 1)]:
        print(f"\n Processing for {day_week}")

        # Filter data
        subset = df_all[df_all['is_weekend'] == is_weekend]

        # Count trips per station and hour
        hourly_counts = subset.groupby(['start_station_id', 'hour']).size().reset_index(name='count')
        
        # Pivot data
        pivot = hourly_counts.pivot(index='start_station_id', columns='hour', values='count').fillna(0)

        # Make sure all hours are present
        for h in range(24):
            if h not in pivot.columns:
                pivot[h] = 0

        # Sort columns by hour
        pivot = pivot[sorted(pivot.columns)]

        # Normalize using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(pivot.T).T

        # Make final DataFrame
        df_signature = pd.DataFrame(normalized_values, 
                                    index=pivot.index, 
                                    columns=[f"h_{h:02d}" for h in range(24)]
                                    )
        
        # Save
        out_file = os.path.join(OUTPUT_DIR, f'signatures_{day_week}.csv')
        df_signature.to_csv(out_file, index=True)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    generate_signatures()