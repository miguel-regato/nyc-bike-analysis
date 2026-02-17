import pandas as pd
import numpy as np
import glob
import os

INPUT_PATH = 'data/processed/trips_final/'
OUTPUT_PATH = 'data/processed/trips-final-cleaned/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

dist_bins = [0, 500, 1500, 3000, np.inf]
dist_labels = ['Very Short (< 500m)', 'Short (500m-1.5km)', 'Medium (1.5km-3km)', 'Long (> 3km)']

VERY_SHORT_LIMIT = 690

all_files = glob.glob(os.path.join(INPUT_PATH, '*.parquet'))

df_list = []
for file in all_files:
    df_chunk = pd.read_parquet(file, columns=['distance_m', 'estimation_error_sec'])
    df_list.append(df_chunk)

full_df = pd.concat(df_list, ignore_index=True)
full_df['dist_group_error'] = pd.cut(
    full_df['distance_m'],
    bins=dist_bins,
    labels=dist_labels,
    right=True
)

upper_limits = {}

categories_to_check = ['Short (500m-1.5km)', 'Medium (1.5km-3km)', 'Long (> 3km)']

for cat in categories_to_check:
    subset = full_df[full_df['dist_group_error'] == cat]['estimation_error_sec']

    if len(subset) > 0:
        Q1 = subset.quantile(0.25)
        Q3 = subset.quantile(0.75)
        IQR = Q3 - Q1

        upper_fence = Q3 + (2 * IQR)
        upper_limits[cat] = upper_fence
        print(f"  [{cat}]: Q3={Q3:.2f} | IQR={IQR:.2f} | Cut limit (> {upper_fence:.2f}s)")
    else:
        upper_limits[cat] = np.inf

del full_df, df_list

total_removed = 0
total_trips = 0

for file in all_files:
    filename = os.path.basename(file)

    df = pd.read_parquet(file)
    original_len = len(df)
    total_trips += original_len

    df['dist_group_error'] = pd.cut(
        df['distance_m'], 
        bins=dist_bins, 
        labels=dist_labels, 
        right=True
    )

    mask_ghost = (df['dist_group_error'] == 'Very Short (< 500m)') & (df['estimation_error_sec'] > VERY_SHORT_LIMIT)
    mask_outlier = pd.Series(False, index=df.index)

    for cat in categories_to_check:
        limit = upper_limits.get(cat, np.inf)
        mask_cat_outlier = (df['dist_group_error'] == cat) & (df['estimation_error_sec'] > limit)
        mask_outlier = mask_outlier | mask_cat_outlier

    rows_to_remove = mask_ghost | mask_outlier
    
    df_clean = df[~rows_to_remove].copy()
    
    df_clean.drop(columns=['dist_group_error'], inplace=True) 
    
    save_path = os.path.join(OUTPUT_PATH, filename)
    df_clean.to_parquet(save_path, index=False)
    
    removed_count = original_len - len(df_clean)
    total_removed += removed_count
    
    print(f"  Processing {filename}: Removed {removed_count} trips ({removed_count/original_len:.1%})")

print(f"PROCESS FINISHED.")
print(f"Total trips analyzed: {total_trips}")
print(f"Total trips removed: {total_removed}")
print(f"Cleaned files saved to: {OUTPUT_PATH}")