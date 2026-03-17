import pandas as pd
import geopandas as gpd
from shapely import wkt
import os

ROUTES_FILE = 'data/processed/infra_analysis/top_routes_with_geometries.parquet'
INFRA_FILE = 'data/raw/infrastructure/New_York_City_Bike_Routes_20260123.geojson'
OUTPUT_FILE = 'data/processed/infra_analysis/infrastructure_gaps.parquet'

NY_METRIC_CRS = 32618 
GPS_CRS = 4326

def main():
    # Load top routes with geometries
    print("Loading top routes...")
    top_routes = pd.read_parquet(ROUTES_FILE)
    top_routes['geometry'] = top_routes['geometry'].apply(wkt.loads)
    gdf_routes = gpd.GeoDataFrame(top_routes, geometry='geometry', crs=f"EPSG:{GPS_CRS}")

    # Load infrastructure data
    print("Loading infrastructure data...")
    infra_gdf = gpd.read_file(INFRA_FILE)

    # Reproject routes to metric CRS for accurate distance calculations
    gdf_routes = gdf_routes.to_crs(epsg=NY_METRIC_CRS)
    infra_gdf = infra_gdf.to_crs(epsg=NY_METRIC_CRS)

    # Create buffered polygon around infrastructure (15m buffer)
    print("Creating buffered polygon around infrastructure...")
    infra_polygon = infra_gdf.buffer(15).union_all()

    # Calculate gap geometries by finding the difference between route geometries and the buffered infrastructure polygon
    gdf_routes['gap_geometry'] = gdf_routes.geometry.difference(infra_polygon)

    # Calculate gaps
    gdf_gaps = gdf_routes.set_geometry('gap_geometry').copy()
    gdf_gaps.drop(columns=['geometry'], inplace=True)
    gdf_gaps.rename_geometry('geometry', inplace=True)

    # Remove empty geometries (routes that are fully covered by infrastructure)
    print("Filtering gaps...")
    gdf_gaps = gdf_gaps[~gdf_gaps.is_empty].copy()

    # Remove small gaps (less than 20 meters)
    # These small gaps are likely to be negligible and may be due to minor misalignments or inaccuracies in the data, so we can filter them out to focus on more significant gaps.
    gdf_gaps = gdf_gaps[gdf_gaps.geometry.length > 20].copy()

    print(f"    {len(gdf_gaps)} gaps identified after filtering.")

    # Reproject gaps back to GPS CRS for output
    gdf_gaps_final = gdf_gaps.to_crs(epsg=GPS_CRS)

    # Convert geometries to WKT for easier storage in Parquet
    df_out = pd.DataFrame(gdf_gaps_final)
    df_out['geometry'] = df_out['geometry'].apply(lambda geom: geom.wkt)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out.to_parquet(OUTPUT_FILE, index=False)
    print(f"Gaps saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()