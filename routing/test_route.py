import pandas as pd
import requests
import os

GH_URL = "http://localhost:8989/route"
OUTPUT_FILE = "data/processed/matrix_routes/test_route.parquet"
PROFILES = ["lts1_kids", "lts2_safe", "lts3_confident", "lts4_direct"]

def get_geometry_wkt(path):
    """Convierte las coordenadas del JSON a formato WKT (LINESTRING)."""
    try:
        points = path['points']['coordinates']
        coords_str = ",".join([f"{round(p[0],5)} {round(p[1],5)}" for p in points])
        return f"LINESTRING({coords_str})"
    except Exception as e:
        print(f"Error al obtener geometría: {e}")
        return None

def main():
    # 1. Hardcodeamos tu viaje de ejemplo
    row = {
        'start_station_id': '6422.08',
        'end_station_id': '5679.08',
        'lat_origin': 40.750224,
        'lon_origin': -73.971214,
        'lat_destination': 40.726156,
        'lon_destination': -73.995102
    }

    coord_start = f"{row['lat_origin']},{row['lon_origin']}"
    coord_end = f"{row['lat_destination']},{row['lon_destination']}"
    
    results = []
    session = requests.Session()

    print(f"Calculando rutas para {row['start_station_id']} -> {row['end_station_id']}...")

    # 2. Iteramos por cada perfil
    for profile in PROFILES:
        print(f" -> Procesando perfil: {profile}")
        
        data = {
            "start_station_id": row['start_station_id'],
            "end_station_id": row['end_station_id'],
            "profile": profile,
            "time_min": None,
            "distance_m": None,
            "geometry": None
        }

        params = {
            "point": [coord_start, coord_end],
            "profile": profile,
            "locale": "es",
            "calc_points": True,
            "points_encoded": False,
            "elevation": False
        }

        try:
            resp = session.get(GH_URL, params=params, timeout=10)

            if resp.status_code == 200:
                json_data = resp.json()
                path = json_data['paths'][0]

                data['time_min'] = round(path['time'] / 60000, 3)
                data['distance_m'] = round(path['distance'], 2)
                data['geometry'] = get_geometry_wkt(path)
                print(f"    [OK] Distancia: {data['distance_m']}m | Tiempo: {data['time_min']}min")
            else:
                print(f"    [ERROR] Código {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"    [EXCEPCIÓN] {e}")

        results.append(data)

    # 3. Guardar el resultado en un Parquet
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n¡Proceso completado! Archivo guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()