import os
import glob
import pandas as pd
import numpy as np

# Path absolut ke root direktori proyek
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_raw_trips(data_dir="./data", pothole_tolerance_sec=0.75):
    """
    Memuat data mentah dari folder Pothole dan RoadCondition secara terisolasi per file/trip.
    Mendukung pemanggilan dari direktori mana pun (CWD independen).
    
    Parameters:
    - data_dir: Path ke direktori data (default: './data')
    - pothole_tolerance_sec: Rentang waktu toleransi (detik) di sekitar timestamp kejadian lubang
    
    Returns:
    - List of DataFrames, di mana setiap DataFrame mewakili 1 trip perjalanan lengkap dengan
      label baris per sampel (0: Normal, 1: Pothole) dan kolom 'trip_id'.
    """
    # Resolusi path absolut jika relative path tidak ditemukan di CWD saat ini
    if not os.path.isabs(data_dir):
        if not os.path.exists(data_dir):
            alt_path = os.path.join(PROJECT_ROOT, data_dir.lstrip("./"))
            if os.path.exists(alt_path):
                data_dir = alt_path
            else:
                data_dir = os.path.join(PROJECT_ROOT, "data")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Direktori dataset tidak ditemukan di '{data_dir}'. "
            f"Pastikan folder 'data/Pothole' dan 'data/RoadCondition' tersedia di {PROJECT_ROOT}/data."
        )

    trips = []
    
    # 1. Muat trip dari folder Pothole
    pothole_dir = os.path.join(data_dir, "Pothole")
    sensor_files = sorted(glob.glob(os.path.join(pothole_dir, "*_sensors.csv")))
    
    for sf in sensor_files:
        trip_name = os.path.basename(sf).replace("_sensors.csv", "")
        pf = os.path.join(pothole_dir, f"{trip_name}_potholes.csv")
        
        df_sensor = pd.read_csv(sf)
        df_sensor = df_sensor.sort_values(by="timestamp").reset_index(drop=True)
        df_sensor["trip_id"] = f"pothole_{trip_name}"
        df_sensor["label"] = 0  # Default normal
        
        if os.path.exists(pf):
            df_potholes = pd.read_csv(pf)
            if "timestamp" in df_potholes.columns and not df_potholes.empty:
                pothole_ts = df_potholes["timestamp"].values
                
                # Tandai baris sensor yang berada dalam rentang toleransi kejadian lubang
                for ts in pothole_ts:
                    mask = (df_sensor["timestamp"] >= (ts - pothole_tolerance_sec)) & \
                           (df_sensor["timestamp"] <= (ts + pothole_tolerance_sec))
                    df_sensor.loc[mask, "label"] = 1
        
        trips.append(df_sensor)
        
    # 2. Muat trip dari folder RoadCondition (Good Roads -> Label 0)
    road_dir = os.path.join(data_dir, "RoadCondition")
    good_files = sorted(glob.glob(os.path.join(road_dir, "good*_sensors.csv")))
    for gf in good_files:
        trip_name = os.path.basename(gf).replace("_sensors.csv", "")
        df_good = pd.read_csv(gf)
        df_good = df_good.sort_values(by="timestamp").reset_index(drop=True)
        df_good["trip_id"] = f"road_{trip_name}"
        df_good["label"] = 0
        trips.append(df_good)
        
    # 3. Muat trip dari folder RoadCondition (Bad Roads -> Label 0 atau jalan bergelombang tapi bukan pothole spesifik)
    bad_files = sorted(glob.glob(os.path.join(road_dir, "bad*_sensors.csv")))
    for bf in bad_files:
        trip_name = os.path.basename(bf).replace("_sensors.csv", "")
        df_bad = pd.read_csv(bf)
        df_bad = df_bad.sort_values(by="timestamp").reset_index(drop=True)
        df_bad["trip_id"] = f"road_{trip_name}"
        df_bad["label"] = 0  # Non-pothole condition
        trips.append(df_bad)
        
    if not trips:
        raise ValueError(
            f"Tidak ada file CSV sensor (*_sensors.csv) yang ditemukan di {data_dir}."
        )

    print(f"Total trips loaded: {len(trips)} (from {data_dir})")
    for t in trips:
        pothole_count = (t['label'] == 1).sum()
        total_count = len(t)
        print(f" - [{t['trip_id'].iloc[0]}] Samples: {total_count}, Pothole samples: {pothole_count} ({pothole_count/total_count:.1%})")
        
    return trips
