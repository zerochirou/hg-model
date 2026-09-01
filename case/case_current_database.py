import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import onnxruntime as rt
from utils.pull_data import pull_road_data
from utils.features import RoadFeatureExtractor


def run_database_inference(model_path="./models/hg_model_v1.onnx"):
    data = pull_road_data()
    
    # Fallback ke dataset lokal jika Supabase kosong atau error
    if not data:
        print("Data Supabase kosong / tidak dapat diakses. Menggunakan data sensor lokal sebagai sampel simulasi...")
        fallback_path = "./data/Pothole/trip1_sensors.csv"
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path).head(200)
        else:
            print("Tidak ada data untuk diproses.")
            return
    else:
        df = pd.DataFrame(data)
        
    print(f"Data yang akan diproses: {len(df)} baris")

    # Mapping nama kolom jika format dari database berbeda
    if "sensor_x" in df.columns:
        df["accelerometerX"] = df["sensor_x"]
        df["accelerometerY"] = df["sensor_y"]
        df["accelerometerZ"] = df["sensor_z"]
    if "speed_kmh" in df.columns:
        df["speed"] = df["speed_kmh"]

    # Ekstraksi Fitur Konsisten menggunakan RoadFeatureExtractor
    extractor = RoadFeatureExtractor(window_size=20, step_size=5)
    X_stream, metadata = extractor.extract_from_stream(df)

    if X_stream.empty:
        print("Data tidak cukup untuk membentuk 1 window (butuh minimal 20 sampel).")
        return

    # Inisialisasi ONNX Runtime
    if not os.path.exists(model_path):
        # Cek jika dipanggil dari subfolder case
        alt_path = os.path.join("..", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Model ONNX tidak ditemukan di {model_path}")
            return

    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    prob_name = session.get_outputs()[1].name

    inputs = X_stream.values.astype(np.float32)
    preds, probs = session.run([label_name, prob_name], {input_name: inputs})

    pothole_probs = probs[:, 1] if probs.ndim == 2 else probs
    metadata["predicted_label"] = preds
    metadata["pothole_probability"] = pothole_probs
    metadata["status"] = np.where(preds == 1, "⚠️ POTHOLE", "✅ NORMAL")

    print("\n=== HASIL INFERENSI REAL-TIME ONNX ===")
    print(metadata["status"].value_counts())
    print("\nSampel Hasil Prediksi:")
    print(metadata[["timestamp", "latitude", "longitude", "peak_shock", "pothole_probability", "status"]].head(10))


if __name__ == "__main__":
    run_database_inference()