"""
HyperGrid Series v1 - Automated End-to-End Pipeline
===================================================
Script ini menjalankan seluruh siklus pemrosesan data, ekstraksi fitur invarian orientasi,
training model XGBoost berbasis Group-Aware cross-validation, evaluasi pada unseen trips,
dan ekspor ke format ONNX.
"""

import os
import sys

# Tambahkan root folder ke sys.path agar import utils selalu bekerja
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import load_raw_trips
from utils.features import RoadFeatureExtractor
from utils.trainer import train_hypergrid_model


def main():
    print("=" * 65)
    print("      HYPERGRID MACHINE LEARNING PIPELINE (AUTORUN V1)")
    print("=" * 65)

    # 1. Load Raw Trips dengan Penyelarasan Ground-Truth Timestamp
    print("\n[Step 1/4] Memuat data perjalanan dan menyelaraskan label lubang...")
    trips = load_raw_trips(data_dir="./data", pothole_tolerance_sec=0.75)

    # 2. Ekstraksi Fitur Invarian Orientasi (Sliding Window)
    print("\n[Step 2/4] Mengekstrak fitur sensor statistik dan sinyal...")
    extractor = RoadFeatureExtractor(window_size=20, step_size=5)
    X, y, groups, metadata = extractor.extract_from_trips(trips)

    print(f" -> Matriks Fitur X : {X.shape}")
    print(f" -> Label Target y  : {y.shape} (Pothole: {(y==1).sum()}, Normal: {(y==0).sum()})")
    print(f" -> Jumlah Sesi Trip: {groups.nunique()}")

    # 3. Training & Evaluasi Model
    print("\n[Step 3/4] Melatih model dengan GroupKFold & Evaluasi Unseen Trips...")
    model_path = "./models/hg_model_v1.onnx"
    results = train_hypergrid_model(
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        model_output_path=model_path,
        test_size=0.2,
        random_state=42,
        tune_hyperparameters=True,
    )

    # 4. Ringkasan
    print("\n[Step 4/4] Pipeline selesai dijalankan!")
    print(f" -> Model ONNX tersimpan di : {model_path}")
    print(f" -> ROC-AUC Test Score      : {results['roc_auc']:.4f}")
    print(f" -> Ambang Batas Optimal    : {results['optimal_threshold']:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()