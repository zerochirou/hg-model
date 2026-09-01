import numpy as np
import pandas as pd


class RoadFeatureExtractor:
    """
    Ekstraktor Fitur Sensor Jalan (HyperGrid) yang invarian terhadap orientasi
    dan konsisten digunakan untuk Training maupun Real-time Inference.
    """
    def __init__(self, window_size=20, step_size=5, g_constant=1.0):
        """
        Parameters:
        - window_size: Jumlah sampel per sliding window (misal: 20 sampel ~ 2 detik pada 10Hz)
        - step_size: Jumlah pergeseran per window (stride)
        - g_constant: Nilai gravitasi (1.0 g jika sensor ternormalisasi ke 1g, atau 9.81 m/s^2)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.g_constant = g_constant
        
    def _preprocess_trip_signals(self, df_trip):
        """
        Menghitung sinyal turunan secara terisolasi untuk satu trip.
        """
        df = df_trip.copy()
        
        # 1. Pastikan kolom sensor standar ada
        ax = df["accelerometerX"] if "accelerometerX" in df.columns else df.get("sensor_x", 0.0)
        ay = df["accelerometerY"] if "accelerometerY" in df.columns else df.get("sensor_y", 0.0)
        az = df["accelerometerZ"] if "accelerometerZ" in df.columns else df.get("sensor_z", 0.0)
        
        gx = df["gyroX"] if "gyroX" in df.columns else df.get("sensor_gx", 0.0)
        gy = df["gyroY"] if "gyroY" in df.columns else df.get("sensor_gy", 0.0)
        gz = df["gyroZ"] if "gyroZ" in df.columns else df.get("sensor_gz", 0.0)
        
        speed = df["speed"] if "speed" in df.columns else df.get("speed_kmh", 0.0)
        
        # 2. Magnitudo Vektor 3D (Orientation-Invariant)
        df["acc_mag"] = np.sqrt(ax**2 + ay**2 + az**2)
        df["gyro_mag"] = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Baseline gravitasi dinamis menggunakan rolling median/mean lokal (menghilangkan kemiringan HP)
        baseline = df["acc_mag"].rolling(window=max(self.window_size * 2, 30), min_periods=1, center=True).median()
        df["dynamic_acc"] = (df["acc_mag"] - baseline).abs()
        
        # 3. Jerk (Turunan pertama dari getaran)
        df["jerk"] = df["dynamic_acc"].diff().abs().fillna(0.0)
        df["angular_jerk"] = df["gyro_mag"].diff().abs().fillna(0.0)
        df["speed_diff"] = speed.diff().abs().fillna(0.0)
        df["speed_clean"] = speed
        
        return df

    def _extract_window_features(self, window_df):
        """
        Mengekstrak fitur statistik dan bentuk sinyal dari 1 window data.
        """
        features = {}
        
        target_signals = ["dynamic_acc", "gyro_mag", "jerk", "angular_jerk", "speed_clean"]
        
        for sig in target_signals:
            vals = window_df[sig].values
            
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            rng_val = max_val - min_val
            rms_val = np.sqrt(np.mean(vals**2))
            
            features[f"{sig}_mean"] = float(mean_val)
            features[f"{sig}_std"] = float(std_val)
            features[f"{sig}_min"] = float(min_val)
            features[f"{sig}_max"] = float(max_val)
            features[f"{sig}_range"] = float(rng_val)
            features[f"{sig}_rms"] = float(rms_val)
            
        # Fitur Sinyal Tambahan untuk dynamic_acc (Karakteristik Getaran Lubang)
        dyn_vals = window_df["dynamic_acc"].values
        rms_dyn = features["dynamic_acc_rms"]
        max_dyn = features["dynamic_acc_max"]
        
        # Crest Factor = Peak / RMS (mengukur ketajaman lonjakan impulsif)
        features["dynamic_acc_crest_factor"] = float(max_dyn / (rms_dyn + 1e-6))
        
        # Zero-Crossing Rate sekitar median lokal
        centered_dyn = dyn_vals - np.median(dyn_vals)
        zcr = ((centered_dyn[:-1] * centered_dyn[1:]) < 0).sum() / (len(dyn_vals) - 1 + 1e-6)
        features["dynamic_acc_zcr"] = float(zcr)
        
        return features

    def extract_from_trips(self, trips_list):
        """
        Mengekstrak fitur dari daftar trip (untuk Training).
        
        Returns:
        - X (pd.DataFrame): Matriks Fitur
        - y (pd.Series): Target Label (0: Normal, 1: Pothole)
        - groups (pd.Series): Trip ID untuk GroupKFold / GroupShuffleSplit
        - metadata (pd.DataFrame): Info GPS, Timestamp, dan Shock Peak
        """
        records = []
        labels = []
        groups = []
        meta_list = []
        
        for trip_df in trips_list:
            if len(trip_df) < self.window_size:
                continue
                
            trip_id = trip_df["trip_id"].iloc[0] if "trip_id" in trip_df.columns else "unknown"
            df_processed = self._preprocess_trip_signals(trip_df)
            
            n_samples = len(df_processed)
            for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                win = df_processed.iloc[start_idx:end_idx]
                
                feat = self._extract_window_features(win)
                records.append(feat)
                
                # Window dilabeli 1 jika ada minimal 1 kejadian pothole di dalam window
                win_label = 1 if (win["label"] == 1).any() else 0
                labels.append(win_label)
                groups.append(trip_id)
                
                # Metadata pendukung
                meta_list.append({
                    "trip_id": trip_id,
                    "timestamp": win["timestamp"].iloc[-1] if "timestamp" in win.columns else 0,
                    "latitude": win["latitude"].mean() if "latitude" in win.columns else 0.0,
                    "longitude": win["longitude"].mean() if "longitude" in win.columns else 0.0,
                    "peak_shock": feat["dynamic_acc_max"],
                    "speed": feat["speed_clean_mean"],
                })
                
        X = pd.DataFrame(records)
        y = pd.Series(labels, name="label")
        groups = pd.Series(groups, name="trip_id")
        metadata = pd.DataFrame(meta_list)
        
        return X, y, groups, metadata

    def extract_from_stream(self, df_stream):
        """
        Mengekstrak fitur dari data sensor stream / database (untuk Real-time Inference).
        
        Returns:
        - X (pd.DataFrame): Matriks Fitur
        - metadata (pd.DataFrame): Metadata GPS dan Timestamp
        """
        if len(df_stream) < self.window_size:
            return pd.DataFrame(), pd.DataFrame()
            
        df_processed = self._preprocess_trip_signals(df_stream)
        records = []
        meta_list = []
        
        n_samples = len(df_processed)
        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            win = df_processed.iloc[start_idx:end_idx]
            
            feat = self._extract_window_features(win)
            records.append(feat)
            
            meta_list.append({
                "timestamp": win["timestamp"].iloc[-1] if "timestamp" in win.columns else 0,
                "latitude": win["latitude"].mean() if "latitude" in win.columns else 0.0,
                "longitude": win["longitude"].mean() if "longitude" in win.columns else 0.0,
                "peak_shock": feat["dynamic_acc_max"],
                "speed": feat["speed_clean_mean"],
            })
            
        X = pd.DataFrame(records)
        metadata = pd.DataFrame(meta_list)
        return X, metadata
