import pandas as pd
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost,
)
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
import folium
from folium.plugins import MarkerCluster
import onnxruntime as rt

pothole_datasets = glob.glob("./data/Pothole/*.csv")
road_datasets = glob.glob("./data/RoadCondition/*.csv")
dfs = []
for file in pothole_datasets:
    df_temp = pd.read_csv(file)
    df_temp["label"] = 1   # 1 → pothole
    dfs.append(df_temp)

for file in road_datasets:
    df_temp = pd.read_csv(file)
    df_temp["label"] = 0   # 0 → smooth road
    dfs.append(df_temp)

df_raw = pd.concat(dfs, ignore_index=True)
df = pd.DataFrame(df_raw)
print("Combined dataset shape:", df.shape)

df_clean = df.sort_values(by="timestamp").reset_index(drop=True)

df_clean = df.dropna(subset=["latitude", "longitude", "speed", "accelerometerX"])

print("Ukuran setelah pembersihan missing values:", df_clean.shape[0])
print("Jumlah missing values:", df.shape[0] - df_clean.shape[0])

df_clean["acc_magnitude"] = np.sqrt(
    df_clean["accelerometerX"] ** 2
    + df_clean["accelerometerY"] ** 2
    + df_clean["accelerometerZ"] ** 2
)

df_clean["gyro_magnitude"] = np.sqrt(
    df_clean["gyroX"] ** 2
    + df_clean["gyroY"] ** 2
    + df_clean["gyroZ"] ** 2
)

df_clean["jerk"] = df_clean["acc_magnitude"].diff().abs()
df_clean["angular_jerk"] = df_clean["gyro_magnitude"].diff().abs()
df_clean["speed_diff"] = df_clean["speed"].diff()

window_size = 10
target_cols = [
    "acc_magnitude",
    "gyro_magnitude",
    "jerk",
    "speed",
    "accelerometerZ",
]

for col in target_cols:
    df_clean[f"{col}_std"] = (
        df_clean[col]
        .rolling(window=window_size, min_periods=window_size)
        .std()
    )
    df_clean[f"{col}_max"] = (
        df_clean[col]
        .rolling(window=window_size, min_periods=window_size)
        .max()
    )
    df_clean[f"{col}_min"] = (
        df_clean[col]
        .rolling(window=window_size, min_periods=window_size)
        .min()
    )
    df_clean[f"{col}_range"] = df_clean[f"{col}_max"] - df_clean[f"{col}_min"]

df_clean = df_clean.dropna().reset_index(drop=True)
metadata_cols = ["timestamp", "latitude", "longitude"]
metadata = df_clean[metadata_cols].copy()
y = df_clean["label"].copy()
X = df_clean.drop(columns=metadata_cols + ["label"])

print("=== PIPELINE DATA CLEANING SELESAI ===")
print(f"X (Matriks Fitur Latih) : {X.shape}")
print(f"y (Target Label)        : {y.shape}")
print(f"Metadata GPS           : {metadata.shape}\n")

print("Daftar Fitur yang dilatih ke Model Machine Learning:")
print(X.columns.tolist())

test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X,
    y,
    metadata,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
)

print(f"Jumlah Data Train : {X_train.shape[0]} baris")
print(f"Jumlah Data Test  : {X_test.shape[0]} baris")
ratio_imbalance = (y_train == 0).sum() / (y_train == 1).sum()
print(
    f"Rasio Imbalance Class (scale_pos_weight untuk XGBoost): {ratio_imbalance:.2f}"
)

pipeline = Pipeline(
    [
        ("scaler", RobustScaler()),  # Scaler dimasukkan ke dalam Pipeline
        (
            "xgb",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=ratio_imbalance,  # Penanganan Imbalanced Data
            ),
        ),
    ]
)

param_distributions = {
    "xgb__n_estimators": [100, 200, 300],
    "xgb__max_depth": [3, 5, 7, 9],
    "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    "xgb__gamma": [0, 0.1, 0.3],
}

cv_strategy = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,  # Jumlah kombinasi sampel acak
    scoring="f1",  # Optimalkan F1-Score (penting untuk kelas Pothole)
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

print("Memulai Hyperparameter Tuning...")
search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\n=== TUNING SELESAI ===")
print("F1-Score Terbaik pada Cross-Validation:", search.best_score_)
print("Hyperparameter Terbaik:")
for param, val in search.best_params_.items():
    print(f" - {param}: {val}")

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("=== METRIK EVALUASI TERHADAP TEST SET (REAL UNSEEN DATA) ===")
print(classification_report(y_test, y_pred, target_names=["Normal", "Pothole"]))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Negative  (Normal Terdeteksi Normal)  : {cm[0, 0]}")
print(f"False Positive (Normal Terdeteksi Pothole) : {cm[0, 1]}")
print(f"False Negative (Pothole Terlewat)          : {cm[1, 0]}")
print(f"True Positive  (Pothole Terdeteksi Akurat) : {cm[1, 1]}")

# Ambil model XGBoost dan nama fitur
xgb_step = best_model.named_steps["xgb"]
feature_names = X_train.columns

# Buat DataFrame Feature Importance
importances = pd.Series(xgb_step.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=True)

# Plot 15 Fitur Teratas
plt.figure(figsize=(10, 8))
importances.tail(15).plot(kind="barh", color="teal")
plt.title(
    "15 Fitur Paling Berpengaruh dalam Deteksi Pothole (XGBoost)", fontsize=13
)
plt.xlabel("Relative Importance Score")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

best_params = search.best_params_
clean_params = {
    k.replace("xgb__", ""): v
    for k, v in best_params.items()
    if k.startswith("xgb__")
}

final_pipeline = Pipeline(
    [
        ("scaler", RobustScaler()),
        (
            "xgb",
            XGBClassifier(
                **clean_params,
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=ratio_imbalance,  # Penanganan Imbalanced Data
            ),
        ),
    ]
)

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

# 2. Cetak Classification Report
print("=== CLASSIFICATION REPORT (DATA TEST) ===")
print(classification_report(y_test, y_pred, target_names=["Normal", "Pothole"]))

# 3. Hitung ROC-AUC Score
roc_score = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_score:.4f}")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    xticklabels=["Normal", "Pothole"],
    yticklabels=["Normal", "Pothole"],
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Prediksi Model")
axes[0].set_ylabel("Aktual (Ground Truth)")

# 2. Plot ROC Curve
RocCurveDisplay.from_estimator(final_pipeline, X_test, y_test, ax=axes[1])
axes[1].set_title("ROC Curve")
axes[1].grid(True, linestyle="--", alpha=0.6)

# 3. Plot Precision-Recall Curve (Krusial untuk Imbalanced Class)
PrecisionRecallDisplay.from_estimator(
    final_pipeline, X_test, y_test, ax=axes[2]
)
axes[2].set_title("Precision-Recall Curve")
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# Ambil estimator XGBoost dari Pipeline
xgb_model = final_pipeline.named_steps["xgb"]

# Ambil Feature Importance (berdasarkan Gain)
importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": xgb_model.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Visualisasi Top 15 Fitur Teratas
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df.head(15), x="Importance", y="Feature", palette="viridis"
)
plt.title("Top 15 Fitur Paling Berpengaruh dalam Deteksi Pothole", fontsize=13)
plt.xlabel("Importance Score (Gain)")
plt.ylabel("Nama Fitur")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Display 10 Fitur Teratas dalam bentuk tabel
importance_df.head(10).reset_index(drop=True)

# Buat DataFrame Hasil Prediksi + Metadata GPS
df_test_results = meta_test.copy()
df_test_results["actual_label"] = y_test.values
df_test_results["predicted_label"] = y_pred
df_test_results["pothole_probability"] = y_proba

# Filter khusus untuk data yang diprediksi sebagai Pothole (Label = 1)
potholes_detected = df_test_results[df_test_results["predicted_label"] == 1]

print(f"Total Sampel Test          : {len(df_test_results)}")
print(f"Total Pothole Terdeteksi   : {len(potholes_detected)} titik lokasi")

# Tampilkan 5 sampel data terdeteksi pothole beserta koordinat GPS-nya
potholes_detected

# 1. Filter hasil prediksi data test yang terdeteksi sebagai Pothole (True Positive & False Positive)
pothole_locations = df_test_results[df_test_results["predicted_label"] == 1]

# 2. Tentukan titik tengah peta berdasarkan rerata GPS
center_lat = pothole_locations["latitude"].mean()
center_lon = pothole_locations["longitude"].mean()

# 3. Inisialisasi Peta Folium
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
marker_cluster = MarkerCluster().add_to(m)

# 4. Tambahkan Marker untuk setiap titik Pothole terdeteksi
for idx, row in pothole_locations.iterrows():
    # Menandai warna: Hijau untuk True Positive (Akurat), Merah untuk False Positive (Alarm Palsu)
    is_correct = row["actual_label"] == row["predicted_label"]
    color = "green" if is_correct else "red"
    status = (
        "Pothole Akurat" if is_correct else "Alarm Palsu (Potensi Jalan Rusak)"
    )

    popup_text = f"""
    <b>Status:</b> {status}<br>
    <b>Probabilitas:</b> {row['pothole_probability']:.2%}<br>
    <b>Lat:</b> {row['latitude']:.5f}<br>
    <b>Lon:</b> {row['longitude']:.5f}
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup_text,
    ).add_to(marker_cluster)


print(
    "Peta interaktif berhasil disimpan sebagai 'pothole_detection_map.html'!"
)
m

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# PREDICTION
# =========================
y_train_pred = final_pipeline.predict(X_train)
y_test_pred = final_pipeline.predict(X_test)


# =========================
# FUNCTION EVALUASI
# =========================
def evaluate_model(y_true, y_pred, name):
    return {
        "Dataset": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }


# =========================
# HASIL
# =========================
results = pd.DataFrame([
    evaluate_model(y_train, y_train_pred, "Train"),
    evaluate_model(y_test, y_test_pred, "Test"),
])

print(results)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for metric in metrics:
    train_score = results.loc[
        results["Dataset"] == "Train", metric
    ].iloc[0]

    test_score = results.loc[
        results["Dataset"] == "Test", metric
    ].iloc[0]

    gap = train_score - test_score

    print(f"{metric}:")
    print(f"  Train = {train_score:.4f}")
    print(f"  Test  = {test_score:.4f}")
    print(f"  Gap   = {gap:.4f}\n")

cv_results = pd.DataFrame(search.cv_results_)

best_idx = search.best_index_

for i in range(5):
    score = cv_results.loc[
        best_idx,
        f"split{i}_test_score"
    ]

    print(f"Fold {i + 1}: {score:.4f}")

scores = []

for i in range(5):
    scores.append(
        cv_results.loc[
            best_idx,
            f"split{i}_test_score"
        ]
    )

print("CV Scores:", scores)
print(f"Mean : {np.mean(scores):.4f}")
print(f"Std  : {np.std(scores):.4f}")

# 1. Register XGBoost ke skl2onnx
update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False]},
)

# 2. Spesifikasi tipe data input (32 fitur float32)
num_features = X_train.shape[1]
initial_type = [("float_input", FloatTensorType([None, num_features]))]

# 3. Konversi Pipeline dengan target_opset dictionary (FIX ERROR)
onnx_model = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset={
        "": 12,
        "ai.onnx.ml": 3,
    },  # <--- PERBAIKAN: Kunci ai.onnx.ml ke versi 3
    options={"zipmap": False},
)

# 4. Simpan model ONNX ke file
onnx_filename = "./models/hg_model_v1.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model Pipeline berhasil dikonversi ke '{onnx_filename}' (Opset ML v3)!")

# 1. Buat Inference Session ONNX
session = rt.InferenceSession("./models/hg_model_v1.onnx")
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
prob_name = session.get_outputs()[1].name

# 2. Uji dengan 5 data dari X_test (harus bertipe float32)
dummy_input = X_test.iloc[:5].values.astype(np.float32)

# 3. Jalankan Prediksi ONNX
onnx_pred, onnx_probs = session.run([label_name, prob_name], {input_name: dummy_input})

# 4. Bandingkan dengan model Python original
py_pred = final_pipeline.predict(X_test.iloc[:5])

print("=== VERIFIKASI AKURASI KONVERSI ===")
print("Prediksi Python Original :", py_pred)
print("Prediksi ONNX Runtime   :", onnx_pred)

# Uji apakah output identik
if np.array_equal(py_pred, onnx_pred):
    print("\n HASIL IDENTIK! Model ONNX 100% Valid dan Siap Deploy.")
else:
    print("\n Terdapat perbedaan hasil antara ONNX dan Python!")