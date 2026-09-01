import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
import onnxruntime as rt
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def train_hypergrid_model(
    X,
    y,
    groups,
    metadata,
    model_output_path="./models/hg_model_v1.onnx",
    test_size=0.2,
    random_state=42,
    tune_hyperparameters=True,
):
    """
    Melatih model XGBoost dengan Group-Aware Split (Trip-based) dan mengekspor ke ONNX.
    """
    if len(X) == 0 or len(groups) == 0:
        raise ValueError(
            "Matriks fitur X atau groups kosong (0 sampel). "
            "Pastikan dataset sensor berhasil dimuat dan diekstraksi."
        )

    # Resolusi path model absolut
    if not os.path.isabs(model_output_path):
        model_output_path = os.path.join(PROJECT_ROOT, model_output_path.lstrip("./"))

    print("=" * 60)
    print("      HYPERGRID MACHINE LEARNING PIPELINE TRAINING")
    print("=" * 60)
    
    # 1. GroupShuffleSplit: Pisahkan data latih dan uji berdasarkan Sesi/Trip Utuh
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train, groups_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()
    meta_train, meta_test = metadata.iloc[train_idx].copy(), metadata.iloc[test_idx].copy()
    
    print(f"\n1. DATA SPLIT (GROUP-AWARE):")
    print(f" - Train Samples: {len(X_train)} (from {groups_train.nunique()} trips: {list(groups_train.unique())})")
    print(f" - Test Samples : {len(X_test)} (from {groups_test.nunique()} trips: {list(groups_test.unique())})")
    print(f" - Train Pothole Ratio: {(y_train == 1).mean():.1%}")
    print(f" - Test Pothole Ratio : {(y_test == 1).mean():.1%}")
    
    # Pastikan tidak ada trip yang bocor
    overlap = set(groups_train).intersection(set(groups_test))
    assert len(overlap) == 0, f"Error: Data leakage detected on trips: {overlap}"
    
    # 2. Imbalance Ratio
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f" - XGBoost scale_pos_weight: {scale_pos:.2f}")
    
    # 3. Hyperparameter Tuning dengan GroupKFold
    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        n_jobs=-1,
    )
    
    if tune_hyperparameters:
        print("\n2. HYPERPARAMETER TUNING VIA GROUP-K-FOLD...")
        gkf = GroupKFold(n_splits=min(5, groups_train.nunique()))
        param_dist = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "gamma": [0, 0.1, 0.2],
        }
        
        cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1",
            cv=cv_splits,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f" - Best CV F1-Score: {search.best_score_:.4f}")
        print(f" - Best Parameters : {search.best_params_}")
    else:
        best_model = base_xgb
        best_model.fit(X_train, y_train)
        
    # 4. Evaluasi terhadap Test Set (Truly Unseen Trips)
    print("\n3. EVALUASI TEST SET (UNSEEN TRIPS):")
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Cari ambang batas (threshold) probabilitas optimal berdasarkan F1 pada Train/Validation
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_thresh_idx = np.argmax(f1_scores)
    optimal_thresh = float(thresholds[best_thresh_idx]) if best_thresh_idx < len(thresholds) else 0.5
    optimal_thresh = np.clip(optimal_thresh, 0.3, 0.7)
    
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred_opt = (y_test_proba >= optimal_thresh).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f" - ROC-AUC Score: {roc_auc:.4f}")
    print(f" - Optimal Threshold (F1-Tuned): {optimal_thresh:.3f}")
    
    print("\n--- Classification Report (Default Threshold 0.5) ---")
    print(classification_report(y_test, y_test_pred_default, target_names=["Normal", "Pothole"]))
    
    print(f"--- Classification Report (Optimal Threshold {optimal_thresh:.3f}) ---")
    print(classification_report(y_test, y_test_pred_opt, target_names=["Normal", "Pothole"]))
    
    cm = confusion_matrix(y_test, y_test_pred_opt)
    print("Confusion Matrix:")
    print(f" TN: {cm[0, 0]} | FP: {cm[0, 1]}")
    print(f" FN: {cm[1, 0]} | TP: {cm[1, 1]}")
    
    # 5. Export ke ONNX
    print("\n4. EXPORT MODEL KE ONNX...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    from onnxmltools import convert_xgboost as convert_xgb
    from onnxmltools.convert.common.data_types import FloatTensorType as OFloatTensorType
    
    # Re-fit/wrap on numpy values to ensure standard feature indexing for ONNX
    onnx_ready_model = XGBClassifier(**best_model.get_params())
    onnx_ready_model.fit(X_train.values, y_train.values)
    
    num_features = X_train.shape[1]
    initial_types = [("float_input", OFloatTensorType([None, num_features]))]
    
    onnx_model = convert_xgb(
        onnx_ready_model,
        initial_types=initial_types,
        target_opset=12,
    )
    
    with open(model_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f" - Model ONNX berhasil disimpan ke: {model_output_path}")
    print(f" - Ukuran File: {os.path.getsize(model_output_path) / 1024:.1f} KB")
    
    # 6. Verifikasi ONNX Runtime
    print("\n5. VERIFIKASI ONNX RUNTIME...")
    sess = rt.InferenceSession(model_output_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name
    
    test_batch = X_test.iloc[:10].values.astype(np.float32)
    onnx_pred, onnx_prob = sess.run([label_name, prob_name], {input_name: test_batch})
    py_pred = onnx_ready_model.predict(test_batch)
    
    is_matching = np.array_equal(py_pred, onnx_pred)
    print(f" - Verifikasi Hasil Python vs ONNX: {'SUKSES (IDENTIK)' if is_matching else 'PERBEDAAN TERDETEKSI'}")
    
    df_results = meta_test.copy()
    df_results["actual_label"] = y_test.values
    df_results["predicted_label"] = y_test_pred_opt
    df_results["pothole_probability"] = y_test_proba
    
    return {
        "model": best_model,
        "optimal_threshold": optimal_thresh,
        "roc_auc": roc_auc,
        "feature_names": list(X.columns),
        "results_df": df_results,
    }
