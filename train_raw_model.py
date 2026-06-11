# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, precision_recall_curve
import optuna
import os
import json
import joblib
import warnings
from sklearn.utils import resample

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
TRAIN_FILE = os.path.join(BASE_DIR, "processed_with_features_LEFT_JOIN_all_rows.csv")
EXTERNAL_FILE = os.path.join(BASE_DIR, "External Verification00.csv")
BASELINE_MODEL_FILE = os.path.join(BASE_DIR, "baseline_best_model.joblib")
BASELINE_PARAMS_FILE = os.path.join(BASE_DIR, "baseline_best_params.json")

def _gpu_params():
    ver = xgb.__version__
    try:
        major = int(ver.split('.')[0])
    except Exception:
        major = 2
    if major >= 2:
        return dict(device="cuda", tree_method="hist")
    else:
        return dict(tree_method="gpu_hist")

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    if TARGET_COL not in df.columns:
        print(f"Warning: {TARGET_COL} not found in {csv_path}")
        return None, None
    
    # Ensure all raw features exist
    missing = [f for f in RAW_FEATURES if f not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing} in {csv_path}")
        return None, None

    X = df[RAW_FEATURES]
    y = df[TARGET_COL]
    return X, y

def optimize_xgboost(X, y, n_trials=100):
    print(f"Optimizing XGBoost with {n_trials} trials...")
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc', 
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0), 
            'random_state': 42,
            'n_jobs': 1,
            'verbosity': 0,
            **_gpu_params()
        }
        
        clf = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        try:
            score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=1).mean()
        except Exception:
            return 0.0
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Best CV AUC: {study.best_value:.4f}")
    return study.best_params

def _best_threshold_for_f1(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0: return 0.5
    idx = np.argmax(f1s[:-1]) 
    return thresholds[idx]

def evaluate_model_bootstrap(model, X, y, threshold=0.5, n_iterations=1000):
    """Evaluate model using bootstrap resampling to provide mean and std (±)."""
    metrics_list = {
        "acc": [], "f1": [], "auc": [], "ap": [], "precision": [], "recall": []
    }
    
    # Pre-calculate probabilities to save time
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    y_array = y.values if hasattr(y, 'values') else np.array(y)
    
    np.random.seed(42)
    for i in range(n_iterations):
        indices = resample(np.arange(len(y_array)), replace=True, random_state=i)
        y_boot = y_array[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = y_pred[indices]
        
        if len(np.unique(y_boot)) < 2:
            continue
            
        metrics_list["acc"].append(accuracy_score(y_boot, y_pred_boot))
        metrics_list["f1"].append(f1_score(y_boot, y_pred_boot))
        metrics_list["auc"].append(roc_auc_score(y_boot, y_prob_boot))
        metrics_list["ap"].append(average_precision_score(y_boot, y_prob_boot))
        metrics_list["precision"].append(precision_score(y_boot, y_pred_boot, zero_division=0))
        metrics_list["recall"].append(recall_score(y_boot, y_pred_boot))
        
    results = {}
    for k, v in metrics_list.items():
        if len(v) > 0:
            results[k] = {"mean": np.mean(v), "std": np.std(v)}
        else:
            results[k] = {"mean": 0.0, "std": 0.0}
            
    return results

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "ap": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

def main():
    print("=== Training Model with Raw Features Only ===")
    
    # 1. Load Pre-split Data
    train_split_file = "train_split.csv"
    test_split_file = "test_split.csv"
    
    if not (os.path.exists(train_split_file) and os.path.exists(test_split_file)):
        print(f"[ERROR] Missing {train_split_file} or {test_split_file}. Please run data splitting script first.")
        return
        
    print(f"Loading training data from {train_split_file}...")
    X_train, y_train = load_data(train_split_file)
    if X_train is None: return
    
    print(f"Loading holdout test data from {test_split_file}...")
    X_test, y_test = load_data(test_split_file)
    if X_test is None: return

    # 2. Select Features
    features_to_use = RAW_FEATURES
    X_train = X_train[features_to_use]
    X_test = X_test[features_to_use]
    print(f"Features used ({len(features_to_use)}): {features_to_use}")
    
    clf = None
    best_thr = 0.5

    # 3. Check for existing model
    if os.path.exists(BASELINE_MODEL_FILE) and os.path.exists(BASELINE_PARAMS_FILE):
        print(f"\nLoading existing model from {BASELINE_MODEL_FILE}...")
        clf = joblib.load(BASELINE_MODEL_FILE)
        with open(BASELINE_PARAMS_FILE, "r") as f:
            saved_params = json.load(f)
        best_thr = saved_params.pop("best_threshold", 0.5)
        best_params = saved_params
        print(f"Model loaded. Best threshold: {best_thr:.4f}")
        
    else:
        print("\nNo existing model found. Starting training pipeline...")
        
        # 4. Optimize Hyperparameters
        best_params = optimize_xgboost(X_train, y_train, n_trials=100)
        # Ensure GPU params are included if needed
        best_params.update(_gpu_params())
        best_params['random_state'] = 42
        best_params['n_jobs'] = 1
        
        # Determine Optimal Threshold (CV Predict)
        print("\nDetermining optimal threshold from Training CV...")
        clf_cv_thr = XGBClassifier(**best_params)
        y_train_prob = cross_val_predict(clf_cv_thr, X_train, y_train, cv=5, method='predict_proba', n_jobs=1)[:, 1]
        best_thr = _best_threshold_for_f1(y_train, y_train_prob)
        print(f"Optimal Threshold: {best_thr:.4f}")
        
        # Train Final Model
        print("\nRetraining model on full training set...")
        clf = XGBClassifier(**best_params)
        clf.fit(X_train, y_train)
        
        # Save Model and Params
        print(f"Saving model to {BASELINE_MODEL_FILE}...")
        joblib.dump(clf, BASELINE_MODEL_FILE)
        
        # Save params with threshold
        save_params = best_params.copy()
        save_params['best_threshold'] = float(best_thr) # ensure JSON serializable
        with open(BASELINE_PARAMS_FILE, "w") as f:
            json.dump(save_params, f, indent=4)
        print(f"Saved parameters to {BASELINE_PARAMS_FILE}")

    # 5. Internal Validation (Strict 10x 5-Fold CV)
    print("\n--- Internal CV Validation (10x Repeated 10-Fold CV) ---")
    clf_cv = XGBClassifier(**best_params)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    
    scoring = ['accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall']
    cv_results = cross_validate(clf_cv, X_train, y_train, cv=rskf, scoring=scoring, n_jobs=1)
    
    for metric in scoring:
        key = f"test_{metric}"
        mean_score = cv_results[key].mean()
        std_score = cv_results[key].std()
        print(f"{metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")

    # 6. External Validation (Always run)
    print(f"\n--- External Validation ({EXTERNAL_FILE}) ---")
    X_ext_raw, y_ext = load_data(EXTERNAL_FILE)
    if X_ext_raw is not None:
        X_ext = X_ext_raw[features_to_use]
        
        metrics_ext = evaluate_model(clf, X_ext, y_ext, threshold=best_thr)
        for k, v in metrics_ext.items():
            print(f"{k.upper()}: {v:.4f}")
    else:
        print("External validation skipped (file not found or invalid).")

if __name__ == "__main__":
    main()
