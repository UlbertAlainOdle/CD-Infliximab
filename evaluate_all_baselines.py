# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "train_split.csv")
TEST_FILE = os.path.join(BASE_DIR, "test_split.csv")
EXTERNAL_FILE = os.path.join(BASE_DIR, "External Verification00.csv")

RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"

def load_data(csv_path):
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    # In case data doesn't have some columns, we only use what's available
    # But ideally it has all RAW_FEATURES and TARGET_COL
    X = df[RAW_FEATURES]
    y = df[TARGET_COL]
    return X, y

def get_metrics(y_true, y_prob):
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AP": average_precision_score(y_true, y_prob)
    }

def _best_threshold_for_f1(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0: return 0.5
    idx = np.argmax(f1s[:-1]) 
    return thresholds[idx]

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def ppv_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def npv_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def _gpu_params():
    """Returns GPU parameters for XGBoost."""
    import xgboost as xgb
    ver = xgb.__version__
    try:
        major = int(ver.split('.')[0])
    except Exception:
        major = 2
    if major >= 2:
        return dict(device="cuda", tree_method="hist")
    else:
        return dict(tree_method="gpu_hist")

def main():
    print("Loading data...")
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    X_ext, y_ext = load_data(EXTERNAL_FILE)
    
    if X_train is None:
        print(f"[ERROR] Could not load {TRAIN_FILE}")
        return

    # Define the requested baseline models
    models = {
        "Standard XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', **_gpu_params()),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=1),
        "CatBoost": CatBoostClassifier(random_state=42,verbose=0,iterations=150,learning_rate=0.05,depth=4, task_type="GPU"),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "MLP (Neural Network)": MLPClassifier(random_state=42, max_iter=1000),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=1, verbose=-1, device_type='gpu')
    }

    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Build a robust pipeline for fairness
        # Missing values are common, use Median Imputer
        steps = [('imputer', SimpleImputer(strategy='median'))]
        
        # Distance-based models MUST have scaled data. Tree models are unaffected.
        # To be completely fair and standard, we apply RobustScaler to the ones that need it.
        if name in ["SVM (RBF)", "MLP (Neural Network)", "Logistic Regression"]:
            steps.append(('scaler', RobustScaler()))
            
        steps.append(('classifier', model))
        pipeline = Pipeline(steps)

        # 1. Internal CV (10-fold 10-repeats)
        print("  -> Running Internal CV (10x10 RepeatedStratifiedKFold)...")
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        cv_res = cross_validate(pipeline, X_train, y_train, cv=rskf, scoring=scoring, n_jobs=1)
        
        model_results = {
            "Model": name,
            "CV AUC": f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}",
            "CV AP": f"{cv_res['test_average_precision'].mean():.4f} ± {cv_res['test_average_precision'].std():.4f}"
        }

        # 3. Fit on Full Train
        pipeline.fit(X_train, y_train)

        # 5. External Test
        if X_ext is not None:
            print("  -> Evaluating on External Test...")
            y_ext_prob = pipeline.predict_proba(X_ext)[:, 1]
            ext_m = get_metrics(y_ext, y_ext_prob)
            model_results["External AUC"] = f"{ext_m['AUC']:.4f}"
            model_results["External AP"] = f"{ext_m['AP']:.4f}"
        else:
            model_results["External AUC"] = "N/A"
            model_results["External AP"] = "N/A"
            
        results.append(model_results)

    # Save to CSV
    res_df = pd.DataFrame(results)
    out_file = os.path.join(BASE_DIR, "baseline_models_comparison.csv")
    res_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nAll evaluations complete. Results saved to {out_file}")

if __name__ == '__main__':
    main()
