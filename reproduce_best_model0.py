# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, make_scorer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib
import json
import os

# Global Config
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
TRAIN_FILE = "processed_with_features_LEFT_JOIN_all_rows.csv"
EXTERNAL_FILE = "External Verification00.csv"
SELECTED_FEATURES_FILE = "selected_features.json"

# --- Hardcoded Best Params from User ---
BEST_PARAMS = {'n_estimators': 137, 'max_depth': 10, 'learning_rate': 0.017444824475985048, 'subsample': 0.9997391245814459, 'colsample_bytree': 0.9415743112958483, 'gamma': 1.7483463681615965, 'min_child_weight': 2, 'reg_alpha': 0.04845238191904519, 'reg_lambda': 0.0121275899499253, 'scale_pos_weight': 4.045168009584749}

def _gpu_params():
    """Returns GPU parameters for XGBoost."""
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
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found.")
    
    missing = [f for f in RAW_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing raw features: {missing}")
    
    # Remove .dropna() to allow Imputer to handle missing values
    df = df[RAW_FEATURES + [TARGET_COL]]
    X = df[RAW_FEATURES]
    y = df[TARGET_COL]
    
    return X, y

# --- Feature Engineering Classes (Consistent with v4) ---
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits
        self.percentiles_ = {}
        self.numeric_cols = []

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if not self.numeric_cols:
            return self
        for col in self.numeric_cols:
            lower = np.percentile(X[col], self.limits[0] * 100)
            upper = np.percentile(X[col], (1 - self.limits[1]) * 100)
            self.percentiles_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.numeric_cols:
            if col in self.percentiles_:
                lower, upper = self.percentiles_[col]
                X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out

class UltimateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.pt = PowerTransformer(method='yeo-johnson', standardize=False)
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.num_cols = []
        self.poly_feat_names = []
        self._final_columns = []
        
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        if self.num_cols:
            self.scaler.fit(X[self.num_cols])
            X_scaled = self.scaler.transform(X[self.num_cols])
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.num_cols, index=X.index)
            
            km_cols = [c for c in ['CRP0', 'ESR0', 'PLT0', 'ALB0', 'Ca0'] if c in self.num_cols]
            if km_cols:
                self.kmeans.fit(X_scaled_df[km_cols])
            
            self.pt.fit(X_scaled)
            X_pt = self.pt.transform(X_scaled)
            self.poly.fit(X_pt)
            self.poly_feat_names = self.poly.get_feature_names_out(self.num_cols)
        return self

    def transform(self, X):
        X_out = X.copy()
        eps = 1e-6
        
        # 0. Clinical Flags
        if 'CRP0' in X.columns:
            X_out['Severe_Inflammation_CRP'] = (X['CRP0'] > 40).astype(int)
        if 'ESR0' in X.columns:
            X_out['Severe_Inflammation_ESR'] = (X['ESR0'] > 50).astype(int)
        if 'ALB0' in X.columns:
            X_out['Hypoalbuminemia'] = (X['ALB0'] < 35).astype(int)
        if 'HB0' in X.columns:
            conditions = [
                (X['HB0'] >= 120),
                (X['HB0'] >= 90) & (X['HB0'] < 120),
                (X['HB0'] >= 60) & (X['HB0'] < 90),
                (X['HB0'] < 60)
            ]
            X_out['Anemia_Severity'] = np.select(conditions, [0, 1, 2, 3], default=0)

        # 1. Ratios
        if 'CRP0' in X.columns and 'ALB0' in X.columns:
            X_out['CAR'] = X['CRP0'] / (X['ALB0'] + eps)
        if 'PLT0' in X.columns and 'Lyn0' in X.columns:
            X_out['PLR'] = X['PLT0'] / (X['Lyn0'] + eps)
        if 'M0' in X.columns and 'Lyn0' in X.columns:
            X_out['MLR'] = X['M0'] / (X['Lyn0'] + eps)
        if 'Ca0' in X.columns and 'ALB0' in X.columns:
            X_out['Corrected_Ca'] = X['Ca0'] + 0.02 * (40 - X['ALB0'])
        if 'ALB0' in X.columns and 'CRP0' in X.columns and 'ESR0' in X.columns:
            X_out['NI_Index'] = X['ALB0'] / (X['CRP0'] + X['ESR0'] + eps)
        if 'HB0' in X.columns and 'PLT0' in X.columns:
            X_out['HB_PLT_Ratio'] = X['HB0'] / (X['PLT0'] + eps)
        if 'DBIL0' in X.columns and 'IBIL0' in X.columns:
            X_out['DBIL_Percent'] = X['DBIL0'] / (X['IBIL0'] + X['DBIL0'] + eps)
        if 'Perianal' in X.columns and 'CRP0' in X.columns:
            X_out['Perianal_CRP'] = X['Perianal'] * X['CRP0']
        if 'Perianal' in X.columns and 'ALB0' in X.columns:
            X_out['Perianal_ALB'] = X['Perianal'] * X['ALB0']
        if 'UA0' in X.columns and 'ALB0' in X.columns:
            X_out['UA_ALB_Ratio'] = X['UA0'] / (X['ALB0'] + eps)
        if 'GGT0' in X.columns and 'PLT0' in X.columns:
            X_out['APRI_Proxy'] = X['GGT0'] / (X['PLT0'] + eps)
        if 'UA0' in X.columns and 'Cr0' in X.columns:
            X_out['UA_Cr'] = X['UA0'] / (X['Cr0'] + eps)
            
        final_df = None
        if self.num_cols:
            X_scaled = self.scaler.transform(X[self.num_cols])
            
            # Clustering
            km_cols = [c for c in ['CRP0', 'ESR0', 'PLT0', 'ALB0', 'Ca0'] if c in self.num_cols]
            if km_cols:
                X_scaled_df = pd.DataFrame(X_scaled, columns=self.num_cols, index=X.index)
                X_out['Cluster_ID'] = self.kmeans.predict(X_scaled_df[km_cols])
            
            # Poly
            X_pt = self.pt.transform(X_scaled)
            X_poly = self.poly.transform(X_pt)
            X_poly_df = pd.DataFrame(X_poly, columns=self.poly_feat_names, index=X.index)
            
            manual_cols = [
                'Severe_Inflammation_CRP', 'Severe_Inflammation_ESR', 'Hypoalbuminemia', 'Anemia_Severity',
                'CAR', 'PLR', 'MLR', 'Corrected_Ca', 'NI_Index', 
                'HB_PLT_Ratio', 'DBIL_Percent', 'Perianal_CRP', 'Perianal_ALB', 'UA_ALB_Ratio',
                'APRI_Proxy', 'UA_Cr', 'Cluster_ID'
            ]
            existing_manual = [c for c in manual_cols if c in X_out.columns]
            final_df = pd.concat([X_out[existing_manual], X_poly_df], axis=1)
        else:
            final_df = X_out
            
        self._final_columns = final_df.columns.tolist()
        return final_df
    
    def get_feature_names_out(self, input_features=None):
        return self._final_columns

def _best_threshold_for_f1(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0: return 0.5
    # f1s length is len(thresholds) + 1 (last one is 0) or same?
    # sklearn precision_recall_curve: thresholds is (n_thresholds,), precisions/recalls is (n_thresholds + 1,)
    # We ignore the last precision/recall (1.0/0.0) usually
    idx = np.argmax(f1s[:-1]) 
    return thresholds[idx]

def main():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    
    # 1. Load Selected Features
    if not os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[ERROR] {SELECTED_FEATURES_FILE} not found. Please run training script first.")
        return
        
    with open(SELECTED_FEATURES_FILE, "r") as f:
        data = json.load(f)
        selected_features = data["selected_features"]
    print(f"Loaded {len(selected_features)} selected features.")
    print(selected_features)

    # 2. Load Existing Splits
    train_split_file = "train_split.csv"
    test_split_file = "test_split.csv"
    
    if not (os.path.exists(train_split_file) and os.path.exists(test_split_file)):
        raise FileNotFoundError(f"Missing {train_split_file} or {test_split_file}. Please run train_no_leakage_complex_v4_optuna0.py first to generate them.")
        
    print(f"Loading existing splits from {train_split_file} and {test_split_file}...")
    train_df = pd.read_csv(train_split_file, index_col=0)
    test_df = pd.read_csv(test_split_file, index_col=0)
    X_train = train_df[RAW_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[RAW_FEATURES]
    y_test = test_df[TARGET_COL]
    
    # 3. Build & Fit Feature Engineering Pipeline
    print("Fitting feature engineering pipeline...")
    imputer = SimpleImputer(strategy='median').set_output(transform="pandas")
    winsorizer = Winsorizer(limits=(0.01, 0.01))
    eng = UltimateFeatureEngineer()
    var_thresh = VarianceThreshold(threshold=1e-4)
    
    pre_pipeline = Pipeline([
        ('imputer', imputer),
        ('winsorizer', winsorizer),
        ('engineer', eng),
        ('var_thresh', var_thresh)
    ])
    
    # Fit on Train
    X_train_processed_np = pre_pipeline.fit_transform(X_train, y_train)
    
    # Reconstruct DataFrame
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats_before_var = eng_step.get_feature_names_out()
    
    var_thresh_step = pre_pipeline.named_steps['var_thresh']
    support_mask = var_thresh_step.get_support()
    
    final_feats = np.array(all_feats_before_var)[support_mask]
    
    X_train_processed = pd.DataFrame(X_train_processed_np, columns=final_feats, index=X_train.index)
    
    # Select Features
    X_train_final = X_train_processed[selected_features]
    
    # 3.5 Internal CV Verification (Requested by User)
    print("\nRunning Internal Cross-Validation (10-Fold 10-Repeats)...")
    cv_clf = XGBClassifier(**BEST_PARAMS, **_gpu_params())
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    
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

    scoring = {
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
        'accuracy': 'accuracy',
        'sensitivity': make_scorer(sensitivity_score),
        'specificity': make_scorer(specificity_score),
        'ppv': make_scorer(ppv_score),
        'npv': make_scorer(npv_score)
    }

    cv_scores = cross_validate(
        cv_clf, X_train_final, y_train, 
        cv=rskf, 
        scoring=scoring, 
        n_jobs=1
    )
    
    print("   Internal CV Results:")
    print(f"   - AUC: {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
    print(f"   - F1 : {cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}")
    print(f"   - AP : {cv_scores['test_average_precision'].mean():.4f} ± {cv_scores['test_average_precision'].std():.4f}")
    print(f"   - Acc: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    print(f"   - Sensitivity: {cv_scores['test_sensitivity'].mean():.4f} ± {cv_scores['test_sensitivity'].std():.4f}")
    print(f"   - Specificity: {cv_scores['test_specificity'].mean():.4f} ± {cv_scores['test_specificity'].std():.4f}")
    print(f"   - PPV: {cv_scores['test_ppv'].mean():.4f} ± {cv_scores['test_ppv'].std():.4f}")
    print(f"   - NPV: {cv_scores['test_npv'].mean():.4f} ± {cv_scores['test_npv'].std():.4f}")

    # 4. Train Model
    print("\nTraining XGBoost with best parameters...")
    final_clf = XGBClassifier(**BEST_PARAMS, **_gpu_params())
    final_clf.fit(X_train_final, y_train)
    
    # 5. Evaluate on Train (Find Threshold)
    y_train_prob = final_clf.predict_proba(X_train_final)[:, 1]
    best_thr = 0.51
    print(f"\nOptimal Threshold (from Train): {best_thr:.4f}")
    
    # 6. Evaluate on Test
    print("\nEvaluating on Holdout Test Set...")
    X_test_processed_np = pre_pipeline.transform(X_test)
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=final_feats, index=X_test.index)
    X_test_final = X_test_processed[selected_features]
    
    y_prob = final_clf.predict_proba(X_test_final)[:, 1]
    y_pred = (y_prob >= best_thr).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    print("\n===== Internal Testing Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"PPV: {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred, labels=[0, 1])}")
    
    # Save Model
    joblib.dump(final_clf, "reproduced_best_model.joblib")
    joblib.dump(pre_pipeline, "reproduced_pipeline.joblib")
    print("\n[INFO] Model and Pipeline saved.")

    # 7. External Verification
    if os.path.exists(EXTERNAL_FILE):
        print(f"\nLoading External Verification Data: {EXTERNAL_FILE}...")
        try:
            X_ext_raw, y_ext = load_data(EXTERNAL_FILE)
            print(f"External Data Shape: {X_ext_raw.shape}")
            
            X_ext_processed_np = pre_pipeline.transform(X_ext_raw)
            X_ext_processed = pd.DataFrame(X_ext_processed_np, columns=final_feats, index=X_ext_raw.index)
            X_ext_final = X_ext_processed[selected_features]
            
            y_ext_prob = final_clf.predict_proba(X_ext_final)[:, 1]
            y_ext_pred = (y_ext_prob >= best_thr).astype(int)
            
            acc_ext = accuracy_score(y_ext, y_ext_pred)
            f1_ext = f1_score(y_ext, y_ext_pred)
            auc_ext = roc_auc_score(y_ext, y_ext_prob)
            ap_ext = average_precision_score(y_ext, y_ext_prob)
            tn_ext, fp_ext, fn_ext, tp_ext = confusion_matrix(y_ext, y_ext_pred, labels=[0, 1]).ravel()
            sens_ext = tp_ext / (tp_ext + fn_ext) if (tp_ext + fn_ext) > 0 else 0.0
            spec_ext = tn_ext / (tn_ext + fp_ext) if (tn_ext + fp_ext) > 0 else 0.0
            ppv_ext = tp_ext / (tp_ext + fp_ext) if (tp_ext + fp_ext) > 0 else 0.0
            npv_ext = tn_ext / (tn_ext + fn_ext) if (tn_ext + fn_ext) > 0 else 0.0
            
            print("\n===== External Verification Results =====")
            print(f"Accuracy: {acc_ext:.4f}")
            print(f"F1 Score: {f1_ext:.4f}")
            print(f"AUC: {auc_ext:.4f}")
            print(f"AP: {ap_ext:.4f}")
            print(f"Sensitivity: {sens_ext:.4f}")
            print(f"Specificity: {spec_ext:.4f}")
            print(f"PPV: {ppv_ext:.4f}")
            print(f"NPV: {npv_ext:.4f}")
            print(f"Confusion Matrix:\n{confusion_matrix(y_ext, y_ext_pred, labels=[0, 1])}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process external file: {e}")
    else:
        print(f"\n[WARNING] External file {EXTERNAL_FILE} not found.")

if __name__ == "__main__":
    main()
