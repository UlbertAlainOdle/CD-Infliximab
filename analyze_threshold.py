# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import json
import os
import sys

# Try to use scienceplots for publication-quality figures
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    sns.set_style("whitegrid")

# --- COPIED FROM reproduce_best_model.py ---
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
TRAIN_FILE = "processed_with_features_LEFT_JOIN_all_rows.csv"
SELECTED_FEATURES_FILE = "selected_features.json"

# Best Params (Ensure these match reproduce_best_model.py)
BEST_PARAMS = {'n_estimators': 137, 'max_depth': 10, 'learning_rate': 0.017444824475985048, 'subsample': 0.9997391245814459, 'colsample_bytree': 0.9415743112958483, 'gamma': 1.7483463681615965, 'min_child_weight': 2, 'reg_alpha': 0.04845238191904519, 'reg_lambda': 0.0121275899499253, 'scale_pos_weight': 4.045168009584749}
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
        print(f"Error: Data file not found at {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found.")
    df = df[RAW_FEATURES + [TARGET_COL]]
    return df[RAW_FEATURES], df[TARGET_COL]

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits
        self.percentiles_ = {}
        self.numeric_cols = []
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if not self.numeric_cols: return self
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
            if km_cols: self.kmeans.fit(X_scaled_df[km_cols])
            self.pt.fit(X_scaled)
            X_pt = self.pt.transform(X_scaled)
            self.poly.fit(X_pt)
            self.poly_feat_names = self.poly.get_feature_names_out(self.num_cols)
        return self
    def transform(self, X):
        X_out = X.copy()
        eps = 1e-6
        if 'CRP0' in X.columns: X_out['Severe_Inflammation_CRP'] = (X['CRP0'] > 40).astype(int)
        if 'ESR0' in X.columns: X_out['Severe_Inflammation_ESR'] = (X['ESR0'] > 50).astype(int)
        if 'ALB0' in X.columns: X_out['Hypoalbuminemia'] = (X['ALB0'] < 35).astype(int)
        if 'HB0' in X.columns:
            conditions = [(X['HB0'] >= 120), (X['HB0'] >= 90) & (X['HB0'] < 120), (X['HB0'] >= 60) & (X['HB0'] < 90), (X['HB0'] < 60)]
            X_out['Anemia_Severity'] = np.select(conditions, [0, 1, 2, 3], default=0)
        if 'CRP0' in X.columns and 'ALB0' in X.columns: X_out['CAR'] = X['CRP0'] / (X['ALB0'] + eps)
        if 'PLT0' in X.columns and 'Lyn0' in X.columns: X_out['PLR'] = X['PLT0'] / (X['Lyn0'] + eps)
        if 'M0' in X.columns and 'Lyn0' in X.columns: X_out['MLR'] = X['M0'] / (X['Lyn0'] + eps)
        if 'Ca0' in X.columns and 'ALB0' in X.columns: X_out['Corrected_Ca'] = X['Ca0'] + 0.02 * (40 - X['ALB0'])
        if 'ALB0' in X.columns and 'CRP0' in X.columns and 'ESR0' in X.columns: X_out['NI_Index'] = X['ALB0'] / (X['CRP0'] + X['ESR0'] + eps)
        if 'HB0' in X.columns and 'PLT0' in X.columns: X_out['HB_PLT_Ratio'] = X['HB0'] / (X['PLT0'] + eps)
        if 'DBIL0' in X.columns and 'IBIL0' in X.columns: X_out['DBIL_Percent'] = X['DBIL0'] / (X['IBIL0'] + X['DBIL0'] + eps)
        if 'Perianal' in X.columns and 'CRP0' in X.columns: X_out['Perianal_CRP'] = X['Perianal'] * X['CRP0']
        if 'Perianal' in X.columns and 'ALB0' in X.columns: X_out['Perianal_ALB'] = X['Perianal'] * X['ALB0']
        if 'UA0' in X.columns and 'ALB0' in X.columns: X_out['UA_ALB_Ratio'] = X['UA0'] / (X['ALB0'] + eps)
        if 'GGT0' in X.columns and 'PLT0' in X.columns: X_out['APRI_Proxy'] = X['GGT0'] / (X['PLT0'] + eps)
        if 'UA0' in X.columns and 'Cr0' in X.columns: X_out['UA_Cr'] = X['UA0'] / (X['Cr0'] + eps)
        final_df = None
        if self.num_cols:
            X_scaled = self.scaler.transform(X[self.num_cols])
            km_cols = [c for c in ['CRP0', 'ESR0', 'PLT0', 'ALB0', 'Ca0'] if c in self.num_cols]
            if km_cols:
                X_scaled_df = pd.DataFrame(X_scaled, columns=self.num_cols, index=X.index)
                X_out['Cluster_ID'] = self.kmeans.predict(X_scaled_df[km_cols])
            X_pt = self.pt.transform(X_scaled)
            X_poly = self.poly.transform(X_pt)
            X_poly_df = pd.DataFrame(X_poly, columns=self.poly_feat_names, index=X.index)
            manual_cols = ['Severe_Inflammation_CRP', 'Severe_Inflammation_ESR', 'Hypoalbuminemia', 'Anemia_Severity', 'CAR', 'PLR', 'MLR', 'Corrected_Ca', 'NI_Index', 'HB_PLT_Ratio', 'DBIL_Percent', 'Perianal_CRP', 'Perianal_ALB', 'UA_ALB_Ratio', 'APRI_Proxy', 'UA_Cr', 'Cluster_ID']
            existing_manual = [c for c in manual_cols if c in X_out.columns]
            final_df = pd.concat([X_out[existing_manual], X_poly_df], axis=1)
        else: final_df = X_out
        self._final_columns = final_df.columns.tolist()
        return final_df
    def get_feature_names_out(self, input_features=None): return self._final_columns

# --- END COPIED SECTION ---

def analyze_thresholds():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    
    # Use the same split as training (80% train, 20% test)
    # We only analyze the TRAIN set for threshold finding (Internal CV)
    X_train, _, y_train, _ = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Internal Data Shape: {X_train.shape}")
    
    # Load Selected Features if available
    selected_features = None
    if os.path.exists(SELECTED_FEATURES_FILE):
        with open(SELECTED_FEATURES_FILE, "r") as f:
            selected_features = json.load(f)["selected_features"]
            print(f"Loaded {len(selected_features)} selected features.")
    
    # Construct Full Pipeline
    imputer = SimpleImputer(strategy='median').set_output(transform="pandas")
    winsorizer = Winsorizer(limits=(0.01, 0.01))
    eng = UltimateFeatureEngineer()
    var_thresh = VarianceThreshold(threshold=1e-4)
    
    # Note: We need to handle feature selection manually after pipeline transform
    # because Pipeline doesn't support 'column subsetting' easily in middle steps unless we write a custom selector.
    # So we'll run preprocessing first, then subset features.
    
    print("Preprocessing and Feature Engineering...")
    pre_pipeline = Pipeline([
        ('imputer', imputer),
        ('winsorizer', winsorizer),
        ('engineer', eng),
        ('var_thresh', var_thresh)
    ])
    
    X_train_processed_np = pre_pipeline.fit_transform(X_train, y_train)
    
    # Reconstruct DataFrame
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats = eng_step.get_feature_names_out()
    var_step = pre_pipeline.named_steps['var_thresh']
    support = var_step.get_support()
    final_feats = np.array(all_feats)[support]
    
    X_train_processed = pd.DataFrame(X_train_processed_np, columns=final_feats, index=X_train.index)
    
    # Select Features
    if selected_features:
        # Ensure all selected features exist
        valid_feats = [f for f in selected_features if f in X_train_processed.columns]
        if len(valid_feats) < len(selected_features):
            print(f"Warning: {len(selected_features) - len(valid_feats)} features missing from processed data.")
        X_train_sel = X_train_processed[valid_feats]
    else:
        print("Warning: No selected features found. Using all processed features.")
        X_train_sel = X_train_processed
        
    print(f"Final Feature Matrix for CV: {X_train_sel.shape}")
    
    # Initialize Model
    clf = XGBClassifier(**BEST_PARAMS, **_gpu_params(), random_state=42)
    
    print("Running cross_val_predict (10-Fold)...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Force CPU to avoid hang
    cpu_params = BEST_PARAMS.copy()
    if 'device' in cpu_params: del cpu_params['device']
    if 'tree_method' in cpu_params: del cpu_params['tree_method']
    clf = XGBClassifier(**cpu_params, tree_method="hist", random_state=42, n_jobs=1)
    
    # Use cross_val_predict to get OOF probabilities
    # method='predict_proba' returns (n_samples, n_classes)
    y_probas = cross_val_predict(clf, X_train_sel, y_train, cv=skf, method='predict_proba', n_jobs=1)
    y_scores = y_probas[:, 1] # Probability of positive class
    
    # Analyze Thresholds
    thresholds = np.arange(0.1, 0.96, 0.01)
    f1_scores = []
    
    print("Calculating F1 scores across thresholds...")
    for thr in thresholds:
        y_pred_thr = (y_scores >= thr).astype(int)
        score = f1_score(y_train, y_pred_thr)
        f1_scores.append(score)
    
    f1_scores = np.array(f1_scores)
    
    # Find Optimal
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Find Default (0.5)
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    default_f1 = f1_scores[default_idx]
    
    print(f"\nResults:")
    print(f"Optimal Threshold: {best_thr:.2f} (F1: {best_f1:.4f})")
    print(f"Default Threshold: 0.50 (F1: {default_f1:.4f})")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, color='#1f77b4', linewidth=2, label='F1 Score Curve')
    
    # Mark Optimal
    plt.axvline(x=best_thr, color='#d62728', linestyle='--', linewidth=1.5, 
                label=f'Optimal Threshold: {best_thr:.2f}\nMax F1: {best_f1:.4f}')
    plt.scatter(best_thr, best_f1, color='#d62728', s=50, zorder=5)
    
    # Mark Default
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Default Threshold: 0.50\nF1: {default_f1:.4f}')
    plt.scatter(0.5, default_f1, color='gray', s=50, zorder=5)
    
    plt.title('F1 Score vs. Decision Threshold (Internal 10-Fold CV)')
    plt.xlabel('Decision Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.xlim(0.1, 0.95)
    plt.ylim(0.0, 1.0)
    
    output_file = "f1_threshold_analysis.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    analyze_thresholds()
