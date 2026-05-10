# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import os
import sys

# Try to use scienceplots for publication-quality figures
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    # Fallback to seaborn style if scienceplots is not installed
    sns.set_style("whitegrid")

# --- COPIED FROM train_no_leakage_complex_v4_optuna.py ---
# Global Config
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
TRAIN_FILE = "processed_with_features_LEFT_JOIN_all_rows.csv"

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
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        sys.exit(1)
        
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
            # Fit transforms
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

# --- END COPIED SECTION ---

def plot_rfecv_curve(rfecv, output_path="rfecv_validation_curve.png"):
    """
    Plots the RFECV validation curve with standard deviation shading.
    """
    n_features = np.array(range(1, len(rfecv.cv_results_['mean_test_score']) + 1))
    mean_scores = rfecv.cv_results_['mean_test_score']
    std_scores = rfecv.cv_results_['std_test_score']
    
    # Calculate optimal number of features
    optimal_n_features = rfecv.n_features_
    best_score = np.max(mean_scores)
    
    # Check score at N=13 (Target features)
    target_n = 13
    if target_n <= len(mean_scores):
        # Index is n-1 because n_features starts at 1
        score_at_target = mean_scores[target_n - 1]
    else:
        score_at_target = 0
        
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot Mean Score
    plt.plot(n_features, mean_scores, 'b-', label='Mean CV Score (AUC)', linewidth=2)
    
    # Fill Standard Deviation
    plt.fill_between(n_features, 
                     mean_scores - std_scores, 
                     mean_scores + std_scores, 
                     alpha=0.2, color='b', label='± 1 Std. Dev.')
    
    # Highlight Optimal Point (Max)
    plt.axvline(x=optimal_n_features, color='g', linestyle=':', alpha=0.6, 
                label=f'Max AUC (N={optimal_n_features}, AUC={best_score:.4f})')
    
    # Highlight Target Point (N=13)
    if score_at_target > 0:
        plt.axvline(x=target_n, color='r', linestyle='--', alpha=0.8, 
                    label=f'Selected (N={target_n}, AUC={score_at_target:.4f})')
        plt.plot(target_n, score_at_target, 'ro')

    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-Validation ROC AUC Score")
    plt.title("Recursive Feature Elimination with Cross-Validation (RFECV)")
    plt.legend(loc="lower right")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Improve layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

def run_rfecv_experiment():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    
    # Use the same split as training to ensure consistency (train set only)
    X_train, _, y_train, _ = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Preprocessing features...")
    # Re-create pipeline steps locally to ensure we have access to intermediate data
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
    
    # Fit and transform to get the feature matrix ready for RFECV
    X_processed_np = pre_pipeline.fit_transform(X_train, y_train)
    
    # Reconstruct DataFrame with feature names
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats = eng_step.get_feature_names_out()
    var_step = pre_pipeline.named_steps['var_thresh']
    support = var_step.get_support()
    final_feats = np.array(all_feats)[support]
    
    X_processed = pd.DataFrame(X_processed_np, columns=final_feats, index=X_train.index)
    
    print(f"Starting RFECV on {X_processed.shape[1]} initial features...")
    
    # Initialize XGBoost (Same params as main script)
    estimator = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, 
        n_jobs=1, random_state=42, **_gpu_params()
    )
    
    # Initialize RFECV
    # step=1 means remove 1 feature at a time -> high resolution curve
    # min_features_to_select=1 to show full curve down to 1 feature
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(5),
        scoring='roc_auc',
        min_features_to_select=1, 
        n_jobs=1,
        verbose=1
    )
    
    rfecv.fit(X_processed, y_train)
    
    print(f"RFECV Complete. Optimal number of features: {rfecv.n_features_}")
    print(f"Best CV Score (AUC): {np.max(rfecv.cv_results_['mean_test_score']):.4f}")
    
    # Plotting
    plot_rfecv_curve(rfecv, "rfecv_validation_curve.png")
    
    # Save the selected features to a text file for reference
    selected_features = np.array(final_feats)[rfecv.support_]
    with open("rfecv_optimal_features.txt", "w") as f:
        f.write(f"Optimal Feature Count: {rfecv.n_features_}\n")
        f.write(f"Best CV AUC: {np.max(rfecv.cv_results_['mean_test_score']):.4f}\n")
        f.write("-" * 30 + "\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
    print("Optimal features saved to rfecv_optimal_features.txt")

if __name__ == "__main__":
    run_rfecv_experiment()
