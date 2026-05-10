# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import json
import os
import sys

# Global Config
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
SELECTED_FEATURES_FILE = "selected_features.json"
PIPELINE_FILE = "reproduced_pipeline.joblib"
MODEL_FILE = "reproduced_best_model.joblib"

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk', index_col=0)
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found.")
    
    # We need to keep track of indices or IDs if possible, but here we just rely on DataFrame index
    # We keep all columns initially to facilitate joining later if needed, but for X we select RAW_FEATURES
    return df

# --- Feature Engineering Classes (COPIED EXACTLY) ---
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

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=None):
        self.selected_features = selected_features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.selected_features is None:
            return X
        valid_feats = [f for f in self.selected_features if f in X.columns]
        return X[valid_feats]
    def get_feature_names_out(self, input_features=None):
        return self.selected_features

def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0: return 0.5
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx]

def run_hard_sample_analysis():
    print("Loading data splits...")
    
    # Check if the split files exist and load them to enforce the fixed subset order and properties.
    train_split_file = "train_split.csv"
    test_split_file = "test_split.csv"
    if not (os.path.exists(train_split_file) and os.path.exists(test_split_file)):
        print(f"[ERROR] Missing {train_split_file} or {test_split_file}. Please run train_no_leakage_complex_v4_optuna0.py first.")
        return
        
    print(f"Loading fixed data splits from {train_split_file} and {test_split_file}...")
    train_df = load_data(train_split_file)
    test_df = load_data(test_split_file)
    # Reconstruct df_full using the exact order: Train then Test
    df_full = pd.concat([train_df, test_df], axis=0)
    
    # Load Selected Features
    if not os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[ERROR] {SELECTED_FEATURES_FILE} not found.")
        return
    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected_features = json.load(f)["selected_features"]
    print(f"Loaded {len(selected_features)} selected features.")
    
    X = df_full[RAW_FEATURES]
    y = df_full[TARGET_COL]
    
    import joblib
    print(f"Loading pipeline from {PIPELINE_FILE}...")
    if not os.path.exists(PIPELINE_FILE) or not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Missing {PIPELINE_FILE} or {MODEL_FILE}")
        return
        
    pre_pipeline = joblib.load(PIPELINE_FILE)
    print(f"Loading model from {MODEL_FILE}...")
    clf = joblib.load(MODEL_FILE)
    
    print("Generating Engineered Features for Display...")
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats_before_var = eng_step.get_feature_names_out()
    
    var_thresh_step = pre_pipeline.named_steps['var_thresh']
    support_mask = var_thresh_step.get_support()
    final_feats = np.array(all_feats_before_var)[support_mask]
    
    X_final_np = pre_pipeline.transform(X)
    X_final_processed = pd.DataFrame(X_final_np, columns=final_feats, index=X.index)
    X_final = X_final_processed[selected_features]
    
    print("Running Inference with Loaded Model...")
    # The loaded model expects only the selected_features
    y_oof_prob = clf.predict_proba(X_final)[:, 1]
    
    # Find Optimal Threshold
    best_thr = 0.53
    print(f"Using locked threshold: {best_thr:.4f}")
    
    y_pred = (y_oof_prob >= best_thr).astype(int)
    
    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({
        'Target': y,
        'Prob': y_oof_prob,
        'Pred': y_pred
    }, index=X.index)
    
    # Determine Categories
    conditions = [
        (analysis_df['Target'] == 1) & (analysis_df['Pred'] == 1), # TP
        (analysis_df['Target'] == 0) & (analysis_df['Pred'] == 0), # TN
        (analysis_df['Target'] == 0) & (analysis_df['Pred'] == 1), # FP
        (analysis_df['Target'] == 1) & (analysis_df['Pred'] == 0)  # FN
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    analysis_df['Category'] = np.select(conditions, choices, default="Unknown")
    
    # Merge with Raw Features and Engineered Features (Adding Prefixes to avoid column collisions)
    raw_df = df_full[RAW_FEATURES].add_prefix('Raw_')
    eng_df = X_final.add_prefix('Eng_')
    analysis_df = pd.concat([analysis_df, raw_df, eng_df], axis=1)
    
    print("\n=== Sample Distribution ===")
    print(analysis_df['Category'].value_counts())
    
    # 1. Quantify Errors
    print("\n=== Error Analysis (Quantify) ===")
    error_df = analysis_df[analysis_df['Category'].isin(['FP', 'FN'])]
    correct_df = analysis_df[analysis_df['Category'].isin(['TP', 'TN'])]
    
    print(f"Total Errors: {len(error_df)} / {len(analysis_df)} ({len(error_df)/len(analysis_df):.2%})")
    print("\nError Probabilities Distribution:")
    print(error_df[['Category', 'Target', 'Prob']].describe())
    
    print("\nFalse Positives (Target=0, Pred=1): High prob indicates confident error")
    fp_df = error_df[error_df['Category'] == 'FP']
    if not fp_df.empty:
        print(fp_df[['Prob']].describe().T)
    else:
        print("No FP samples.")
        
    print("\nFalse Negatives (Target=1, Pred=0): Low prob indicates confident error")
    fn_df = error_df[error_df['Category'] == 'FN']
    if not fn_df.empty:
        print(fn_df[['Prob']].describe().T)
    else:
        print("No FN samples.")
        
    # 2. Compare Feature Means
    print("\n=== Feature Comparison (Correct vs Error) ===")
    # Compare Engineered Features
    print(f"{'Feature':<30} | {'Correct Mean':<12} | {'Error Mean':<12} | {'Diff':<12} | {'P-Value':<10}")
    print("-" * 90)
    
    for feat in eng_df.columns:
        correct_vals = correct_df[feat].astype(float) # Ensure float type
        error_vals = error_df[feat].astype(float)
        
        mean_c = correct_vals.mean()
        mean_e = error_vals.mean()
        diff = mean_e - mean_c
        
        # Ensure scalar
        if isinstance(mean_c, (pd.Series, np.ndarray)) and mean_c.size == 1: mean_c = mean_c.item()
        if isinstance(mean_e, (pd.Series, np.ndarray)) and mean_e.size == 1: mean_e = mean_e.item()
        if isinstance(diff, (pd.Series, np.ndarray)) and diff.size == 1: diff = diff.item()
        
        # T-test (ind)
        if len(correct_vals) > 1 and len(error_vals) > 1:
            try:
                t_stat, p_val = ttest_ind(correct_vals, error_vals, equal_var=False)
                # Ensure p_val is a scalar float
                if isinstance(p_val, np.ndarray):
                    if p_val.size == 1:
                         p_val = p_val.item()
                    else:
                         p_val = p_val[0]
                p_str = f"{p_val:.4f}"
            except Exception as e:
                p_str = "Error"
        else:
            p_str = "N/A"
            
        # Fallback if still not scalar (should happen if column duplicated)
        if isinstance(mean_c, pd.Series): mean_c = mean_c.iloc[0]
        if isinstance(mean_e, pd.Series): mean_e = mean_e.iloc[0]
        if isinstance(diff, pd.Series): diff = diff.iloc[0]

        print(f"{feat:<30} | {mean_c:<12.4f} | {mean_e:<12.4f} | {diff:<12.4f} | {p_str:<10}")
        
    # 3. Print Hard Sample Data
    print("\n=== Hard Sample Data (FP & FN) ===")
    # Output columns: Category, Prob, Target, + Raw Features + Engineered Features
    cols_to_print = ['Category', 'Prob', 'Target'] + list(raw_df.columns) + list(eng_df.columns)
    hard_samples = error_df[cols_to_print].sort_values(by=['Category', 'Prob'])
    
    # Print to console (formatted)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    print(hard_samples)
    
    # Save to CSV
    output_csv = "hard_sample_analysis_data.csv"
    hard_samples.to_csv(output_csv, index=True)
    print(f"\nHard sample data saved to {output_csv}")

    # Task 1: Export full analysis dataframe to full_sample_analysis_data.csv
    # Ensure all columns are present and unique
    full_samples = analysis_df[cols_to_print].copy()
    full_output_csv = "full_sample_analysis_data.csv"
    full_samples.to_csv(full_output_csv, index=True)
    print(f"\nFull sample analysis data saved to {full_output_csv}")

if __name__ == "__main__":
    run_hard_sample_analysis()
