# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import json
import os
import sys
import joblib

# Try to use scienceplots for publication-quality figures
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    sns.set_style("whitegrid")

# Global Config
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
TRAIN_FILE = "processed_with_features_LEFT_JOIN_all_rows.csv"
SELECTED_FEATURES_FILE = "selected_features.json"

# Best Params (From reproduce_best_model.py)
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

def run_shap_analysis():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    
    # Load Selected Features
    if not os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[ERROR] {SELECTED_FEATURES_FILE} not found.")
        return
    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected_features = json.load(f)["selected_features"]
    print(f"Loaded {len(selected_features)} selected features.")
    
    # Load Fixed Preprocessing Pipeline
    pipeline_file = "reproduced_pipeline.joblib"
    if not os.path.exists(pipeline_file):
        print(f"[ERROR] {pipeline_file} not found. Please run reproduce_best_model0.py first.")
        return
    print(f"Loading fixed preprocessing pipeline from {pipeline_file}...")
    pre_pipeline = joblib.load(pipeline_file)
    
    # Load Fixed Splits
    train_split_file = "train_split.csv"
    test_split_file = "test_split.csv"
    if not (os.path.exists(train_split_file) and os.path.exists(test_split_file)):
        print(f"[ERROR] Missing {train_split_file} or {test_split_file}. Please run train_no_leakage_complex_v4_optuna0.py first.")
        return
        
    print(f"Loading fixed data splits from {train_split_file} and {test_split_file}...")
    train_df = pd.read_csv(train_split_file, index_col=0)
    test_df = pd.read_csv(test_split_file, index_col=0)
    X_train_raw = train_df[RAW_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test_raw = test_df[RAW_FEATURES]
    y_test = test_df[TARGET_COL]
    
    print("Generating Engineered Features using loaded pipeline...")
    
    # We need all feature names before variance threshold
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats_before_var = eng_step.get_feature_names_out()
    
    var_thresh_step = pre_pipeline.named_steps['var_thresh']
    support_mask = var_thresh_step.get_support()
    final_feats = np.array(all_feats_before_var)[support_mask]
    
    # Transform TRAIN and TEST
    X_train_processed_np = pre_pipeline.transform(X_train_raw)
    X_train_processed = pd.DataFrame(X_train_processed_np, columns=final_feats, index=X_train_raw.index)
    X_train_final = X_train_processed[selected_features]
    
    X_test_processed_np = pre_pipeline.transform(X_test_raw)
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=final_feats, index=X_test_raw.index)
    X_test_final = X_test_processed[selected_features]
    
    # Also prepare External Data if available
    X_ext_final = None
    y_ext = None
    if os.path.exists("External Verification00.csv"):
        try:
            print("Loading External Data...")
            X_ext_raw, y_ext = load_data("External Verification00.csv")
            X_ext_processed_np = pre_pipeline.transform(X_ext_raw)
            X_ext_processed = pd.DataFrame(X_ext_processed_np, columns=final_feats, index=X_ext_raw.index)
            X_ext_final = X_ext_processed[selected_features]
        except Exception as e:
            print(f"External data load failed: {e}")

    # Load Fixed Best Model
    model_file = "reproduced_best_model.joblib"
    if not os.path.exists(model_file):
        print(f"[ERROR] {model_file} not found. Please run reproduce_best_model0.py first.")
        return
    print(f"Loading fixed XGBoost model from {model_file}...")
    clf = joblib.load(model_file)
    
    print("Calculating SHAP values (using Internal Test Set + External Validation Set)...")
    # Merge datasets for broader candidate search
    X_target_internal = X_test_final
    y_target_internal = y_test
    
    # Create a source label to track where the sample came from
    # 0 for Internal, 1 for External
    sources = pd.Series(["Internal"] * len(X_target_internal), index=X_target_internal.index)
    
    if X_ext_final is not None:
        X_target = pd.concat([X_target_internal, X_ext_final], axis=0)
        y_target = pd.concat([y_target_internal, y_ext], axis=0)
        sources_ext = pd.Series(["External"] * len(X_ext_final), index=X_ext_final.index)
        sources = pd.concat([sources, sources_ext], axis=0)
    else:
        X_target = X_target_internal
        y_target = y_target_internal
    
    print(f"Total Candidate Samples: {len(X_target)} (Internal: {len(X_target_internal)}, External: {len(X_ext_final) if X_ext_final is not None else 0})")
    
    # Use get_booster() to avoid some SKLearn wrapper issues with SHAP
    booster = clf.get_booster()
    
    # Fix for XGBoost 2.0+ saving base_score as a JSON list string which SHAP < 0.46 (or incompatible) fails to parse
    # We patch the class method temporarily because instance patching might be bypassed or failing.
    
    original_save_config = xgb.Booster.save_config
    
    def fixed_save_config(self):
        config_str = original_save_config(self)
        config = json.loads(config_str)
        try:
             if 'learner' in config and 'learner_model_param' in config['learner']:
                 p = config['learner']['learner_model_param']
                 if 'base_score' in p and isinstance(p['base_score'], str) and p['base_score'].startswith('['):
                     # Fix: extract the number (remove brackets)
                     p['base_score'] = p['base_score'].strip('[]')
        except Exception as e:
            print(f"Warning: could not patch base_score: {e}")
        return json.dumps(config)
    
    # Apply patch to class
    xgb.Booster.save_config = fixed_save_config
    
    try:
        # data=X_target allows TreeExplainer to use data-dependent masking (interventional)
        # model_output="probability" converts log-odds to probability space directly!
        explainer = shap.TreeExplainer(booster, data=X_target, model_output="probability")
    except Exception as e:
        print(f"TreeExplainer failed even with class patch: {e}. Falling back to KernelExplainer (slow)...")
        # Fallback to KernelExplainer if absolutely necessary
        # Use lambda to avoid 'feature_names_in_' attribute setter error on XGBClassifier
        background = shap.sample(X_train_final, min(100, len(X_train_final)), random_state=42)
        explainer = shap.TreeExplainer(
            booster,
            data=background,
            model_output="probability"
        )
    finally:
        # Restore original method
        xgb.Booster.save_config = original_save_config
    
    # Calculate SHAP values
    # For TreeExplainer with model_output="probability", shap_values are in probability space
    # It returns an Explanation object directly if called as explainer(X)
    # If called as explainer.shap_values(X), it returns array.
    # We prefer the Explanation object for waterfall plots.
    
    try:
        shap_explanation = explainer(X_target)
    except Exception as e:
        print(f"Explainer call failed: {e}")
        return

    # shap_explanation.values shape: (n_samples, n_features)
    # If model_output="probability", values sum to prob - base_value
    
    shap_values = shap_explanation.values
    
    # Ensure numpy array and correct shape
    # If binary classification, sometimes it returns (samples, features, classes)?
    # TreeExplainer with 'probability' usually returns just for the output (if binary, maybe just positive class?)
    # Let's check shape
    print(f"SHAP values shape: {shap_values.shape}")
    
    # 1. Summary Plot (Overall Importance)
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    display_name_map = {
        "DBIL0 Ca0": "DBIL0 × Ca0",
        "ESR0 ALB0": "ESR0 × ALB0",
        "IBIL0 Ca0": "IBIL0 × Ca0",
        "GGT0 DBIL0": "GGT0 × DBIL0",
        "Perianal UA0": "Perianal × UA0",
        "ESR0 Cr0": "ESR0 × Cr0",
        "ESR0 CRP0": "ESR0 × CRP0",
        "ALB0 Ca0": "ALB0 × Ca0",
        "ESR0 IBIL0": "ESR0 × IBIL0",
    }
    X_target_display = X_target.rename(columns=display_name_map)
    shap.summary_plot(
        shap_values,
        X_target_display,
        show=False,
        alpha=0.65,
        plot_size=None
    )

    fig = plt.gcf()
    ax = plt.gca()

    # 去掉黑色方框/边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 只保留每个特征位置的横向虚线，去掉竖向网格
    ax.grid(False)
    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )
    ax.xaxis.grid(False)

    # 保留 x=0 的参考线
    ax.axvline(0, color="gray", linewidth=1.0, alpha=0.8)

    # 白色背景
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.title("SHAP Feature Importance (Combined Test & External Sets)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        "shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()
    print("Saved shap_summary_plot.png")
    
    # 2. Dependence Plots for Top Features
    # Identify top features by mean absolute SHAP value
    mean_shap = np.abs(shap_values).mean(axis=0)
    # Ensure mean_shap is 1D
    if len(mean_shap.shape) > 1:
         mean_shap = mean_shap.flatten()
         
    feature_importance = pd.DataFrame({
        'feature': X_target.columns,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance['feature'].head(4).tolist()
    print(f"Top 4 features: {top_features}")
    
    for feat in top_features:
        print(f"Generating Dependence Plot for {feat}...")
        plt.figure(figsize=(8, 6))
        # dependence_plot usually creates its own figure, but we can try to control it
        # We pass the feature name and the matrix
        shap.dependence_plot(feat, shap_values, X_target, show=False)
        plt.title(f"SHAP Dependence Plot: {feat}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{feat.replace(' ', '_')}.png", dpi=300)
        plt.close()

    print("SHAP analysis complete.")

if __name__ == "__main__":
    run_shap_analysis()
