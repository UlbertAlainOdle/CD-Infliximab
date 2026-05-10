# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans

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

def plot_correlation_heatmap():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    
    # Load Selected Features
    if not os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[ERROR] {SELECTED_FEATURES_FILE} not found.")
        return
    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected_features = json.load(f)["selected_features"]
    print(f"Loaded {len(selected_features)} selected features.")
    
    # Prepare Pipeline (Must match reproduce_best_model.py logic)
    imputer = SimpleImputer(strategy='median').set_output(transform="pandas")
    winsorizer = Winsorizer(limits=(0.01, 0.01))
    eng = UltimateFeatureEngineer()
    var_thresh = VarianceThreshold(threshold=1e-4).set_output(transform="pandas")
    selector = FeatureSelector(selected_features=selected_features)
    
    pre_pipeline = Pipeline([
        ('imputer', imputer),
        ('winsorizer', winsorizer),
        ('engineer', eng),
        ('var_thresh', var_thresh),
        ('selector', selector)
    ])
    
    print("Generating Engineered Features...")
    X_final_np = pre_pipeline.fit_transform(X_raw, y)
    X_final = pd.DataFrame(X_final_np, columns=selected_features, index=X_raw.index)
    
    # Calculate Correlation Matrix
    print("Calculating Pearson Correlation...")
    corr_matrix = X_final.corr(method='pearson')
    
    # Plotting
    print("Plotting Heatmap...")
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('white') # Ensure white background for the figure
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    ax = sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot_kws={"size": 8}
    )
    
    # Remove any underlying grid lines and set axes background to white
    ax.grid(False)
    ax.set_facecolor('white')
    
    plt.title('Pearson Correlation Heatmap of Selected Features', fontsize=16)
    plt.tight_layout()
    
    output_file = "pearson_correlation_heatmap.png"
    plt.savefig(output_file, dpi=300)
    print(f"Heatmap saved to {output_file}")

if __name__ == "__main__":
    plot_correlation_heatmap()
