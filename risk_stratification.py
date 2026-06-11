# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
import xgboost as xgb
import scipy.stats as stats

# Global Config
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
SELECTED_FEATURES_FILE = "selected_features.json"

# Best Params (From analyze_threshold.py / reproduce_best_model.py)
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
        df = pd.read_csv(csv_path, index_col=0 if 'split' in csv_path else None)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk', index_col=0 if 'split' in csv_path else None)
    
    # Handle external dataset that might not have index_col=0 saved
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in {csv_path}.")
    
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

def wilson_ci(k, n, confidence=0.95):
    """Calculate the Wilson Score Interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - spread) / denominator, (center + spread) / denominator

def assign_risk_group(p, cutoff):
    if p < cutoff:
        return "Low predicted response"
    else:
        return "High predicted response"

def analyze_risk_strata():
    print("Loading datasets...")
    # Load Internal Set (previously development set)
    X_internal_raw, y_internal = load_data("train_split.csv")
    print(f"Internal Set: {X_internal_raw.shape}")

    # Load External Validation Set
    X_ext_raw, y_ext = load_data("External Verification00.csv")
    print(f"External Validation Set: {X_ext_raw.shape}")

    # Load Selected Features
    if not os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[ERROR] {SELECTED_FEATURES_FILE} not found.")
        sys.exit(1)
    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected_features = json.load(f)["selected_features"]
    print(f"Loaded {len(selected_features)} selected features.")

    # Preprocessing
    print("\nFitting Preprocessing Pipeline on Internal Set...")
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

    X_internal_processed_np = pre_pipeline.fit_transform(X_internal_raw, y_internal)

    # Reconstruct DataFrame with feature names
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats = eng_step.get_feature_names_out()
    var_step = pre_pipeline.named_steps['var_thresh']
    support = var_step.get_support()
    final_feats = np.array(all_feats)[support]

    X_internal_processed = pd.DataFrame(X_internal_processed_np, columns=final_feats, index=X_internal_raw.index)
    X_internal_sel = X_internal_processed[selected_features]

    # Process External
    X_ext_processed_np = pre_pipeline.transform(X_ext_raw)
    X_ext_processed = pd.DataFrame(X_ext_processed_np, columns=final_feats, index=X_ext_raw.index)
    X_ext_sel = X_ext_processed[selected_features]

    # Initialize Model for OOF Probabilities
    print("\n--- Step 1: Generating OOF Probabilities on Internal Set ---")
    cpu_params = BEST_PARAMS.copy()
    if 'device' in cpu_params: del cpu_params['device']
    if 'tree_method' in cpu_params: del cpu_params['tree_method']
    clf_cv = XGBClassifier(**cpu_params, tree_method="hist", random_state=42, n_jobs=1)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_internal_probas = cross_val_predict(clf_cv, X_internal_sel, y_internal, cv=skf, method='predict_proba', n_jobs=1)
    y_internal_scores = y_internal_probas[:, 1]

    # Calculate Cutoff
    print("\n--- Step 2: Calculating Cutoffs using Median ---")
    cutoff = np.median(y_internal_scores)
    print(f"Cutoff (Median/50th percentile): {cutoff:.4f}")

    # Train Final Model
    print("\n--- Step 3: Training Final Model and Predicting ---")
    final_clf = XGBClassifier(**BEST_PARAMS, **_gpu_params(), random_state=42)
    final_clf.fit(X_internal_sel, y_internal)

    # Predict External
    y_ext_scores = final_clf.predict_proba(X_ext_sel)[:, 1]

    # Stratify and Evaluate
    def evaluate_strata(y_true, y_scores, dataset_name):
        print(f"\n[{dataset_name}] Risk Stratification Results:")
        df = pd.DataFrame({'True_Response': y_true, 'Predicted_Prob': y_scores})
        df['Risk_Group'] = df['Predicted_Prob'].apply(lambda p: assign_risk_group(p, cutoff))
        
        # Order the groups for display
        group_order = ["Low predicted response", "High predicted response"]
        
        # Calculate stats
        stats_list = []
        raw_stats = {}
        for group in group_order:
            group_data = df[df['Risk_Group'] == group]
            n_total = len(group_data)
            n_responders = group_data['True_Response'].sum() if n_total > 0 else 0
            raw_stats[group] = {'N': n_total, 'Responders': n_responders}
            
            if n_total > 0:
                response_rate = n_responders / n_total
                ci_lower, ci_upper = wilson_ci(n_responders, n_total)
                stats_list.append({
                    'Group': group,
                    'N': n_total,
                    'Responders': n_responders,
                    'Response_Rate': f"{response_rate:.1%} ({n_responders}/{n_total})",
                    '95%_CI': f"[{ci_lower:.1%} - {ci_upper:.1%}]"
                })
            else:
                stats_list.append({
                    'Group': group,
                    'N': 0,
                    'Responders': 0,
                    'Response_Rate': "N/A",
                    '95%_CI': "N/A"
                })
                
        stats_df = pd.DataFrame(stats_list)
        print(stats_df.to_string(index=False))
        
        # Statistical Tests
        print("  Statistical Tests:")
        # Chi-square and Fisher's Exact Test
        table = [[raw_stats[g]['Responders'], raw_stats[g]['N'] - raw_stats[g]['Responders']] for g in group_order if raw_stats[g]['N'] > 0]
        if len(table) == 2:
            _, p_overall, _, _ = stats.chi2_contingency(table)
            print(f"  - Chi-square p-value: {p_overall:.4f}")
            
            # Fisher's Exact Test
            n1, k1 = raw_stats[group_order[0]]['N'], raw_stats[group_order[0]]['Responders']
            n2, k2 = raw_stats[group_order[1]]['N'], raw_stats[group_order[1]]['Responders']
            if n1 > 0 and n2 > 0:
                _, p_val = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
                print(f"  - Fisher's Exact Test ({group_order[0]} vs {group_order[1]}): p = {p_val:.4f}")
        
        return stats_df

    evaluate_strata(y_internal, y_internal_scores, "Internal Set (OOF)")
    evaluate_strata(y_ext, y_ext_scores, "External Validation Set")

if __name__ == "__main__":
    analyze_risk_strata()
