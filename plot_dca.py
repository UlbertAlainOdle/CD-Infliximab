import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from scipy.ndimage import gaussian_filter1d
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import json
import os
import warnings

warnings.filterwarnings("ignore")

# --- Academic Plotting Style ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

COLOR_PROPOSED = '#D55E00'  # Vermilion
COLOR_BASELINE = '#7F7F7F'  # Gray
COLOR_TREAT_ALL = '#4682B4'
COLOR_TREAT_NONE = '#000000'

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FEATURES = [
    'Perianal', 'M0', 'Lyn0', 'PLT0', 'HB0', 'ESR0', 'CRP0', 
    'GGT0', 'IBIL0', 'DBIL0', 'ALB0', 'Ca0', 'Cr0', 'UA0'
]
TARGET_COL = "Result"
TRAIN_FILE = os.path.join(BASE_DIR, "processed_with_features_LEFT_JOIN_all_rows.csv")
EXTERNAL_FILE = os.path.join(BASE_DIR, "External Verification00.csv")
SELECTED_FEATURES_FILE = os.path.join(BASE_DIR, "selected_features.json")

# --- Best Params ---
BEST_PARAMS_PROPOSED = {'n_estimators': 137, 'max_depth': 10, 'learning_rate': 0.017444824475985048, 'subsample': 0.9997391245814459, 'colsample_bytree': 0.9415743112958483, 'gamma': 1.7483463681615965, 'min_child_weight': 2, 'reg_alpha': 0.04845238191904519, 'reg_lambda': 0.0121275899499253, 'scale_pos_weight': 4.045168009584749}


# Feature Engineering Classes (must match exactly to load models properly, though we will also recreate OOF)
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

# --- Data Loading ---
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    X = df[RAW_FEATURES]
    y = df[TARGET_COL]
    return X, y

def calculate_net_benefit(y_true, y_prob, pt_arr):
    net_benefits = []
    n = len(y_true)
    for pt in pt_arr:
        if pt == 0:
            net_benefits.append(y_true.mean())
            continue
        elif pt == 1:
            net_benefits.append(0)
            continue
        
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def main():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    X_ext, y_ext = load_data(EXTERNAL_FILE)
    
    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected_features = json.load(f)["selected_features"]
    
    # Development Set Split
    X_train, _, y_train, _ = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Models
    print("Loading pre-trained models...")
    proposed_pipeline = joblib.load(os.path.join(BASE_DIR, "reproduced_pipeline.joblib"))
    proposed_model = joblib.load(os.path.join(BASE_DIR, "reproduced_best_model.joblib"))
    baseline_model = joblib.load(os.path.join(BASE_DIR, "baseline_best_model.joblib"))
    
    # --- 1. Generate OOF Predictions on Development Set (N=106) ---
    print("Generating OOF predictions on Development Set...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Proposed Model OOF
    # We must replicate the pipeline steps to get pure OOF
    oof_proposed_probs = np.zeros(len(y_train))
    oof_baseline_probs = np.zeros(len(y_train))
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Proposed OOF
        pre_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median').set_output(transform="pandas")),
            ('winsorizer', Winsorizer(limits=(0.01, 0.01))),
            ('engineer', UltimateFeatureEngineer()),
            ('var_thresh', VarianceThreshold(threshold=1e-4))
        ])
        
        X_tr_proc = pre_pipe.fit_transform(X_tr, y_tr)
        # Reconstruct DataFrame
        eng_step = pre_pipe.named_steps['engineer']
        var_thresh_step = pre_pipe.named_steps['var_thresh']
        final_feats = np.array(eng_step.get_feature_names_out())[var_thresh_step.get_support()]
        X_tr_proc = pd.DataFrame(X_tr_proc, columns=final_feats)[selected_features]
        
        X_val_proc = pre_pipe.transform(X_val)
        X_val_proc = pd.DataFrame(X_val_proc, columns=final_feats)[selected_features]
        
        clf_prop = XGBClassifier(**BEST_PARAMS_PROPOSED, random_state=42, n_jobs=1)
        clf_prop.fit(X_tr_proc, y_tr)
        oof_proposed_probs[val_idx] = clf_prop.predict_proba(X_val_proc)[:, 1]

    # For Baseline OOF, instantiate a fresh model using the best params
    with open(os.path.join(BASE_DIR, "baseline_best_params.json"), "r") as f:
        baseline_params = json.load(f)
    
    base_clf = XGBClassifier(**baseline_params, random_state=42, n_jobs=1)
    base_full_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', base_clf)
    ])
    oof_baseline_probs = cross_val_predict(base_full_pipe, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    # --- 2. Generate External Predictions (N=60) ---
    print("Generating External Validation predictions...")
    X_ext_proc_np = proposed_pipeline.transform(X_ext)
    # The saved pipeline was fit on train set. We need the final_feats from it.
    eng_step_saved = proposed_pipeline.named_steps['engineer']
    var_thresh_step_saved = proposed_pipeline.named_steps['var_thresh']
    final_feats_saved = np.array(eng_step_saved.get_feature_names_out())[var_thresh_step_saved.get_support()]
    
    X_ext_proc = pd.DataFrame(X_ext_proc_np, columns=final_feats_saved)[selected_features]
    ext_proposed_probs = proposed_model.predict_proba(X_ext_proc)[:, 1]
    
    # Baseline External
    imputer_ext = SimpleImputer(strategy='median')
    X_train_imputed = imputer_ext.fit_transform(X_train)
    X_ext_imputed = imputer_ext.transform(X_ext)
    ext_baseline_probs = baseline_model.predict_proba(X_ext_imputed)[:, 1]
    
    # --- Plotting ---
    fig = plt.figure(figsize=(16, 7))
    
    # DCA Setup
    pt_arr = np.linspace(0.01, 0.99, 100)
    
    # A: DCA (Internal)
    ax1 = plt.subplot(1, 2, 1)
    nb_proposed_int = calculate_net_benefit(y_train, oof_proposed_probs, pt_arr)
    # Apply Gaussian smoothing to internal DCA
    nb_proposed_int_smooth = gaussian_filter1d(nb_proposed_int, sigma=2)
    
    nb_all_int = calculate_net_benefit(y_train, np.ones(len(y_train)), pt_arr)
    
    ax1.plot(pt_arr, nb_proposed_int_smooth, color=COLOR_PROPOSED, label="Proposed Model (OOF)", linestyle='-', linewidth=2.4)
    ax1.plot(pt_arr, nb_all_int, color=COLOR_TREAT_ALL, label="Treat All", linestyle='-.', linewidth=2)
    ax1.plot(pt_arr, np.zeros_like(pt_arr), color=COLOR_TREAT_NONE, label="Treat None", linestyle='-', linewidth=2)
    
    ax1.set_xlim(0, 0.7)
    ax1.set_ylim(-0.1, 0.6)
    ax1.set_xlabel("Threshold Probability", fontweight='bold')
    ax1.set_ylabel("Net Benefit", fontweight='bold')
    ax1.set_title("Decision Curve Analysis (Internal)", fontweight='bold')
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # B: DCA (External)
    ax2 = plt.subplot(1, 2, 2)
    nb_proposed_ext = calculate_net_benefit(y_ext, ext_proposed_probs, pt_arr)
    nb_baseline_ext = calculate_net_benefit(y_ext, ext_baseline_probs, pt_arr)
    
    # Apply Gaussian smoothing to external DCA
    nb_proposed_ext_smooth = gaussian_filter1d(nb_proposed_ext, sigma=2)
    nb_baseline_ext_smooth = gaussian_filter1d(nb_baseline_ext, sigma=2)
    
    nb_all_ext = calculate_net_benefit(y_ext, np.ones(len(y_ext)), pt_arr)
    
    ax2.plot(pt_arr, nb_proposed_ext_smooth, color=COLOR_PROPOSED, label="Proposed Model", linestyle='-', linewidth=2.4)
    ax2.plot(pt_arr, nb_baseline_ext_smooth, color=COLOR_BASELINE, label="Raw XGBoost", linestyle=':', linewidth=2.0)
    ax2.plot(pt_arr, nb_all_ext, color=COLOR_TREAT_ALL, label="Treat All", linestyle='-.', linewidth=2)
    ax2.plot(pt_arr, np.zeros_like(pt_arr), color=COLOR_TREAT_NONE, label="Treat None", linestyle='-', linewidth=2)
    
    ax2.set_xlim(0, 0.8)
    ax2.set_ylim(-0.1, 0.8)
    ax2.set_xlabel("Threshold Probability", fontweight='bold')
    ax2.set_ylabel("Net Benefit", fontweight='bold')
    ax2.set_title("Decision Curve Analysis (External)", fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "dca_curves_only.png")
    save_path_pdf = os.path.join(BASE_DIR, "dca_curves_only.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"\nSaved DCA plots to {save_path} and {save_path_pdf}")
    
if __name__ == "__main__":
    main()