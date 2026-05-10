# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import json
import os
import sys

# Try to use scienceplots for publication-quality figures
import matplotlib.pyplot as plt
import seaborn as sns
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
TRAIN_FILE = "processed_with_features_LEFT_JOIN_all_rows33.csv"
SELECTED_FEATURES_FILE = "selected_features.json"

# Subgroup Columns
SUBGROUP_COLS = ['Gender', 'Age20s', 'Age30s', 'Age40s']

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

def load_data_with_subgroups(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found.")
    
    # Check for subgroup columns
    for col in SUBGROUP_COLS:
        if col not in df.columns:
            raise ValueError(f"Subgroup column '{col}' not found in data.")
            
    # Keep RAW_FEATURES, TARGET, and SUBGROUP_COLS
    cols_to_keep = RAW_FEATURES + [TARGET_COL] + SUBGROUP_COLS
    # Ensure no duplicates
    cols_to_keep = list(set(cols_to_keep))
    
    df = df[cols_to_keep]
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

def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    }

def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    # f1s has length thresholds + 1
    if len(thresholds) == 0: return 0.5
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx]

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=None):
        self.selected_features = selected_features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.selected_features is None:
            return X
        # Ensure only valid columns are selected
        valid_feats = [f for f in self.selected_features if f in X.columns]
        if len(valid_feats) < len(self.selected_features):
             print(f"Warning: {len(self.selected_features) - len(valid_feats)} features missing.")
        return X[valid_feats]
    def get_feature_names_out(self, input_features=None):
        return self.selected_features

def calculate_auc_ci(y_true, y_prob, n_bootstraps=1000, alpha=0.95):
    """Calculates AUC and 95% CI using bootstrapping and empirical P-value (H0: AUC=0.5)."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan
        
    original_auc = roc_auc_score(y_true, y_prob)
    bootstrapped_scores = []
    
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    if len(bootstrapped_scores) == 0:
        return original_auc, np.nan, np.nan, np.nan
        
    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, ((1.0 - alpha) / 2.0) * 100)
    upper_bound = np.percentile(sorted_scores, (alpha + (1.0 - alpha) / 2.0) * 100)
    
    # Empirical P-value for one-sided test (AUC > 0.5)
    n_worse = np.sum(np.array(bootstrapped_scores) <= 0.5)
    p_value = (n_worse + 1) / (len(bootstrapped_scores) + 1)
    
    return original_auc, lower_bound, upper_bound, p_value

def plot_forest_plot(results, overall_auc, output_path="subgroup_forest_plot.png"):
    """
    Draws a table-forest plot hybrid for subgroup analysis (Refined).
    results: List of dicts with keys 'Label', 'Mean', 'Lower', 'Upper', 'N', 'P_Value'
    overall_auc: The overall mean AUC to plot as the reference blue dashed line.
    """
    # Create DataFrame for plotting
    df_plot = pd.DataFrame(results)
    
    # Sort by Mean AUC (optional, or keep original order)
    # Keeping original order as per request usually (Gender then Age)
    # But reverse order for plotting so top item is at top
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)
    
    # Setup Figure and Grid
    # Height: rows + padding + legend + header + title
    row_height = 0.8
    fig_height = len(df_plot) * row_height + 2.0 
    fig = plt.figure(figsize=(10, fig_height)) 
    
    # 4 Columns: Subgroup, N, AUC(text), Plot
    # Ratios adjusted to exclude F1 column
    # [Subgroup, N, AUC, Plot]
    # Ratios: 1.5 : 0.8 : 1.5 : 2.5
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 0.8, 1.5, 2.5], wspace=0.0)
    
    ax_label = fig.add_subplot(gs[0])
    ax_n = fig.add_subplot(gs[1], sharey=ax_label)
    ax_auc = fig.add_subplot(gs[2], sharey=ax_label)
    ax_plot = fig.add_subplot(gs[3], sharey=ax_label)
    
    axes = [ax_label, ax_n, ax_auc, ax_plot]
    
    # --- Shared Y-axis setup ---
    ax_label.set_ylim(-0.5, len(df_plot) + 0.5)
    
    # Turn off axes for Text panels
    for ax in axes:
        ax.axis('off')
        
    # --- Headers ---
    header_y = len(df_plot)
    ax_label.text(0, header_y, "Subgroup", weight='bold', ha='left', va='center', fontsize=12)
    ax_n.text(0.5, header_y, "Sample Size", weight='bold', ha='center', va='center', fontsize=12)
    ax_auc.text(0.5, header_y, "AUC (95% CI)", weight='bold', ha='center', va='center', fontsize=12)
    
    # Draw Separator Line under header across all axes
    # We can draw line on each axis at y=header_y - 0.5
    for ax in axes:
        ax.axhline(y=header_y - 0.5, color='black', linewidth=1.5)

    # --- Draw Rows ---
    for i, row in df_plot.iterrows():
        # Alternating background - Darker gray
        if i % 2 == 1:
            color = '#e0e0e0' # Darker than #f0f0f0
            for ax in axes:
                ax.axhspan(i - 0.5, i + 0.5, color=color, zorder=-1)
            
        # Text Data
        label = row['Label']
        n_val = f"{row['N']}"
        mean = row['Mean']
        lower = row['Lower']
        upper = row['Upper']
        ci_text = f"{mean:.3f} ({lower:.3f}-{upper:.3f})"
        
        # Draw Text
        ax_label.text(0, i, label, ha='left', va='center', fontsize=11, weight='bold')
        ax_n.text(0.5, i, n_val, ha='center', va='center', fontsize=11, style='italic')
        ax_auc.text(0.5, i, ci_text, ha='center', va='center', fontsize=11)

    # --- Forest Plot Panel ---
    for i, row in df_plot.iterrows():
        mean = row['Mean']
        lower = row['Lower']
        upper = row['Upper']
        
        # Error Bar
        ax_plot.hlines(i, lower, upper, color='black', linewidth=1.5)
        # Point
        ax_plot.plot(mean, i, 's', color='black', markersize=6)
        
    # Reference Lines
    ax_plot.axvline(x=0.5, color='gray', linestyle='--', linewidth=1) # Random
    ax_plot.axvline(x=overall_auc, color='blue', linestyle='--', linewidth=1, label=f'Overall AUC ({overall_auc:.3f})') # Target
    
    # Ticks and Labels
    ax_plot.set_xlabel('AUC Score')
    ax_plot.set_yticks([]) 
    
    # Set x-limits
    ax_plot.set_xlim(0.4, 1.0)
    ax_plot.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Remove borders
    ax_plot.spines['left'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['top'].set_visible(False)
    
    # Title
    fig.suptitle("Internal Validation Subgroup Analysis", fontsize=16, weight='bold', y=0.98)
    
    # Bottom Annotation
    
    plt.tight_layout()
    # Adjust layout manually to minimize gap if needed, but wspace=0 should handle it.
    plt.subplots_adjust(top=0.90, wspace=0.0)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Table-Forest plot saved to {output_path}")

def run_subgroup_analysis():
    print("Loading data...")
    df_all = load_data_with_subgroups(TRAIN_FILE)
    
    # Load Selected Features
    if not os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[ERROR] {SELECTED_FEATURES_FILE} not found.")
        return
    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected_features = json.load(f)["selected_features"]
    print(f"Loaded {len(selected_features)} selected features.")
    
    # Prepare Data for Modeling
    X = df_all[RAW_FEATURES]
    y = df_all[TARGET_COL]
    
    # Prepare Pipeline
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
    
    print("Preprocessing, Engineering Features, and SELECTING Features...")
    # fit_transform will now return the selected features only
    X_final_np = pre_pipeline.fit_transform(X, y)
    
    # Reconstruct DataFrame (Pipeline outputs numpy array usually, but let's check)
    # The last step 'selector' returns a DataFrame if input is DataFrame? 
    # Actually Scikit-learn Pipeline often converts to numpy if steps do.
    # But FeatureSelector uses pandas indexing, so it might return DataFrame if input was DataFrame.
    # However, intermediate steps might have converted to numpy.
    # Let's force reconstruction if needed.
    
    # To be safe and explicit:
    # We can rely on 'X_final_np' shape check.
    
    X_final = pd.DataFrame(X_final_np, columns=selected_features, index=X.index)
    print(f"Final Feature Matrix Shape: {X_final.shape}")
    
    # STRICT CHECK
    if X_final.shape[1] != 13:
        print(f"[CRITICAL WARNING] Expected 13 features, but got {X_final.shape[1]}. Check selection logic!")
    else:
        print("[SUCCESS] Verified 13 features used for training.")
    
    # Run 10x10 Cross-Validation to get OOF Predictions (Ensemble of 10 repeats)
    print("Running 10x10 Cross-Validation (OOF Predictions)...")
    
    # Force CPU to avoid GPU hang/overhead
    cpu_params = BEST_PARAMS.copy()
    if 'device' in cpu_params: del cpu_params['device']
    if 'tree_method' in cpu_params: del cpu_params['tree_method']
    clf = XGBClassifier(**cpu_params, tree_method="hist", random_state=42, n_jobs=1)
    
    n_repeats = 10
    n_splits = 10
    y_oof_probs_sum = np.zeros(len(y))
    
    for i in range(n_repeats):
        # Use different random state for each repeat
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + i)
        y_oof_prob_i = cross_val_predict(clf, X_final, y, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
        y_oof_probs_sum += y_oof_prob_i
        print(f"Repeat {i+1}/{n_repeats} completed.")
        
    y_oof_prob = y_oof_probs_sum / n_repeats
    
    # Calculate Optimal Threshold on Full Dataset
    best_thr = 0.53
    print(f"Fixed Optimal Threshold (Full Dataset): {best_thr:.4f}")
    
    # Overall Performance
    overall_metrics = calculate_metrics(y, y_oof_prob, best_thr)
    print(f"Overall Performance: AUC={overall_metrics['AUC']:.4f}")
    
    # Create Analysis DataFrame
    analysis_df = df_all[SUBGROUP_COLS + [TARGET_COL]].copy()
    analysis_df['Prob'] = y_oof_prob
    analysis_df['Pred'] = (y_oof_prob >= best_thr).astype(int)
    
    print("\n=== Subgroup Analysis Results ===")
    
    forest_data = []

    # 1. Gender Analysis
    print("\n--- Gender Subgroups ---")
    genders = {0: 'Female', 1: 'Male'}
    for g_val, g_name in genders.items():
        sub_df = analysis_df[analysis_df['Gender'] == g_val]
        if len(sub_df) == 0:
            print(f"{g_name}: No samples")
            continue
        
        metrics = calculate_metrics(sub_df[TARGET_COL], sub_df['Prob'], best_thr)
        print(f"{g_name} (n={len(sub_df)}): AUC={metrics['AUC']:.4f}")
        
        # Calculate CI for Forest Plot
        auc, lower, upper, p_val = calculate_auc_ci(sub_df[TARGET_COL], sub_df['Prob'])
        forest_data.append({'Label': g_name, 'Mean': auc, 'Lower': lower, 'Upper': upper, 'N': len(sub_df), 'P_Value': p_val})
        
    # 2. Age Analysis
    print("\n--- Age Subgroups ---")
    # Age groups: Age20s, Age30s, Age40s
    age_cols = ['Age20s', 'Age30s', 'Age40s']
    for col in age_cols:
        # Filter: Only include rows where this column is 1
        sub_df = analysis_df[analysis_df[col] == 1]
        
        # Determine label (e.g., "Age 20s")
        label = col.replace("Age", "Age ").replace("s", "s")
        
        if len(sub_df) == 0:
            print(f"{label}: No samples")
            continue
            
        metrics = calculate_metrics(sub_df[TARGET_COL], sub_df['Prob'], best_thr)
        print(f"{label} (n={len(sub_df)}): AUC={metrics['AUC']:.4f}")
        
        # Calculate CI for Forest Plot
        auc, lower, upper, p_val = calculate_auc_ci(sub_df[TARGET_COL], sub_df['Prob'])
        forest_data.append({'Label': label, 'Mean': auc, 'Lower': lower, 'Upper': upper, 'N': len(sub_df), 'P_Value': p_val})
        
    # Check "Other" (all 0) count just for info
    mask_others = (analysis_df[age_cols].sum(axis=1) == 0)
    n_others = mask_others.sum()
    print(f"\n(Note: {n_others} samples excluded from Age analysis as they don't fall into 20s/30s/40s)")
    
    print("\n=== Bias Check (Bootstrap Subsampling 1000x) ===")
    
    # 1. Gender Bias Check
    print("\n--- Gender Bias Check (Male Subsampling) ---")
    male_df = analysis_df[analysis_df['Gender'] == 1]
    female_df = analysis_df[analysis_df['Gender'] == 0]
    
    n_female = len(female_df)
    n_male = len(male_df)
    print(f"Original: Male n={n_male}, Female n={n_female}")
    
    if n_male > n_female:
        print(f"Subsampling Male to n={n_female} (1000 iterations)...")
        aucs = []
        for i in range(1000):
            # Sample without replacement to mimic 'having only that many samples'
            sample_male = male_df.sample(n=n_female, replace=False, random_state=i)
            auc = roc_auc_score(sample_male[TARGET_COL], sample_male['Prob']) if len(sample_male[TARGET_COL].unique()) > 1 else np.nan
            if not np.isnan(auc):
                aucs.append(auc)
        
        aucs = np.array(aucs)
        mean_auc = np.mean(aucs)
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)
        
        print(f"Male Subsampled AUC: Mean={mean_auc:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        female_auc = roc_auc_score(female_df[TARGET_COL], female_df['Prob'])
        print(f"Female Original AUC: {female_auc:.4f}")
        
        if female_auc < ci_lower or female_auc > ci_upper:
            print("Result: Significant difference (Female AUC outside Male 95% CI).")
        else:
            print("Result: No significant difference (Female AUC inside Male 95% CI).")
    
    # 2. Age Bias Check
    print("\n--- Age Bias Check (Age Subsampling) ---")
    # Identify sample sizes
    age_dfs = {}
    for col in age_cols:
        label = col.replace("Age", "Age ").replace("s", "s")
        sub_df = analysis_df[analysis_df[col] == 1]
        age_dfs[label] = sub_df
        
    sizes = {k: len(v) for k, v in age_dfs.items()}
    min_size = min(sizes.values())
    target_n = min_size
    print(f"Subsampling larger groups to n={target_n} (Size of smallest group)...")
    
    for label, df_age in age_dfs.items():
        original_auc = roc_auc_score(df_age[TARGET_COL], df_age['Prob'])
        if len(df_age) == target_n:
            print(f"{label} (Baseline, n={target_n}): AUC={original_auc:.4f}")
        else:
            aucs = []
            for i in range(1000):
                sample_age = df_age.sample(n=target_n, replace=False, random_state=i)
                # Check if sample has both classes
                if len(sample_age[TARGET_COL].unique()) > 1:
                    auc = roc_auc_score(sample_age[TARGET_COL], sample_age['Prob'])
                    aucs.append(auc)
            
            aucs = np.array(aucs)
            mean_auc = np.mean(aucs)
            ci_lower = np.percentile(aucs, 2.5)
            ci_upper = np.percentile(aucs, 97.5)
            print(f"{label} (Subsampled n={target_n}): Mean AUC={mean_auc:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    # Generate Forest Plot
    plot_forest_plot(forest_data, overall_auc=overall_metrics['AUC'])

if __name__ == "__main__":
    run_subgroup_analysis()
