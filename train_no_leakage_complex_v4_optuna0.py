# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import optuna
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
SELECTED_FEATURES_FILE = "selected_features.json"
MAX_FEATURES_TO_KEEP = 15

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

def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    return neg / pos if pos > 0 else 1.0

def objective(trial, X, y, scale_pos_weight_base):
    # Optuna Parameters (Centered around specified best parameters)
    # Best params:
    # {'n_estimators': 133, 'max_depth': 10, 'learning_rate': 0.015116953223049432, 
    #  'subsample': 0.9772729578366293, 'colsample_bytree': 0.999397767953322, 
    #  'gamma': 1.4255464907790143, 'min_child_weight': 2, 'reg_alpha': 0.004379539885837058, 
    #  'reg_lambda': 0.007078275448431759, 'scale_pos_weight': 3.691776296383791}
    
    # We create a search space around these values (+/- ~20-30% or logical bounds)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'n_estimators': trial.suggest_int('n_estimators', 140, 205),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.014, 0.019, log=True),
        'subsample': trial.suggest_float('subsample', 0.955, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.935, 1.0),
        'gamma': trial.suggest_float('gamma', 1.72, 2.05),
        'min_child_weight': 2,
        'reg_alpha': trial.suggest_float('reg_alpha', 0.030, 0.055),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.008, 0.040),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 5.2),

        'random_state': 42,
        'n_jobs': 1,
        'verbosity': 0,
        **_gpu_params()
    }

    
    clf = XGBClassifier(**params)
    
    # UNIFIED VALIDATION: Use RepeatedStratifiedKFold (Same as final evaluation)
    # User requested Optuna validation logic matches Final validation logic (10-Fold 2-Repeats)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42) 
    
    scores = cross_validate(clf, X, y, cv=rskf, scoring=['f1', 'roc_auc'], n_jobs=1)
    
    mean_f1 = scores['test_f1'].mean()
    mean_auc = scores['test_roc_auc'].mean()
    
    trial.set_user_attr("auc", mean_auc)
    trial.set_user_attr("f1", mean_f1)
    
    # We optimize for AUC or F1? 
    # User wants to save BOTH best AUC and best F1 models.
    # Optuna optimizes one scalar. Let's optimize AUC as it's more robust for imbalance.
    # We will track F1 separately.
    
    print(f"[Trial {trial.number}] AUC: {mean_auc:.4f}, F1: {mean_f1:.4f}")
    
    return mean_auc

def run_feature_selection(X, y):
    if os.path.exists(SELECTED_FEATURES_FILE):
        print(f"[INFO] Loading cached selected features from {SELECTED_FEATURES_FILE}...")
        try:
            with open(SELECTED_FEATURES_FILE, "r") as f:
                data = json.load(f)
                selected_features = data["selected_features"]
            
            print("   - Re-building pre-processing pipeline...")
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
            pre_pipeline.fit(X, y)
            return selected_features, pre_pipeline
        except Exception as e:
            print(f"[WARNING] Failed to load cached features: {e}. Re-running selection.")

    print("Running RFECV for Feature Selection (GPU)...")
    
    imputer = SimpleImputer(strategy='median').set_output(transform="pandas")
    winsorizer = Winsorizer(limits=(0.01, 0.01))
    eng = UltimateFeatureEngineer()
    var_thresh = VarianceThreshold(threshold=1e-4)
    
    print("   - Pre-processing features...")
    pre_pipeline = Pipeline([
        ('imputer', imputer),
        ('winsorizer', winsorizer),
        ('engineer', eng),
        ('var_thresh', var_thresh)
    ])
    X_processed_np = pre_pipeline.fit_transform(X, y)
    
    # Correctly handle feature names with VarianceThreshold
    eng_step = pre_pipeline.named_steps['engineer']
    all_feats_before_var = eng_step.get_feature_names_out()
    
    var_thresh_step = pre_pipeline.named_steps['var_thresh']
    support_mask = var_thresh_step.get_support()
    
    final_feats = np.array(all_feats_before_var)[support_mask]
    
    X_processed = pd.DataFrame(X_processed_np, columns=final_feats, index=X.index)
    
    print(f"   - Processed Feature Count: {X_processed.shape[1]}")
    print("   - Starting RFECV...")
    estimator = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, 
        n_jobs=1, random_state=42, **_gpu_params()
    )
    
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(5),
        scoring='roc_auc', # Use AUC for selection stability
        min_features_to_select=10,
        n_jobs=1
    )
    
    selector.fit(X_processed, y)
    
    selected_mask = selector.support_
    selected_features = [f for f, s in zip(final_feats, selected_mask) if s]
    
    if len(selected_features) > MAX_FEATURES_TO_KEEP:
        print(f"   - Reducing features from {len(selected_features)} to {MAX_FEATURES_TO_KEEP} based on importance...")
        estimator.fit(X_processed[selected_features], y)
        imps = pd.Series(estimator.feature_importances_, index=selected_features)
        selected_features = imps.sort_values(ascending=False).head(MAX_FEATURES_TO_KEEP).index.tolist()
    
    try:
        with open(SELECTED_FEATURES_FILE, "w") as f:
            json.dump({"selected_features": selected_features}, f, indent=4)
        print(f"[INFO] Selected features saved to {SELECTED_FEATURES_FILE}")
    except Exception as e:
        print(f"[WARNING] Failed to save selected features: {e}")

    return selected_features, pre_pipeline

def evaluate_final_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "ap": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

def _best_threshold_for_f1(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0: return 0.5
    idx = np.argmax(f1s[:-1]) 
    return thresholds[idx]

def main():
    print("Loading data...")
    X_raw, y = load_data(TRAIN_FILE)
    
    # 1. Split Data or Load Existing Splits
    train_split_file = "train_split.csv"
    test_split_file = "test_split.csv"
    
    if os.path.exists(train_split_file) and os.path.exists(test_split_file):
        print(f"[WARNING] Existing splits found ({train_split_file}, {test_split_file}). Loading them instead of generating new random splits.")
        train_df = pd.read_csv(train_split_file, index_col=0)
        test_df = pd.read_csv(test_split_file, index_col=0)
        X_train = train_df[RAW_FEATURES]
        y_train = train_df[TARGET_COL]
        X_test = test_df[RAW_FEATURES]
        y_test = test_df[TARGET_COL]
    else:
        print("Generating new random train/test splits and saving to CSV...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=26, random_state=42, stratify=y
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        train_df.to_csv(train_split_file)
        test_df.to_csv(test_split_file)
        
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. Feature Selection
    selected_features, pre_pipeline = run_feature_selection(X_train, y_train)
    print(f"\nFinal Selected Features ({len(selected_features)}):")
    print(selected_features)
        
    # Transform Data
    X_train_processed_np = pre_pipeline.transform(X_train)
    
    eng = pre_pipeline.named_steps['engineer']
    all_feats_before_var = eng.get_feature_names_out()
    
    var_thresh = pre_pipeline.named_steps['var_thresh']
    support_mask = var_thresh.get_support()
    final_feats = np.array(all_feats_before_var)[support_mask]
    
    X_train_processed = pd.DataFrame(X_train_processed_np, columns=final_feats, index=X_train.index)
    X_train_sel = X_train_processed[selected_features]
    
    X_test_processed_np = pre_pipeline.transform(X_test)
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=final_feats, index=X_test.index)
    X_test_sel = X_test_processed[selected_features]
    
    # 3. Optuna Optimization
    print("\nStarting Optuna Hyperparameter Optimization (10-Fold 3-Repeats) - Focused Search...")
    scale_pos = compute_scale_pos_weight(y_train)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_sel, y_train, scale_pos), n_trials=150)
    
    # Find Best Trials
    best_trial_auc = study.best_trial
    best_trial_f1 = max(study.trials, key=lambda t: t.user_attrs.get("f1", 0))
    
    print("\n=== Optimization Results ===")
    print(f"Best AUC Trial: {best_trial_auc.number} | AUC: {best_trial_auc.value:.4f} | F1: {best_trial_auc.user_attrs['f1']:.4f}")
    print(f"Best F1 Trial : {best_trial_f1.number} | F1: {best_trial_f1.user_attrs['f1']:.4f} | AUC: {best_trial_f1.user_attrs['auc']:.4f}")
    
    # 4. Final Validation & Saving (Two Models)
    models_to_save = [
        ("best_auc_model", best_trial_auc.params),
        ("best_f1_model", best_trial_f1.params)
    ]
    
    for name, params in models_to_save:
        print(f"\nTraining & Validating {name} (10-Fold 10-Repeats)...")
        final_clf = XGBClassifier(**params, **_gpu_params())
        
        # Stability Check
        rskf_final = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
        cv_results = cross_validate(final_clf, X_train_sel, y_train, cv=rskf_final, 
                                  scoring=['f1', 'roc_auc', 'average_precision', 'accuracy'], n_jobs=1)
        
        print(f"   CV Results for {name}:")
        print(f"   - AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
        print(f"   - F1 : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
        
        # Final Fit & Holdout
        final_clf.fit(X_train_sel, y_train)
        
        # Threshold
        y_train_prob = final_clf.predict_proba(X_train_sel)[:, 1]
        thr = _best_threshold_for_f1(y_train, y_train_prob)
        print(f"   - Optimal Threshold: {thr:.4f}")
        
        metrics = evaluate_final_model(final_clf, X_test_sel, y_test, threshold=thr)
        print(f"   - Holdout AUC: {metrics['auc']:.4f}")
        print(f"   - Holdout F1 : {metrics['f1']:.4f}")
        
        # Save
        result_pkg = {
            "features": selected_features,
            "params": params,
            "threshold": float(thr),
            "metrics": metrics,
            "cv_results": {k: float(v.mean()) for k, v in cv_results.items()}
        }
        joblib.dump(result_pkg, f"{name}_package.pkl")
        joblib.dump(final_clf, f"{name}.joblib")

    joblib.dump(pre_pipeline, "preprocessing_pipeline.joblib")
    print("\n[INFO] All models and pipelines saved.")

if __name__ == "__main__":
    main()
