# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "train_split.csv")
EXTERNAL_FILE = os.path.join(BASE_DIR, "External Verification00.csv")

TARGET_COL = "Result"

def load_data(csv_path):
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    return df

def add_ni_index(df):
    df_out = df.copy()
    eps = 1e-6
    if 'ALB0' in df_out.columns and 'CRP0' in df_out.columns and 'ESR0' in df_out.columns:
        df_out['NI_Index'] = df_out['ALB0'] / (df_out['CRP0'] + df_out['ESR0'] + eps)
    return df_out

def get_metrics(y_true, y_prob):
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AP": average_precision_score(y_true, y_prob)
    }

def main():
    print("Loading data...")
    df_train = load_data(TRAIN_FILE)
    df_ext = load_data(EXTERNAL_FILE)
    
    if df_train is None or df_ext is None:
        print("[ERROR] Could not load data files.")
        return

    # Add NI_Index
    df_train = add_ni_index(df_train)
    df_ext = add_ni_index(df_ext)

    y_train = df_train[TARGET_COL]
    y_ext = df_ext[TARGET_COL]

    # Define the 5 levels of clinical models
    models_features = {
        "Level 0: CRP0 + ESR0": ['CRP0', 'ESR0'],
        "Level 1: ALB0 alone": ['ALB0'],
        "Level 2: CRP0 + ALB0": ['CRP0', 'ALB0'],
        "Level 3: CRP0 + ESR0 + ALB0": ['CRP0', 'ESR0', 'ALB0'],
        "Level 4: NI_Index alone": ['NI_Index']
    }

    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision'
    }

    results = []

    for name, features in models_features.items():
        print(f"\nEvaluating {name} with features: {features}...")
        
        X_train = df_train[features]
        X_ext = df_ext[features]

        # Use Logistic Regression with scaling and imputation
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]
        pipeline = Pipeline(steps)

        # 1. Internal CV (10-fold 10-repeats)
        print("  -> Running Internal CV (10x10 RepeatedStratifiedKFold)...")
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
        cv_res = cross_validate(pipeline, X_train, y_train, cv=rskf, scoring=scoring, n_jobs=1)
        
        model_results = {
            "Model": name,
            "Features": ", ".join(features),
            "CV AUC": f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}",
            "CV AP": f"{cv_res['test_average_precision'].mean():.4f} ± {cv_res['test_average_precision'].std():.4f}"
        }

        # 2. Fit on Full Train and Evaluate on External
        pipeline.fit(X_train, y_train)

        print("  -> Evaluating on External Test...")
        y_ext_prob = pipeline.predict_proba(X_ext)[:, 1]
        ext_m = get_metrics(y_ext, y_ext_prob)
        
        model_results["External AUC"] = f"{ext_m['AUC']:.4f}"
        model_results["External AP"] = f"{ext_m['AP']:.4f}"
            
        results.append(model_results)

    # Output results as Markdown table
    print("\n\n### Clinical Models Performance Comparison")
    print("| Level | Model Name | Features | CV AUC | External AUC | CV AP | External AP |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, res in enumerate(results):
        level = res['Model'].split(": ")[0]
        name = res['Model'].split(": ")[1]
        features = res['Features']
        cv_auc = res['CV AUC']
        ext_auc = res['External AUC']
        cv_ap = res['CV AP']
        ext_ap = res['External AP']
        print(f"| {level} | {name} | {features} | {cv_auc} | {ext_auc} | {cv_ap} | {ext_ap} |")

if __name__ == '__main__':
    main()
