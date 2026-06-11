# CD-Infliximab Routine-Data Prediction Pipeline

This repository contains the analysis code and model artifacts for the study:

**Pretreatment Routine-Data Prediction of Mucosal Healing After Infliximab in Crohn’s Disease: A Multicenter Retrospective Study**

This project develops and externally validates an interpretable routine-data model for pretreatment prediction of week-52 mucosal healing after infliximab therapy in biologic-naïve patients with Crohn’s disease. The workflow includes robust preprocessing, clinical feature engineering, RFECV-based feature selection, cost-sensitive XGBoost modeling, internal holdout testing, independent external validation, decision-curve analysis, predicted-response stratification, and SHAP-based interpretability analysis.

## Data Availability

Patient-level clinical data are not included in this repository because of privacy, ethical, and institutional restrictions.

The data underlying the study are available from the corresponding author upon reasonable request and subject to appropriate ethical and institutional approval.

This repository provides the complete methodological workflow, analysis scripts, selected model artifacts, parameter files, and configuration files needed to reproduce the reported analyses when authorized data are available.

## Repository Contents

### Main model development and reproduction

* **`train_no_leakage_complex_v4_optuna0.py`**
  Main model-development script. It performs robust preprocessing, clinical feature engineering, RFECV-based feature selection, Optuna hyperparameter optimization, cost-sensitive XGBoost training, and locked model construction.

* **`reproduce_best_model0.py`**
  Reproduces the final locked model using fixed settings and saved artifacts. This is the main entry point for reproducing the reported final-model evaluation.

* **`selected_features.json`**
  Stores the final locked feature subset used by the final model.

* **`reproduced_best_model.joblib`**
  Saved final XGBoost model artifact.

* **`reproduced_pipeline.joblib`**
  Saved preprocessing and transformation pipeline used for deterministic inference.

### Baseline and feature-ablation analyses

* **`evaluate_clinical_models.py`**
  Evaluates simple clinical baseline models, including inflammatory-marker, albumin-related, and NI_Index-based models.

* **`evaluate_all_baselines.py`**
  Performs baseline screening across multiple machine-learning algorithms using raw clinical features. This script supports the raw-feature baseline model comparison.

* **`train_raw_model.py`**
  Trains and evaluates the XGBoost model using the original raw clinical variables only. 

* **`baseline_best_model.joblib`** and **`baseline_best_params.json`**
  Saved model and parameter files for the raw-feature baseline screening analysis.

* **`train_raw_plus_ni_index.py`**
  Trains and evaluates the raw-variable-plus-NI_Index model used in the feature-ablation comparison.

* **`raw_ni_model.joblib`** and **`raw_ni_params.json`**
  Saved model and parameter files for the raw-variable-plus-NI_Index configuration.

### Decision analysis, stratification, and threshold analysis

* **`plot_dca.py`**
  Generates decision-curve analysis results for internal and external evaluation.

* **`risk_stratification.py`**
  Performs predicted-response stratification analysis using the locked probability cutoff.

* **`analyze_threshold.py`**
  Performs F1-score sensitivity analysis across decision thresholds.

### Interpretability and supplementary analyses

* **`shap_analysis.py`**
  Performs SHAP-based model interpretability analysis, including global feature importance, dependence plots.

* **`plot_correlation_heatmap.py`**
  Generates the Pearson correlation heatmap of the final 13-feature subset.

* **`visualize_rfecv.py`**
  Generates the RFECV performance curve used to support the final compact feature subset.

* **`hard_sample_analysis.py`**
  Identifies and summarizes hard samples, including false-positive and false-negative patterns.

* **`plot_hard_sample.py`**
  Generates the hard-sample comparison plot used in the supplementary analysis.

## Relationship to Manuscript Outputs

| Manuscript output                                                                                         | Related scripts/files                                                                                 |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Table 2. Model performance across internal development, internal holdout, and external validation cohorts | `reproduce_best_model0.py`, `reproduced_best_model.joblib`, `reproduced_pipeline.joblib`              |
| Table 4. Impact of clinical feature engineering                                                           | `evaluate_clinical_models.py`, `train_raw_model.py`, `train_raw_plus_ni_index.py`, `reproduce_best_model0.py`|
| Figure 3. Decision curve analysis and external predicted-response stratification                          | `plot_dca.py`, `risk_stratification.py`                                                               |
| Figure 4. SHAP-based interpretability analysis                                                            | `shap_analysis.py`                                                                                    |
| Supplementary Table S1. Hyperparameter search space and final locked configuration                        | `train_no_leakage_complex_v4_optuna0.py`, `reproduce_best_model0.py`                                  |
| Supplementary Table S2. Baseline machine-learning model screening                                         | `evaluate_all_baselines.py`                                                                           |
| Supplementary Fig S1. Pearson correlation heatmap                                                         | `plot_correlation_heatmap.py`                                                                         |
| Supplementary Fig S2. RFECV performance curve                                                             | `visualize_rfecv.py`                                                                                  |
| Supplementary Fig S3. Threshold-sensitivity analysis                                                      | `analyze_threshold.py`                                                                                |
| Supplementary Fig S4. Internal predicted-response stratification                                          | `risk_stratification.py`                                                                              |
| Supplementary Fig S5. Hard-sample feature comparison                                                      | `hard_sample_analysis.py`, `plot_hard_sample.py`                                                      |

## Requirements

The analysis was implemented in Python. Required packages are listed in:

```bash
requirements.txt
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproduction Workflow

Because patient-level data are not publicly distributed, full numerical reproduction requires authorized access to the de-identified clinical datasets.

After placing the approved data files in the paths specified in the scripts, the recommended workflow is:

```bash
# 1. Reproduce the locked final model
python reproduce_best_model0.py

# 2. Evaluate baseline and feature-ablation models
python evaluate_clinical_models.py
python evaluate_all_baselines.py
python train_raw_model.py
python train_raw_plus_ni_index.py

# 3. Generate decision-curve and predicted-response stratification analyses
python plot_dca.py
python risk_stratification.py

# 4. Generate interpretability and supplementary analyses
python shap_analysis.py
python plot_correlation_heatmap.py
python visualize_rfecv.py
python analyze_threshold.py
python hard_sample_analysis.py
python plot_hard_sample.py
```

For complete retraining and hyperparameter optimization, run:

```bash
python train_no_leakage_complex_v4_optuna0.py
```

This full training script may require more time because it includes feature engineering, RFECV-based feature selection, and Optuna hyperparameter optimization.

## Notes on Reproducibility

* The final model uses locked preprocessing parameters, a locked 13-feature subset, locked hyperparameters, and a locked decision threshold.
* The independent external validation cohort is not used for feature selection, hyperparameter tuning, threshold selection, or preprocessing-parameter estimation.
* The saved `.joblib` and `.json` files are provided to support deterministic reuse of the reported model configuration when authorized input data are available.
* Small numerical differences may occur across environments because of package versions, random seeds, and hardware-dependent implementation details.
* Since the clinical datasets are not included, running the scripts without authorized data will require adapting the data-loading paths to the user’s local approved dataset.

## Privacy and Ethics

No identifiable patient-level data are stored in this repository. Any use of the model or scripts with clinical data should comply with local ethical approval, institutional data-governance requirements, and applicable privacy regulations.

## Clinical Use Disclaimer

This repository is intended for research reproducibility and methodological transparency. The model is not a standalone clinical decision-making tool and should not replace physician judgment, endoscopic reassessment, therapeutic drug monitoring, imaging evaluation, or other forms of clinical follow-up.

## License

This project is released under the terms specified in the `LICENSE` file.
