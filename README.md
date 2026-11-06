# Microsoft Azure ML — Benefits Eligibility Model

## 1. Project Overview

**Project Name:** Azure-Deployment (Benefits Eligibility Model)

### Description

This project packages and deploys a Benefits Eligibility Classifier to Microsoft Azure ML. The model predicts whether an employee qualifies for company benefits (`eligible` / `not_eligible`) based on 7 employment-profile features, using a scikit-learn `GradientBoostingClassifier`. It includes the Azure ML scoring script and governance metadata for the deployment.

## 2. Core Features

- Gradient Boosting classifier trained on 7 employment features (age, employment type, years of service, weekly hours, department, salary band, dependents) to predict benefits eligibility.
- Azure ML-compliant `azure_score.py` implementing the required `init()` / `run()` entry points.
- `run()` accepts a bare JSON record/batch or the managed-endpoint envelope `{"data": <payload>}`, and returns labeled predictions with class probabilities.
- Governance metadata tracking owner, risk level, protected attributes (`age`, `has_dependents`), training date, and hyperparameters, intended for monitoring via the Azure ML Responsible AI dashboard.

## 3. Tech Stack

| Component | Technology |
|---|---|
| Model | scikit-learn `GradientBoostingClassifier`, XGBoost |
| Data handling | pandas, numpy, joblib |
| Platform | Azure Machine Learning (`azureml-defaults`, Managed Online Endpoints) |

## 4. Project Structure

```
Azure-Deployment/
├── azure_score.py            # Azure ML entry-point script (init/run)
├── doc/                      # architecture diagram
├── *_guide.md                 # deployment/runbook guides (ML, NN, SDOH variants)
└── LICENSE
```

## 5. Prerequisites

- An Azure subscription with an Azure ML workspace
- A trained model bundle (`model.pkl`, `preprocessor.joblib`, `label_map.json`) — this repository contains the scoring script only, not a training script; see the guide documents below for the training walkthrough
- `azureml-defaults`, numpy, pandas, scikit-learn, xgboost, joblib on the deployment environment image

## 6. How to Run

### Train

This repository does not include a training script. Produce `model.pkl`, `preprocessor.joblib`, and `label_map.json` using your own pipeline (see `benefits_ml_azure_deployment_guide.md` / `benefits_nn_azure_guide.md` for the full training walkthrough).

### Deploy

1. Register the trained `model/` artifacts as an Azure ML Model asset via the SDK or CLI.
2. Create an Azure ML Environment from a curated base image (e.g. `AzureML-sklearn-1.0-ubuntu20.04-py38-cpu`) plus the required packages (numpy, pandas, scikit-learn, xgboost, joblib, azureml-defaults).
3. Create a `ManagedOnlineEndpoint` and `ManagedOnlineDeployment` with `entry_point` set to `azure_score.py`.
4. Invoke the endpoint with `Content-Type: application/json` — `run()` handles both bare and `{"data": ...}`-enveloped payloads.
5. Monitor `age` and `has_dependents` (protected attributes) for bias drift using the Azure ML Responsible AI dashboard.

### Feature schema

| Feature | Type | Allowed values |
| :--- | :--- | :--- |
| age | int | 22-65 |
| employment_type | string | `full_time`, `part_time`, `contractor` |
| years_of_service | int | 0-40 |
| weekly_hours | int | 10-55 |
| department | string | `engineering`, `sales`, `hr`, `finance`, `operations` |
| salary_band | string | `band_1`...`band_5` |
| has_dependents | int | 0 or 1 |

## 7. Known Limitations

- No training script or test payloads are included in this repository (unlike the AWS/Vertex/Watson counterparts) — it covers scoring/deployment only.
- Bias/drift monitoring on protected attributes is a manual Responsible AI dashboard setup step, not automated in this codebase.

## 8. License

MIT License (see `LICENSE`).
