# MICROSOFT AZURE ML
## Benefits Eligibility Model
### Technical Specification & Model Documentation

**Version 1.0.0**  •  **March 28, 2026**  •  **HR Analytics Team**

---

## 1. Project Overview

This document describes all files required to deploy the Benefits Eligibility Classifier to Microsoft Azure ML. The model predicts whether an employee qualifies for company benefits based on their employment profile.

| Attribute | Value |
|---|---|
| Use Case | Employee benefits eligibility classification |
| Algorithm | Gradient Boosting Classifier (scikit-learn) |
| Owner | HR Analytics Team |
| Version | 1.0.0 |
| Risk Level | Medium — affects employee compensation |
| Training Date | 2026-03-28 |
| Output Classes | `not_eligible` \| `eligible` |

## 2. Repository & Artifact Structure

All files must be arranged exactly as shown. The `model/` directory is registered as an Azure ML Model asset.

```text
benefits-model-azure/
├── model/
│   ├── model.pkl               ← trained GradientBoostingClassifier
│   ├── preprocessor.joblib     ← fitted ColumnTransformer
│   └── label_map.json          ← {"0": "not_eligible", "1": "eligible"}
├── code/
│   ├── score.py                ← Azure ML entry-point script
│   ├── requirements.txt        ← runtime Python dependencies
│   └── config.yaml             ← (optional) deployment config
├── metadata/
│   └── model_metadata.json     ← governance & audit fields
├── test/
│   ├── sample_request.json     ← validation payload
│   └── expected_response.json  ← expected inference output
└── README.md
```

## 3. Artifact 1 — Model Bundle (MANDATORY)

### 3.1 Registered model contents

1. `model.pkl` — serialised `GradientBoostingClassifier`
2. `preprocessor.joblib` — fitted `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
3. `label_map.json` — integer-to-class-name mapping

### 3.2 Feature Schema

| Feature | Type | Description / Allowed Values |
|---|---|---|
| age | int | Employee age in years (22–65) |
| employment_type | string | `full_time` \| `part_time` \| `contractor` |
| years_of_service | int | Completed years at the company (0–40) |
| weekly_hours | int | Contracted weekly hours (10–55) |
| department | string | `engineering` \| `sales` \| `hr` \| `finance` \| `operations` |
| salary_band | string | `band_1` \| `band_2` \| `band_3` \| `band_4` \| `band_5` |
| has_dependents | int | Binary flag — `0 = no dependents`, `1 = has dependents` |

### 3.3 label_map.json

```json
{
 "0": "not_eligible",
 "1": "eligible"
}
```

## 4. Artifact 2 — Inference Script (MANDATORY)

**File:** `code/score.py` — defines the two Azure ML entry-point functions `init()` and `run()`.

### 4.1 init()

Called once at container startup. Reads `AZUREML_MODEL_DIR`, loads `model.pkl`, `preprocessor.joblib`, and `label_map.json` into module-level globals.

```python
def init():
    global model, preprocessor, label_map
    model_dir = os.getenv('AZUREML_MODEL_DIR', '.')
    model        = joblib.load(os.path.join(model_dir, 'model.pkl'))
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        label_map = json.load(f)
```

### 4.2 run(raw_data)

Called for every inference request. Accepts a single JSON dict or a list of dicts (batch). Also handles the Azure ML managed endpoint envelope `{"data": <payload>}`. Returns a JSON string.

```python
def run(raw_data):
    payload = json.loads(raw_data)
    if isinstance(payload, dict) and 'data' in payload:
        payload = payload['data']
    if isinstance(payload, dict): payload = [payload]
    df = pd.DataFrame(payload)[EXPECTED_FEATURES]
    X           = preprocessor.transform(df)
    raw_preds   = model.predict(X).tolist()
    class_proba = model.predict_proba(X).tolist()
    labeled     = [label_map[str(p)] for p in raw_preds]
    return json.dumps({'predictions': labeled, 'probabilities': class_proba})
```

## 5. Artifact 3 — Requirements File (MANDATORY)

**File:** `code/requirements.txt` — installed by Azure ML at environment build time.

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
azureml-defaults
```

## 6. Artifact 4 — Model Metadata

**File:** `metadata/model_metadata.json` — used for Azure ML model registry, governance, and auditing.

| Field | Value for This Model |
|---|---|
| model_version | 1.0.0 |
| owner | HR Analytics Team |
| use_case | Employee benefits eligibility classification |
| risk_level | medium |
| protected_attributes | age, has_dependents |
| training_date | 2026-03-28 |
| algorithm | GradientBoostingClassifier |
| training_samples | 1,600 |
| test_samples | 400 |

```json
{
  "model_version": "1.0.0",
  "owner": "HR Analytics Team",
  "use_case": "Employee benefits eligibility classification",
  "risk_level": "medium",
  "protected_attributes": ["age", "has_dependents"],
  "training_date": "2026-03-28",
  "algorithm": "GradientBoostingClassifier",
  "hyperparameters": {
    "n_estimators": 200, "learning_rate": 0.05,
    "max_depth": 4, "subsample": 0.8
  }
}
```

## 7. Artifact 5 — Sample Payloads (Validation)

### 7.1 test/sample_request.json

```json
{
  "data": [
    {
      "age": 34, "employment_type": "full_time",
      "years_of_service": 5, "weekly_hours": 40,
      "department": "engineering", "salary_band": "band_3",
      "has_dependents": 1
    },
    {
      "age": 27, "employment_type": "part_time",
      "years_of_service": 0, "weekly_hours": 20,
      "department": "sales", "salary_band": "band_1",
      "has_dependents": 0
    }
  ]
}
```

### 7.2 test/expected_response.json

```json
{
  "predictions": ["eligible", "not_eligible"],
  "probabilities": [[0.08, 0.92], [0.87, 0.13]],
  "label_order": ["not_eligible", "eligible"]
}
```

## 8. Deployment Notes

1. Register the `model/` directory as an Azure ML Model asset via the SDK or CLI.
2. Create an Azure ML Environment referencing `requirements.txt` and a base curated image (e.g. `AzureML-sklearn-1.0-ubuntu20.04-py38-cpu`).
3. Create a `ManagedOnlineEndpoint` and `ManagedOnlineDeployment` pointing `score.py` as `entry_point`.
4. Invoke the endpoint with `Content-Type: application/json`; the `run()` wrapper handles bare and enveloped payloads.
5. Monitor `protected_attributes` (`age`, `has_dependents`) for bias drift using Azure ML Responsible AI dashboard.

---

End of Document
