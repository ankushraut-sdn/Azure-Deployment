"""
Azure ML Inference Script — Benefits Eligibility Classifier
===========================================================
Azure ML entry-point script. Defines init() and run() as required
by the Azure ML online/batch endpoint contract.

Environment variable set by Azure ML at runtime:
    AZUREML_MODEL_DIR  →  directory containing registered model artifacts
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Module-level globals — populated once in init()
model        = None
preprocessor = None
label_map    = None

EXPECTED_FEATURES = [
    "age",
    "employment_type",       # "full_time" | "part_time" | "contractor"
    "years_of_service",
    "weekly_hours",
    "department",
    "salary_band",           # "band_1" … "band_5"
    "has_dependents",        # 0 or 1
]


# ---------------------------------------------------------------------------
# 1. init() — Load model artifacts at container startup
# ---------------------------------------------------------------------------

def init():
    """
    Called once when the scoring container starts.
    Loads model, preprocessor, and label map from AZUREML_MODEL_DIR.
    """
    global model, preprocessor, label_map

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    logger.info("Loading model artifacts from: %s", model_dir)

    model        = joblib.load(os.path.join(model_dir, "model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))

    with open(os.path.join(model_dir, "label_map.json")) as f:
        label_map = json.load(f)   # {"0": "not_eligible", "1": "eligible"}

    logger.info("Artifacts loaded. Classes: %s", list(label_map.values()))


# ---------------------------------------------------------------------------
# 2. run(raw_data) — Handle a single scoring request
# ---------------------------------------------------------------------------

def run(raw_data: str) -> str:
    """
    Called for every inference request.

    Accepted input formats
    ----------------------
    Single record:
        {"age": 34, "employment_type": "full_time", ...}

    Batch (list of records):
        [{"age": 34, ...}, {"age": 27, ...}]

    Azure ML wraps the payload in {"data": <your payload>} when using the
    managed online endpoint — both bare and wrapped formats are handled.

    Returns
    -------
    JSON string:
        {
            "predictions":   ["eligible", "not_eligible"],
            "probabilities": [[0.08, 0.92], [0.87, 0.13]],
            "label_order":   ["not_eligible", "eligible"]
        }
    """
    try:
        payload = json.loads(raw_data)

        # Unwrap Azure ML managed endpoint envelope if present
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]

        # Normalise to list
        if isinstance(payload, dict):
            payload = [payload]

        df = pd.DataFrame(payload)

        missing = set(EXPECTED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        df = df[EXPECTED_FEATURES]

        # Preprocess → predict
        X           = preprocessor.transform(df)
        raw_preds   = model.predict(X).tolist()
        class_proba = model.predict_proba(X).tolist()

        labeled_preds = [label_map[str(p)] for p in raw_preds]

        return json.dumps({
            "predictions":   labeled_preds,
            "probabilities": class_proba,
            "label_order":   [label_map[k] for k in sorted(label_map)],
        })

    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})
