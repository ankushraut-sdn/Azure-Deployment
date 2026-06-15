# Deploying the SDoH Health Risk Classifier to Microsoft Azure ML

This guide covers the full path from your local codebase to a live Azure ML
managed endpoint. The model is a PyTorch MLP (`SDoHMLP`) trained on Social
Determinants of Health features. Because it uses PyTorch rather than scikit-learn,
the Azure Environment must include `torch>=2.2` and the model artifacts include
a `model_config.json` that `score.py` reads to reconstruct the network architecture
before loading weights from `model.pt`.

---

## Prerequisites

**Azure resources**
- An active Azure subscription.
- An **Azure ML Workspace**. Create via CLI:

  ```bash
  az ml workspace create \
      --name sdoh-ml-workspace \
      --resource-group my-resource-group \
      --location eastus
  ```

- A **Service Principal** or your own user identity with the
  `AzureML Data Scientist` role on the workspace.

**Local tooling**

```bash
pip install azure-ai-ml azure-identity torch
az extension add --name ml
az login
az account set --subscription "My Subscription Name"
```

Confirm access:

```bash
az ml workspace show \
    --name sdoh-ml-workspace \
    --resource-group my-resource-group
```

---

## Step 1 — Generate Labeled Training Data (optional)

```bash
python generate_sdoh_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

Writes `./data/sdoh_examples.csv` and `./data/sdoh_examples.json`.

---

## Step 2 — Train the Model and Build the Azure Package Locally

```bash
# With your own labeled data
python train_dl_sdoh.py \
    --platform azure \
    --data-file ./data/sdoh_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_dl_sdoh.py \
    --platform azure \
    --samples 2000 \
    --output-dir ./dist
```

After completion, `./dist/azure/` contains:

```
dist/azure/
├── model/
│   ├── model.pt                  ← PyTorch state dict (weights only)
│   ├── model_config.json         ← architecture spec: input_dim, hidden_dims, dropout
│   ├── preprocessor.joblib       ← sklearn ColumnTransformer
│   └── label_map.json
├── code/
│   ├── score.py                  ← init() + run() entry-point
│   └── requirements.txt
├── metadata/
│   └── model_metadata.json
└── test/
    ├── sample_request.json       ← {"data": [...]}
    └── expected_response.json
```

> **Why model_config.json matters:** PyTorch saves only the weight tensor values
> in `model.pt`, not the class definition. `score.py` reads `model_config.json`
> first to reconstruct the `SDoHMLP` architecture, then calls `load_state_dict()`
> to populate the weights. Both files must be present in `AZUREML_MODEL_DIR`.

---

## Step 3 — Register the Model in Azure ML

```bash
az ml model create \
    --name sdoh-health-risk-classifier \
    --version 1 \
    --path ./dist/azure/model \
    --type custom_model \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group
```

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential          = DefaultAzureCredential(),
    subscription_id     = "00000000-0000-0000-0000-000000000000",
    resource_group_name = "my-resource-group",
    workspace_name      = "sdoh-ml-workspace",
)

model = ml_client.models.create_or_update(
    Model(
        name        = "sdoh-health-risk-classifier",
        version     = "1",
        path        = "./dist/azure/model",
        type        = "custom_model",
        description = "PyTorch MLP SDoH health risk classifier",
    )
)

print("Registered:", model.name, "version:", model.version)
```

---

## Step 4 — Create a PyTorch Environment for Scoring

Azure ML needs a versioned Environment that declares `torch>=2.2` among its
dependencies. This is the key difference from deploying a pure scikit-learn model.

### Option A — Azure CLI v2

Create `sdoh-env.yml`:

```yaml
# sdoh-env.yml
name: sdoh-pytorch-env
version: "1"
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
conda_file:
  name: sdoh-env
  channels:
    - defaults
    - conda-forge
    - pytorch
  dependencies:
    - python=3.10
    - pip:
        - numpy>=1.24
        - pandas>=2.0
        - scikit-learn>=1.3
        - torch>=2.2
        - joblib>=1.3
        - azureml-defaults
```

Register it:

```bash
az ml environment create \
    --file sdoh-env.yml \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group
```

### Option B — Python SDK v2

```python
from azure.ai.ml.entities import Environment

env = ml_client.environments.create_or_update(
    Environment(
        name    = "sdoh-pytorch-env",
        version = "1",
        image   = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file = {
            "name": "sdoh-env",
            "channels": ["defaults", "conda-forge", "pytorch"],
            "dependencies": [
                "python=3.10",
                {"pip": [
                    "numpy>=1.24",
                    "pandas>=2.0",
                    "scikit-learn>=1.3",
                    "torch>=2.2",
                    "joblib>=1.3",
                    "azureml-defaults",
                ]},
            ],
        },
    )
)

print("Environment registered:", env.name, "version:", env.version)
```

> **GPU environments:** If you want GPU-accelerated inference, replace the base
> image with a CUDA-enabled one, for example
> `mcr.microsoft.com/azureml/curated/pytorch-2.2-cuda11.8-cudnn8-py310`.
> For the inference volumes in this use case, CPU is sufficient.

---

## Step 5 — (Optional) Run a Training Job in Azure ML

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Register training data
data_asset = ml_client.data.create_or_update(
    Data(
        name    = "sdoh-training-data",
        version = "1",
        path    = "./data/sdoh_examples.csv",
        type    = AssetTypes.URI_FILE,
    )
)

job = command(
    code    = ".",
    command = (
        "python train_dl_sdoh.py "
        "--platform azure "
        "--data-file ${{inputs.training_data}} "
        "--output-dir ${{outputs.model_output}}"
    ),
    inputs  = {"training_data": Input(type="uri_file", path=data_asset.id)},
    outputs = {"model_output": Output(type="uri_folder")},
    environment  = "sdoh-pytorch-env:1",
    compute      = "gpu-cluster",        # use a GPU cluster for faster training
    display_name = "sdoh-training-run",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

> Create a GPU compute cluster: `az ml compute create --name gpu-cluster
> --type AmlCompute --size Standard_NC6 --min-instances 0 --max-instances 2
> --workspace-name sdoh-ml-workspace --resource-group my-resource-group`

---

## Step 6 — Create a Managed Online Endpoint and Deployment

### Option A — Azure CLI v2

```bash
# 1. Create the endpoint
az ml online-endpoint create \
    --name sdoh-health-risk-ep \
    --auth-mode key \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group

# 2. Create a deployment
az ml online-deployment create \
    --name blue \
    --endpoint-name sdoh-health-risk-ep \
    --model azureml:sdoh-health-risk-classifier:1 \
    --code-path ./dist/azure/code \
    --scoring-script score.py \
    --environment azureml:sdoh-pytorch-env:1 \
    --instance-type Standard_DS3_v2 \
    --instance-count 1 \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group

# 3. Route 100 % of traffic to blue
az ml online-endpoint update \
    --name sdoh-health-risk-ep \
    --traffic "blue=100" \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group
```

### Option B — Python SDK v2

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)

# 1. Create the endpoint
endpoint = ml_client.online_endpoints.begin_create_or_update(
    ManagedOnlineEndpoint(
        name      = "sdoh-health-risk-ep",
        auth_mode = "key",
    )
).result()

# 2. Create a deployment
deployment = ml_client.online_deployments.begin_create_or_update(
    ManagedOnlineDeployment(
        name          = "blue",
        endpoint_name = "sdoh-health-risk-ep",
        model         = "azureml:sdoh-health-risk-classifier:1",
        code_configuration = CodeConfiguration(
            code           = "./dist/azure/code",
            scoring_script = "score.py",
        ),
        environment    = "azureml:sdoh-pytorch-env:1",
        instance_type  = "Standard_DS3_v2",
        instance_count = 1,
    )
).result()

# 3. Route all traffic to blue
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("Scoring URI:", endpoint.scoring_uri)
```

Deployment typically takes 5–10 minutes. Check status:

```bash
az ml online-endpoint show \
    --name sdoh-health-risk-ep \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group \
    --query "provisioning_state"
```

---

## Step 7 — Retrieve the Endpoint Key and Scoring URI

```bash
# Scoring URI
az ml online-endpoint show \
    --name sdoh-health-risk-ep \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group \
    --query "scoring_uri" -o tsv

# Primary auth key
az ml online-endpoint get-credentials \
    --name sdoh-health-risk-ep \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group \
    --query "primaryKey" -o tsv
```

---

## Step 8 — Run Inference Against the Live Endpoint

The `score.py` entry-point expects `{"data": [...]}`.

### Python — azure-ai-ml SDK

```python
import json, tempfile, os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential          = DefaultAzureCredential(),
    subscription_id     = "00000000-0000-0000-0000-000000000000",
    resource_group_name = "my-resource-group",
    workspace_name      = "sdoh-ml-workspace",
)

payload = {
    "data": [
        {
            "age": 58, "prior_er_visits": 4, "household_size": 5,
            "pct_income_on_housing": 62,
            "income_level": "below_poverty", "insurance_status": "uninsured",
            "food_security": "very_low", "education_level": "less_than_hs",
            "health_literacy": "low", "housing_stability": "unstable",
            "neighborhood_safety": "unsafe", "rural_urban": "rural",
            "social_support": "none", "language_barrier": "yes",
            "has_primary_care": "no"
        },
        {
            "age": 34, "prior_er_visits": 0, "household_size": 3,
            "pct_income_on_housing": 22,
            "income_level": "middle", "insurance_status": "employer",
            "food_security": "secure", "education_level": "bachelors",
            "health_literacy": "high", "housing_stability": "homeowner",
            "neighborhood_safety": "high", "rural_urban": "urban",
            "social_support": "strong", "language_barrier": "no",
            "has_primary_care": "yes"
        }
    ]
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(payload, f)
    tmp = f.name

response = ml_client.online_endpoints.invoke(
    endpoint_name   = "sdoh-health-risk-ep",
    deployment_name = "blue",
    request_file    = tmp,
)
os.unlink(tmp)

print(json.loads(response))
# Expected:
# {
#   "predictions":   ["high_risk", "low_risk"],
#   "probabilities": [[0.09, 0.91], [0.88, 0.12]],
#   "label_order":   ["low_risk", "high_risk"]
# }
```

### Python — requests (raw HTTP)

```python
import requests, json

SCORING_URI = "https://sdoh-health-risk-ep.<region>.inference.ml.azure.com/score"
API_KEY     = "your-primary-key-here"

payload = {
    "data": [
        {
            "age": 58, "prior_er_visits": 4, "household_size": 5,
            "pct_income_on_housing": 62,
            "income_level": "below_poverty", "insurance_status": "uninsured",
            "food_security": "very_low", "education_level": "less_than_hs",
            "health_literacy": "low", "housing_stability": "unstable",
            "neighborhood_safety": "unsafe", "rural_urban": "rural",
            "social_support": "none", "language_barrier": "yes",
            "has_primary_care": "no"
        }
    ]
}

response = requests.post(
    SCORING_URI,
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json = payload,
)
response.raise_for_status()
print(response.json())
```

### Azure CLI

```bash
SCORING_URI=$(az ml online-endpoint show \
    --name sdoh-health-risk-ep \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group \
    --query "scoring_uri" -o tsv)

API_KEY=$(az ml online-endpoint get-credentials \
    --name sdoh-health-risk-ep \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group \
    --query "primaryKey" -o tsv)

curl -X POST "${SCORING_URI}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"data":[{"age":58,"prior_er_visits":4,"household_size":5,"pct_income_on_housing":62,"income_level":"below_poverty","insurance_status":"uninsured","food_security":"very_low","education_level":"less_than_hs","health_literacy":"low","housing_stability":"unstable","neighborhood_safety":"unsafe","rural_urban":"rural","social_support":"none","language_barrier":"yes","has_primary_care":"no"}]}'
```

---

## Step 9 — (Optional) Batch Inference with Azure ML Batch Endpoints

```python
from azure.ai.ml.entities import (
    BatchEndpoint, ModelBatchDeployment,
    ModelBatchDeploymentSettings, BatchRetrySettings,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction

batch_endpoint = ml_client.batch_endpoints.begin_create_or_update(
    BatchEndpoint(name="sdoh-batch-ep")
).result()

batch_deployment = ml_client.batch_deployments.begin_create_or_update(
    ModelBatchDeployment(
        name          = "batch-blue",
        endpoint_name = "sdoh-batch-ep",
        model         = "azureml:sdoh-health-risk-classifier:1",
        code_configuration = CodeConfiguration(
            code           = "./dist/azure/code",
            scoring_script = "score.py",
        ),
        environment = "azureml:sdoh-pytorch-env:1",
        compute     = "cpu-cluster",
        settings    = ModelBatchDeploymentSettings(
            instance_count   = 2,
            mini_batch_size  = 10,
            output_action    = BatchDeploymentOutputAction.APPEND_ROW,
            retry_settings   = BatchRetrySettings(max_retries=3, timeout=60),
        ),
    )
).result()

from azure.ai.ml import Input

batch_job = ml_client.batch_endpoints.invoke(
    endpoint_name = "sdoh-batch-ep",
    input = Input(type="uri_file", path="azureml:sdoh-training-data:1"),
)
ml_client.jobs.stream(batch_job.name)
```

---

## Step 10 — Tear Down the Endpoint

```bash
az ml online-endpoint delete \
    --name sdoh-health-risk-ep \
    --workspace-name sdoh-ml-workspace \
    --resource-group my-resource-group \
    --yes
```

```python
ml_client.online_endpoints.begin_delete(name="sdoh-health-risk-ep").result()
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Generate labeled data (`generate_sdoh_examples.py`) | ☐ |
| 2 | Run `train_dl_sdoh.py --platform azure` | ☐ |
| 3 | Register model folder in Azure ML | ☐ |
| 4 | Register PyTorch environment (must include `torch>=2.2`) | ☐ |
| 5 | Create managed online endpoint | ☐ |
| 6 | Create deployment and route 100 % traffic | ☐ |
| 7 | Retrieve scoring URI and API key | ☐ |
| 8 | Send a test request and validate output | ☐ |
| 9 | Delete endpoint when finished | ☐ |

---

## Key PyTorch-Specific Notes

| Concern | Detail |
|---------|--------|
| Model artifact | `model.pt` contains **weights only** (state dict), not the class definition |
| Architecture recovery | `score.py` reads `model_config.json` to rebuild `SDoHMLP`, then calls `load_state_dict()` |
| Eval mode | `load_artifacts()` calls `model.eval()` after loading; omitting this leaves BatchNorm and Dropout in training mode, causing non-deterministic predictions |
| No-grad context | `run_inference()` wraps the forward pass in `torch.no_grad()` to prevent gradient computation during inference |
| CPU vs GPU | `torch.load(..., map_location="cpu")` ensures weights load correctly on CPU instances even when saved from a GPU training run |
| Environment | `torch>=2.2` must be listed in the Azure Environment — it is absent from the Azure ML default curated environments for scikit-learn |
| Instance size | `Standard_DS3_v2` (4 vCPU, 14 GB RAM) is comfortable for CPU PyTorch inference; scale up if handling high concurrency |
