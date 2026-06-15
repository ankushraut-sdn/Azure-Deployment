# Deploying the Benefits Eligibility Classifier (Neural Network) to Microsoft Azure ML

This guide covers the full path from your local codebase to a live Azure ML
managed endpoint. The model is a PyTorch MLP (`BenefitsMLP`) that predicts
employee benefits eligibility. Because it uses PyTorch, the Azure Environment
must explicitly declare `torch>=2.2` and the model artifacts include a
`model_config.json` that `score.py` reads to reconstruct the network architecture
before loading weights from `model.pt`.

---

## Prerequisites

**Azure resources**
- An active Azure subscription.
- An **Azure ML Workspace**:

  ```bash
  az ml workspace create \
      --name benefits-ml-workspace \
      --resource-group my-resource-group \
      --location eastus
  ```

- A Service Principal or your own user identity with the
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
    --name benefits-ml-workspace \
    --resource-group my-resource-group
```

---

## Step 1 — Generate Labeled Training Data (optional)

```bash
python generate_benefits_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

Writes `./data/benefits_examples.csv` and `./data/benefits_examples.json`.

---

## Step 2 — Train the Model and Build the Azure Package Locally

```bash
# With your own labeled data
python train_dl_benefits.py \
    --platform azure \
    --data-file ./data/benefits_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_dl_benefits.py \
    --platform azure \
    --samples 2000 \
    --output-dir ./dist
```

After completion `./dist/azure/` contains:

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

> **Why `model_config.json` is required:** PyTorch saves only the weight tensor
> values in `model.pt`. `score.py` reads `model_config.json` first to reconstruct
> the `BenefitsMLP` architecture, then calls `load_state_dict()` to populate the
> weights. Both files must be present in `AZUREML_MODEL_DIR` at runtime.

---

## Step 3 — Register the Model in Azure ML

### Option A — Azure CLI v2

```bash
az ml model create \
    --name benefits-eligibility-nn \
    --version 1 \
    --path ./dist/azure/model \
    --type custom_model \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group
```

### Option B — Python SDK v2

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential          = DefaultAzureCredential(),
    subscription_id     = "00000000-0000-0000-0000-000000000000",
    resource_group_name = "my-resource-group",
    workspace_name      = "benefits-ml-workspace",
)

model = ml_client.models.create_or_update(
    Model(
        name        = "benefits-eligibility-nn",
        version     = "1",
        path        = "./dist/azure/model",
        type        = "custom_model",
        description = "PyTorch MLP benefits eligibility classifier v2",
    )
)

print("Registered:", model.name, "version:", model.version)
```

---

## Step 4 — Create a PyTorch Environment for Scoring

The Azure Environment must include `torch>=2.2`. This is absent from Azure ML's
curated scikit-learn environments and must be declared explicitly.

### Option A — Azure CLI v2

Create `benefits-env.yml`:

```yaml
# benefits-env.yml
name: benefits-pytorch-env
version: "1"
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
conda_file:
  name: benefits-env
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
    --file benefits-env.yml \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group
```

### Option B — Python SDK v2

```python
from azure.ai.ml.entities import Environment

env = ml_client.environments.create_or_update(
    Environment(
        name    = "benefits-pytorch-env",
        version = "1",
        image   = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file = {
            "name": "benefits-env",
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

---

## Step 5 — (Optional) Run a Training Job in Azure ML

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Register training data
data_asset = ml_client.data.create_or_update(
    Data(
        name    = "benefits-training-data",
        version = "1",
        path    = "./data/benefits_examples.csv",
        type    = AssetTypes.URI_FILE,
    )
)

job = command(
    code    = ".",
    command = (
        "python train_dl_benefits.py "
        "--platform azure "
        "--data-file ${{inputs.training_data}} "
        "--output-dir ${{outputs.model_output}}"
    ),
    inputs  = {"training_data": Input(type="uri_file", path=data_asset.id)},
    outputs = {"model_output": Output(type="uri_folder")},
    environment  = "benefits-pytorch-env:1",
    compute      = "gpu-cluster",
    display_name = "benefits-nn-training-run",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

> Create a GPU compute cluster:
> `az ml compute create --name gpu-cluster --type AmlCompute
> --size Standard_NC6 --min-instances 0 --max-instances 2
> --workspace-name benefits-ml-workspace --resource-group my-resource-group`

---

## Step 6 — Create a Managed Online Endpoint and Deployment

### Option A — Azure CLI v2

```bash
# 1. Create the endpoint
az ml online-endpoint create \
    --name benefits-eligibility-nn-ep \
    --auth-mode key \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group

# 2. Create a deployment
az ml online-deployment create \
    --name blue \
    --endpoint-name benefits-eligibility-nn-ep \
    --model azureml:benefits-eligibility-nn:1 \
    --code-path ./dist/azure/code \
    --scoring-script score.py \
    --environment azureml:benefits-pytorch-env:1 \
    --instance-type Standard_DS3_v2 \
    --instance-count 1 \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group

# 3. Route 100 % of traffic to blue
az ml online-endpoint update \
    --name benefits-eligibility-nn-ep \
    --traffic "blue=100" \
    --workspace-name benefits-ml-workspace \
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
        name      = "benefits-eligibility-nn-ep",
        auth_mode = "key",
    )
).result()

# 2. Create a deployment
deployment = ml_client.online_deployments.begin_create_or_update(
    ManagedOnlineDeployment(
        name          = "blue",
        endpoint_name = "benefits-eligibility-nn-ep",
        model         = "azureml:benefits-eligibility-nn:1",
        code_configuration = CodeConfiguration(
            code           = "./dist/azure/code",
            scoring_script = "score.py",
        ),
        environment    = "azureml:benefits-pytorch-env:1",
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
    --name benefits-eligibility-nn-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "provisioning_state"
```

---

## Step 7 — Retrieve the Endpoint Key and Scoring URI

```bash
# Scoring URI
az ml online-endpoint show \
    --name benefits-eligibility-nn-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "scoring_uri" -o tsv

# Primary auth key
az ml online-endpoint get-credentials \
    --name benefits-eligibility-nn-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "primaryKey" -o tsv
```

---

## Step 8 — Run Inference Against the Live Endpoint

`score.py` expects `{"data": [...]}`.

### Python — azure-ai-ml SDK

```python
import json, tempfile, os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential          = DefaultAzureCredential(),
    subscription_id     = "00000000-0000-0000-0000-000000000000",
    resource_group_name = "my-resource-group",
    workspace_name      = "benefits-ml-workspace",
)

payload = {
    "data": [
        {
            "age": 34, "employment_type": "full_time", "years_of_service": 5,
            "weekly_hours": 40, "department": "engineering",
            "salary_band": "band_3", "has_dependents": 1
        },
        {
            "age": 27, "employment_type": "part_time", "years_of_service": 0,
            "weekly_hours": 20, "department": "sales",
            "salary_band": "band_1", "has_dependents": 0
        }
    ]
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(payload, f)
    tmp = f.name

response = ml_client.online_endpoints.invoke(
    endpoint_name   = "benefits-eligibility-nn-ep",
    deployment_name = "blue",
    request_file    = tmp,
)
os.unlink(tmp)

print(json.loads(response))
# Expected:
# {
#   "predictions":   ["eligible", "not_eligible"],
#   "probabilities": [[0.08, 0.92], [0.87, 0.13]],
#   "label_order":   ["not_eligible", "eligible"]
# }
```

### Python — requests (raw HTTP)

```python
import requests, json

SCORING_URI = "https://benefits-eligibility-nn-ep.<region>.inference.ml.azure.com/score"
API_KEY     = "your-primary-key-here"

payload = {
    "data": [
        {
            "age": 34, "employment_type": "full_time", "years_of_service": 5,
            "weekly_hours": 40, "department": "engineering",
            "salary_band": "band_3", "has_dependents": 1
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
    --name benefits-eligibility-nn-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "scoring_uri" -o tsv)

API_KEY=$(az ml online-endpoint get-credentials \
    --name benefits-eligibility-nn-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "primaryKey" -o tsv)

curl -X POST "${SCORING_URI}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"data":[{"age":34,"employment_type":"full_time","years_of_service":5,"weekly_hours":40,"department":"engineering","salary_band":"band_3","has_dependents":1}]}'
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
    BatchEndpoint(name="benefits-nn-batch-ep")
).result()

batch_deployment = ml_client.batch_deployments.begin_create_or_update(
    ModelBatchDeployment(
        name          = "batch-blue",
        endpoint_name = "benefits-nn-batch-ep",
        model         = "azureml:benefits-eligibility-nn:1",
        code_configuration = CodeConfiguration(
            code           = "./dist/azure/code",
            scoring_script = "score.py",
        ),
        environment = "azureml:benefits-pytorch-env:1",
        compute     = "cpu-cluster",
        settings    = ModelBatchDeploymentSettings(
            instance_count  = 2,
            mini_batch_size = 10,
            output_action   = BatchDeploymentOutputAction.APPEND_ROW,
            retry_settings  = BatchRetrySettings(max_retries=3, timeout=60),
        ),
    )
).result()

from azure.ai.ml import Input

batch_job = ml_client.batch_endpoints.invoke(
    endpoint_name = "benefits-nn-batch-ep",
    input = Input(type="uri_file", path="azureml:benefits-training-data:1"),
)
ml_client.jobs.stream(batch_job.name)
```

---

## Step 10 — Tear Down the Endpoint

```bash
az ml online-endpoint delete \
    --name benefits-eligibility-nn-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --yes
```

```python
ml_client.online_endpoints.begin_delete(name="benefits-eligibility-nn-ep").result()
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Generate labeled data (`generate_benefits_examples.py`) | ☐ |
| 2 | Run `train_dl_benefits.py --platform azure` | ☐ |
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
| Model artifact | `model.pt` is a **state dict only** — weights but not the class definition |
| Architecture recovery | `score.py` reads `model_config.json` to instantiate `BenefitsMLP`, then calls `load_state_dict()` |
| Both files required | `model.pt` and `model_config.json` must both be present in the registered model folder |
| Eval mode | `load_artifacts()` calls `model.eval()`; without this `BatchNorm1d` and `Dropout` remain in training mode |
| No-grad context | `run_inference()` wraps the forward pass in `torch.no_grad()` to prevent gradient computation |
| CPU loading | `map_location="cpu"` ensures weights load correctly on CPU instances even when trained on GPU |
| Environment | `torch>=2.2` must be listed explicitly — it is absent from Azure ML's curated sklearn environments |
| Instance size | `Standard_DS3_v2` (4 vCPU, 14 GB RAM) is comfortable for CPU PyTorch inference at typical HR workloads |
