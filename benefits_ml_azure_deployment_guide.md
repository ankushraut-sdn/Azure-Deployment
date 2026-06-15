# Porting the Benefits Eligibility Classifier to Microsoft Azure ML

This guide walks from your local codebase to a live Azure ML managed endpoint —
training, packaging, uploading, deploying, and calling the inference API.
Your existing `train_gb_benefits.py` already contains the Azure packager, so the
artifacts are generated for you automatically.

---

## Prerequisites

**Azure resources**
- An active Azure subscription.
- An **Azure ML Workspace**. Create one in the portal at portal.azure.com →
  "Machine Learning" → "+ Create", or via CLI:

  ```bash
  az ml workspace create \
      --name benefits-ml-workspace \
      --resource-group my-resource-group \
      --location eastus
  ```

- A **Storage Account** (created automatically with the workspace; used as the
  default datastore).
- A **Service Principal** or your own user identity with the
  `AzureML Data Scientist` role on the workspace.

**Local tooling**

```bash
pip install azure-ai-ml azure-identity azureml-core
az extension add --name ml          # Azure ML CLI v2
az login                            # opens browser auth
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

If you do not already have a labeled CSV, use the bundled generator.

```bash
python generate_benefits_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

This writes `./data/benefits_examples.csv` and `./data/benefits_examples.json`.

---

## Step 2 — Train the Model and Build the Azure Package Locally

Run the training script targeting the Azure platform. Pass `--data-file` if you
have your own labeled data; omit it to use the built-in synthetic generator.

```bash
# With your own labeled data
python train_gb_benefits.py \
    --platform azure \
    --data-file ./data/benefits_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_gb_benefits.py \
    --platform azure \
    --samples 2000 \
    --output-dir ./dist
```

After the script completes, `./dist/azure/` will contain:

```
dist/azure/
├── model/
│   ├── model.pkl
│   ├── preprocessor.joblib
│   └── label_map.json
├── code/
│   ├── score.py              ← init() + run() entry-point
│   └── requirements.txt
├── metadata/
│   └── model_metadata.json
└── test/
    ├── sample_request.json   ← {"data": [...]}
    └── expected_response.json
```

---

## Step 3 — Register the Model in Azure ML

Upload the model artifacts to the workspace model registry. Azure ML will version
and track the files for you.

### Option A — Azure CLI v2

```bash
az ml model create \
    --name benefits-eligibility-classifier \
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
    credential      = DefaultAzureCredential(),
    subscription_id = "00000000-0000-0000-0000-000000000000",  # ← your sub ID
    resource_group_name = "my-resource-group",
    workspace_name  = "benefits-ml-workspace",
)

model = ml_client.models.create_or_update(
    Model(
        name    = "benefits-eligibility-classifier",
        version = "1",
        path    = "./dist/azure/model",
        type    = "custom_model",
        description = "GradientBoosting benefits eligibility classifier",
    )
)

print("Registered model:", model.name, "version:", model.version)
```

---

## Step 4 — Create an Environment for Scoring

Azure ML needs a versioned Environment that describes the Python dependencies
used during inference. Build it from the `requirements.txt` the packager
already wrote.

### Option A — Azure CLI v2

Create an environment YAML file:

```yaml
# azure-env.yml
name: benefits-scorer-env
version: "1"
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
conda_file:
  name: benefits-env
  channels:
    - defaults
    - conda-forge
  dependencies:
    - python=3.10
    - pip:
        - numpy>=1.24
        - pandas>=2.0
        - scikit-learn>=1.3
        - xgboost>=2.0
        - joblib>=1.3
        - azureml-defaults
```

Register it:

```bash
az ml environment create \
    --file azure-env.yml \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group
```

### Option B — Python SDK v2

```python
from azure.ai.ml.entities import Environment, BuildContext

env = ml_client.environments.create_or_update(
    Environment(
        name    = "benefits-scorer-env",
        version = "1",
        image   = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file = {
            "name": "benefits-env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.10",
                {"pip": [
                    "numpy>=1.24",
                    "pandas>=2.0",
                    "scikit-learn>=1.3",
                    "xgboost>=2.0",
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

If you want Azure ML to manage training rather than doing it locally:

```python
from azure.ai.ml import command, Input
from azure.ai.ml.entities import JobService

# Upload local training data to the default datastore
data_asset = ml_client.data.create_or_update(
    Data(
        name    = "benefits-training-data",
        version = "1",
        path    = "./data/benefits_examples.csv",
        type    = "uri_file",
    )
)

job = command(
    code         = ".",                                   # uploads current directory
    command      = (
        "python train_gb_benefits.py "
        "--platform azure "
        "--data-file ${{inputs.training_data}} "
        "--output-dir ${{outputs.model_output}}"
    ),
    inputs       = {"training_data": Input(type="uri_file", path=data_asset.id)},
    outputs      = {"model_output": Output(type="uri_folder")},
    environment  = "benefits-scorer-env:1",
    compute      = "cpu-cluster",                         # ← your compute cluster name
    display_name = "benefits-training-run",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)   # tails logs until completion
```

> **Note:** Create a compute cluster in the portal (Compute → Compute clusters →
> "+ New") or via CLI:
> `az ml compute create --name cpu-cluster --type AmlCompute --min-instances 0
> --max-instances 2 --size Standard_DS3_v2 --workspace-name benefits-ml-workspace
> --resource-group my-resource-group`

---

## Step 6 — Create a Managed Online Endpoint

Azure ML managed online endpoints handle scaling, authentication, and traffic
splitting for you.

### Option A — Azure CLI v2

```bash
# 1. Create the endpoint
az ml online-endpoint create \
    --name benefits-eligibility-ep \
    --auth-mode key \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group

# 2. Create a deployment on that endpoint
az ml online-deployment create \
    --name blue \
    --endpoint-name benefits-eligibility-ep \
    --model azureml:benefits-eligibility-classifier:1 \
    --code-path ./dist/azure/code \
    --scoring-script score.py \
    --environment azureml:benefits-scorer-env:1 \
    --instance-type Standard_DS2_v2 \
    --instance-count 1 \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group

# 3. Route 100 % of traffic to the blue deployment
az ml online-endpoint update \
    --name benefits-eligibility-ep \
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
        name      = "benefits-eligibility-ep",
        auth_mode = "key",
    )
).result()

# 2. Create a deployment
deployment = ml_client.online_deployments.begin_create_or_update(
    ManagedOnlineDeployment(
        name          = "blue",
        endpoint_name = "benefits-eligibility-ep",
        model         = "azureml:benefits-eligibility-classifier:1",
        code_configuration = CodeConfiguration(
            code           = "./dist/azure/code",
            scoring_script = "score.py",
        ),
        environment   = "azureml:benefits-scorer-env:1",
        instance_type  = "Standard_DS2_v2",
        instance_count = 1,
    )
).result()

# 3. Send all traffic to blue
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("Endpoint URI:", endpoint.scoring_uri)
```

Deployment typically takes 5–10 minutes. Check status:

```bash
az ml online-endpoint show \
    --name benefits-eligibility-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "provisioning_state"
```

---

## Step 7 — Retrieve the Endpoint Key and Scoring URI

```bash
# Scoring URI
az ml online-endpoint show \
    --name benefits-eligibility-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "scoring_uri" -o tsv

# Primary auth key
az ml online-endpoint get-credentials \
    --name benefits-eligibility-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "primaryKey" -o tsv
```

---

## Step 8 — Run Inference Against the Live Endpoint

The `score.py` generated by your packager expects `{"data": [...]}` — matching
the `sample_request.json` in `dist/azure/test/`.

### Python — azure-ai-ml SDK

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential          = DefaultAzureCredential(),
    subscription_id     = "00000000-0000-0000-0000-000000000000",
    resource_group_name = "my-resource-group",
    workspace_name      = "benefits-ml-workspace",
)

import json, tempfile, os

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

# Write to a temp file — SDK invoke expects a file path
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(payload, f)
    tmp = f.name

response = ml_client.online_endpoints.invoke(
    endpoint_name   = "benefits-eligibility-ep",
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

### Python — requests (raw HTTP, no Azure SDK)

```python
import requests, json

SCORING_URI = "https://benefits-eligibility-ep.<region>.inference.ml.azure.com/score"
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

### Azure CLI v2

```bash
SCORING_URI=$(az ml online-endpoint show \
    --name benefits-eligibility-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --query "scoring_uri" -o tsv)

API_KEY=$(az ml online-endpoint get-credentials \
    --name benefits-eligibility-ep \
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

For scoring large files offline without a persistent real-time endpoint:

```python
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    BatchRetrySettings,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction

# 1. Create a batch endpoint
batch_endpoint = ml_client.batch_endpoints.begin_create_or_update(
    BatchEndpoint(name="benefits-batch-ep")
).result()

# 2. Create a batch deployment
batch_deployment = ml_client.batch_deployments.begin_create_or_update(
    ModelBatchDeployment(
        name          = "batch-blue",
        endpoint_name = "benefits-batch-ep",
        model         = "azureml:benefits-eligibility-classifier:1",
        code_configuration = CodeConfiguration(
            code           = "./dist/azure/code",
            scoring_script = "score.py",
        ),
        environment   = "azureml:benefits-scorer-env:1",
        compute       = "cpu-cluster",
        settings = ModelBatchDeploymentSettings(
            instance_count  = 2,
            max_concurrency_per_instance = 2,
            mini_batch_size = 10,
            output_action   = BatchDeploymentOutputAction.APPEND_ROW,
            retry_settings  = BatchRetrySettings(max_retries=3, timeout=30),
        ),
    )
).result()

# 3. Submit a batch job
from azure.ai.ml import Input

batch_job = ml_client.batch_endpoints.invoke(
    endpoint_name = "benefits-batch-ep",
    input = Input(
        type = "uri_file",
        path = "azureml:benefits-training-data:1",
    ),
)
ml_client.jobs.stream(batch_job.name)
```

---

## Step 10 — Tear Down the Endpoint

Managed online endpoints incur charges while running. Delete when done.

```bash
az ml online-endpoint delete \
    --name benefits-eligibility-ep \
    --workspace-name benefits-ml-workspace \
    --resource-group my-resource-group \
    --yes
```

Or via Python:

```python
ml_client.online_endpoints.begin_delete(name="benefits-eligibility-ep").result()
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Generate labeled data (`generate_benefits_examples.py`) | ☐ |
| 2 | Run `train_gb_benefits.py --platform azure` | ☐ |
| 3 | Register model (`az ml model create` or SDK) | ☐ |
| 4 | Register environment with dependencies | ☐ |
| 5 | Create managed online endpoint | ☐ |
| 6 | Create deployment and route traffic | ☐ |
| 7 | Retrieve scoring URI and API key | ☐ |
| 8 | Send a test request and validate output | ☐ |
| 9 | Delete endpoint when finished | ☐ |

---

## Key Environment Variables Inside the Container

| Variable | Set by Azure ML | Purpose |
|----------|----------------|---------|
| `AZUREML_MODEL_DIR` | Path to registered model files | Where `init()` loads artifacts from |
| `AML_APP_ROOT` | Scoring script directory | Working directory for `score.py` |
| `AZUREML_RUN_ID` | Unique job identifier | Useful for logging and traceability |
| `AZUREML_ARM_WORKSPACE_NAME` | Workspace name | Available for SDK calls inside the container |

Your `score.py` already reads `AZUREML_MODEL_DIR` correctly in `init()` — Azure
ML sets this variable automatically before calling `init()`.

---

## Differences from the SageMaker Deployment

| Concern | SageMaker | Azure ML |
|---------|-----------|----------|
| Entry-point functions | `model_fn / input_fn / predict_fn / output_fn` | `init() / run()` |
| Request envelope key | `instances` | `data` |
| Model archive format | `model.tar.gz` (required) | Plain folder upload (no tar needed) |
| Auth header | `--` (IAM-based by default) | `Authorization: Bearer <key>` |
| Dependency declaration | Container image + `requirements.txt` in tar | Versioned Environment asset |
| Traffic splitting | Endpoint config weights | Deployment-level traffic percentages |
