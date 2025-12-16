## Deploying & serving models from MLflow Model Registry

The API supports loading models directly from an MLflow Model Registry. This allows
centralized model promotion and hot-swapping of production models without rebuilding
containers.

Configuration (environment variables):

- `MLFLOW_TRACKING_URI` (optional): URL of the MLflow tracking server (e.g. `http://localhost:5000`).
  If set, the service will attempt to load a registered model from the registry.
- `MLFLOW_MODEL_NAME` (optional): Registry model name. Defaults to `credit-risk-best`.
- `MLFLOW_MODEL_STAGE` (optional): Model stage to load from the registry. Defaults to `Production`.
- `LOCAL_MODEL_PATH` (optional): Local fallback path for a pickled model artifact. Defaults to `/app/artifacts/best_model.pkl`.

Model selection logic

1. If `MLFLOW_TRACKING_URI` is provided the service constructs a model URI of the
   form `models:/<MLFLOW_MODEL_NAME>/<MLFLOW_MODEL_STAGE>` (for example
   `models:/credit-risk-best/Production`) and tries to load it with
   `mlflow.sklearn.load_model`.
2. If registry loading fails (e.g. network/authorization or model not found) the
   service falls back to loading a local file at `LOCAL_MODEL_PATH` (useful for
   development/docker-compose workflows).

Example (docker-compose.override.yml env snippet):

```yaml
services:
  api:
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_MODEL_NAME=credit-risk-best
      - MLFLOW_MODEL_STAGE=Production
      - LOCAL_MODEL_PATH=/app/artifacts/best_model.pkl
```

Notes

- The training script logs models to MLflow using `mlflow.sklearn.log_model`, so
  loading via `mlflow.sklearn.load_model("models:/...")` preserves the sklearn
  estimator API (including `predict_proba`).
- Make sure the API container has network access to the MLflow tracking server and
  the model is registered under the given name and stage.
- If you prefer to use mlflow.pyfunc for arbitrary pyfunc models, switch the load call to:
  `mlflow.pyfunc.load_model(model_uri)` and adapt prediction handling for the pyfunc output.
