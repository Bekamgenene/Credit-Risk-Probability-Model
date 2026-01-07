# Credit Risk Probability Model

This repository contains the codebase for building, training, tracking, and deploying a **credit risk probability scoring service** for **Bati Bank**. The project is designed with strong emphasis on **regulatory compliance, interpretability, and production readiness**.

---

## Project Overview

The Credit Risk Probability Model estimates the likelihood that a borrower will default on a loan. It provides an end-to-end pipeline covering:

* Data preprocessing and feature engineering
* Exploratory data analysis (EDA)
* Model training, evaluation, and selection
* Experiment tracking and model registry using MLflow
* Containerized deployment with Docker
* A RESTful prediction API using FastAPI

The system balances **predictive performance** with **model transparency**, making it suitable for regulated financial environments.

---

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml        # CI pipeline running tests & linting
├── data/                           # <- Git-ignored raw & processed data
│   ├── raw/                        # Raw data files
│   └── processed/                  # Cleaned / feature-engineered data
├── notebooks/
│   └── 1.0-eda.ipynb               # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering utilities
│   ├── train.py                    # Model training script (CLI)
│   ├── predict.py                  # Batch inference script (CLI)
│   └── api/
│       ├── main.py                 # FastAPI app exposing prediction endpoint
│       └── pydantic_models.py      # Request/response schemas
├── tests/
│   └── test_data_processing.py     # Unit tests
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Local orchestration (API + model)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files/directories excluded from VCS
└── README.md                      # Project documentation
```

---

## Quickstart

### 1. Build Docker Image

```bash
docker compose build
```

### 2. Start the API Locally (Hot Reload)

```bash
docker compose up
```

### 3. Run Unit Tests

```bash
pytest -q
```

---

## Model Training & Experiment Tracking (Task 5)

The training pipeline supports multiple candidate models:

* Logistic Regression
* Random Forest
* Gradient Boosting Machine (GBM)

Models are tuned using grid search and evaluated using **AUC on a hold-out validation set**.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch MLflow UI (Optional)

```bash
mlflow ui
```

### Train and Register Models

```bash
python -m src.train \
  --raw-path data/raw/data.csv \
  --model-out artifacts/best_model.pkl
```

The best-performing model is registered in the MLflow Model Registry as:

```
credit-risk-best
```

---

## Deploying & Serving Models from MLflow Model Registry

The API supports loading models directly from the **MLflow Model Registry**, enabling centralized governance and seamless promotion of models across stages.

### Environment Configuration

* `MLFLOW_TRACKING_URI` (optional): MLflow server URL (e.g. `http://localhost:5000`)
* `MLFLOW_MODEL_NAME` (optional): Registry model name (default: `credit-risk-best`)
* `MLFLOW_MODEL_STAGE` (optional): Model stage (default: `Production`)
* `LOCAL_MODEL_PATH` (optional): Local fallback model path

### Model Loading Logic

1. If `MLFLOW_TRACKING_URI` is set, the API loads:

```
models:/<MLFLOW_MODEL_NAME>/<MLFLOW_MODEL_STAGE>
```

2. If registry loading fails, the API falls back to the local model path.

This ensures robustness across development and production environments.

---

## Development Guidelines

* Use **feature branches** and Pull Requests targeting `main`
* All new code must include unit tests
* Code must pass `pytest` and `ruff` checks
* Keep notebooks lightweight; move reusable logic to `src/`
* Do **not** commit sensitive data or trained models
* Use DVC, object storage, or MLflow artifacts for large files

---

## Credit Scoring Business Understanding

### Basel II and Model Interpretability

Basel II links regulatory capital requirements to measured credit risk. Therefore:

* Models must be transparent and auditable
* Assumptions and risk estimates must be well-documented
* Validation, back-testing, and monitoring are mandatory

Interpretable models simplify regulatory review and governance.

---

### Use of Proxy Variables for Default

When true default labels are unavailable, proxy variables are required, such as:

* 30/60/90+ days past due
* Account write-off or charge-off
* Bankruptcy filing
* Severe repayment deterioration

#### Risks

* Label noise and misclassification
* Timing bias and censoring
* Policy and operational risk
* Regulatory scrutiny
* Feedback loops and selection bias

Careful proxy definition and sensitivity analysis are essential.

---

### Model Trade-offs

**Simple Models (Logistic Regression + WoE)**

* High interpretability
* Easier validation and monitoring
* Stable performance

**Complex Models (GBM, Ensembles)**

* Higher predictive power
* Harder to explain and govern
* Require advanced explainability tools (e.g., SHAP)

---

### Recommended Approach

1. Deploy an interpretable Logistic Regression scorecard as the primary model
2. Evaluate complex models in parallel for benchmarking
3. Carefully document proxy definitions
4. Maintain strong governance and monitoring
5. Apply conservative thresholds and human-in-the-loop controls

---

## Summary

This repository provides a production-ready, regulator-aware credit risk scoring system. It combines strong engineering practices with transparent modeling to meet both business and compliance requirements.

---

## License

MIT
