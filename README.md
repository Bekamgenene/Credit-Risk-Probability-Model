# Credit Risk Probability Model

This repository contains the code base for building, training, and deploying a credit-risk scoring service for **Bati Bank**.

## Project Structure

````text
credit-risk-model/
├── .github/workflows/ci.yml        # CI pipeline running tests & linting
├── data/                           # <- Git-ignored raw & processed data
│   ├── raw/                        # Raw data files
│   └── processed/                  # Cleaned / feature-engineered data
├── notebooks/
│   └── 1.0-eda.ipynb              # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering utilities
│   ├── train.py                   # Model training script (CLI)
│   ├── predict.py                 # Batch inference script (CLI)
│   └── api/
│       ├── main.py                # FastAPI app exposing prediction endpoint
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile                     # Container image definition
├── docker-compose.yml             # Local orchestration (API + model)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files/directories excluded from VCS
└── README.md                      # Project docs

## Quickstart

1.  Build Docker image:

    ```bash
    docker compose build
    ```
2.  Start the API locally (hot-reloaded):

    ```bash
    docker compose up
    ```
3.  Run unit tests:

    ```bash
    pytest -q
    ```

## Development Guidelines

* Use **feature branches** + PRs targeting `main`.
* All new code must include unit-tests and pass `pytest` & `ruff` linters.
* Keep notebooks lightweight; move reusable code to `src/`.
* Sensitive artefacts (models, data) must not be committed. Use DVC or S3.

# Credit Scoring Business Understanding

This section summarizes core business and regulatory considerations for building a credit scoring model for this project.

## 1. Why Basel II’s emphasis on risk measurement demands an interpretable, well-documented model

* **Capital & regulatory accountability:** Basel II ties regulatory capital to measured credit risk. Lenders must demonstrate how models quantify risk so regulators can assess capital adequacy. Clear, documented models support audits and capital calculations.
* **Model governance and validation:** Basel II expects robust model validation, back‑testing, and ongoing monitoring. Interpretable models make it easier to validate assumptions, trace decisions, and detect model drift.
* **Transparency to stakeholders:** Regulators, auditors, and senior management require reproducible evidence of how risk estimates are produced. An interpretable model simplifies explanations for provisioning, pricing, and risk limits.

## 2. Why a proxy variable is necessary and the business risks of using a proxy

* **Necessity of a proxy:** When a true ‘default’ label is unavailable, we must construct a proxy (e.g., 90+ days past due, charge‑off, bankruptcy, or a composite of delinquency and recovery events). This creates a target that approximates credit failure for supervised learning.
* **Types of commonly used proxies:** severe delinquency (30/60/90+ days), account write‑off, bankruptcy filing, significant drop in repayment behavior, or behavioral-clustered failure labels.
* **Business risks and caveats:**

  * *Label noise & measurement error:* A proxy may misclassify customers (false positives/negatives), reducing model accuracy and leading to poor credit decisions.
  * *Temporal mismatch:* Proxy definitions (e.g., 90+ days) introduce censoring and timing bias: recent accounts may not have had time to default.
  * *Policy and operational risk:* Policies driven by proxy-based scores (e.g., automated declines) can unfairly reject good customers or accept bad ones.
  * *Regulatory scrutiny:* Regulators may challenge models built on weak or opaque proxies—requiring additional justification, stress testing, and conservative capital buffers.
  * *Economic feedback loops:* Rejecting or accepting based on an imperfect proxy can change future observed defaults (selection bias), complicating model retraining and performance measurement.

## 3. Key trade-offs: simple interpretable models vs. complex high-performance models

* **Interpretability & compliance**

  * *Simple models (Logistic Regression + WoE):* Highly interpretable, easier to document, validate, and justify to regulators. Weight‑of‑Evidence (WoE) binning gives stable, monotonic relationships and simplifies scorecard deployment.
  * *Complex models (Gradient Boosting, ensembles):* Often attain higher predictive power but are harder to explain in human‑readable terms. Explainability tools (SHAP, LIME) help but may not fully satisfy regulatory expectations.
* **Performance vs. operational risk**

  * *Simple model advantages:* Lower risk of overfitting, easier to monitor and maintain, predictable behavior under distributional shifts, simpler to implement in legacy production pipelines.
  * *Complex model advantages:* Better at capturing nonlinearities and interactions, improving discrimination and profit lift — especially when features are abundant and richly informative.
* **Stability & monitoring**

  * Simple models tend to be more stable over time and less sensitive to small changes in feature distributions. Complex models may require more frequent retraining and a stronger monitoring framework.
* **Fairness and bias**

  * Complex models can amplify subtle biases in data. Interpretable models make it easier to detect, explain, and mitigate disparate impacts on protected groups.

## 4. Practical recommendation for a regulated financial context

1. **Start with an interpretable baseline:** Build a well‑documented Logistic Regression scorecard using WoE binning as the primary production model. This satisfies regulatory preference for transparency and forms a conservative, auditable baseline.
2. **Parallel evaluation of complex models:** Train Gradient Boosting or other high‑performance models in parallel for performance benchmarking and to extract additional signals (feature importances, nonlinear patterns). Use them to inform feature engineering in the interpretable model or as an advisory layer in decisioning.
3. **Robust proxy definition and sensitivity analysis:** Carefully document the proxy construction, test alternative proxy definitions, and report how sensitive model outcomes are to the proxy choice.
4. **Strong governance and validation:** Maintain comprehensive documentation (data lineage, preprocessing, hyperparameters, validation metrics, back‑tests), implement a model monitoring pipeline (population stability, PSI, AUC over time), and schedule periodic independent validations.
5. **Explainability toolkit:** Adopt explainability tools (WoE plots, scorecards, SHAP summaries) to provide both global and local explanations. For complex models, provide SHAP‑based narratives but anchor decisions in the interpretable baseline.
6. **Operational controls:** Use conservative decision thresholds, manual review for risky edge cases, and human‑in‑the‑loop workflows when the model is uncertain.

## 5. Summary

* Basel II increases the need for measurable, auditable, and well‑documented risk models. In the absence of true default labels, proxies are necessary but introduce label noise and regulatory risk. For deployment in a regulated environment, the recommended approach is a transparent scorecard as the main production model, supplemented by complex models for insight and performance benchmarking, backed by rigorous validation, monitoring, and governance.

---

*Deliverable:* this `README.md` section is prepared to be included in the project repository under the title **Credit Scoring Business Understanding**. If you want, I can also add a short bullet list of recommended proxy definitions and a checklist for documentation and validation.

## License

MIT
````
