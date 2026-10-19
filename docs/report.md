# Heart Disease MLOps Assignment Report

## 1. Overview

This project implements an end-to-end machine learning operations (MLOps) workflow for the UCI Heart Disease dataset. The main objective is to build a reproducible pipeline that acquires data, performs exploratory analysis, trains and compares predictive models, tracks experiments, packages the selected model, exposes it through a FastAPI service, and deploys the application in a containerized and cloud-ready environment.

The system is designed to mirror a real-world ML production lifecycle: data preparation, model validation, CI/CD automation, reproducibility, deployment, and operational monitoring. The repository contains all relevant components, including data scripts, model training logic, FastAPI inference service, unit tests, Docker packaging, Kubernetes deployment manifests, and documentation.

## 2. Problem Statement and Objectives

Heart disease prediction is a binary classification problem where the model predicts whether a patient is likely to have heart disease based on multiple health indicators. The assignment required the implementation of a complete ML pipeline using modern MLOps practices and the deployment of a production-style inference API.

The project objectives were to:

- Obtain and clean the dataset from the UCI Machine Learning Repository.
- Perform exploratory data analysis (EDA) to understand the data distribution and relationships.
- Engineer and preprocess the feature set consistently for training.
- Train and compare at least two classification models.
- Use experiment tracking to document model parameters and metrics.
- Package the best model for reproducible deployment.
- Build a FastAPI-based prediction service.
- Containerize and deploy the service using Docker and Kubernetes.
- Add testing, automation, and monitoring features.
- Present all outputs in a professional project report.

## 3. Dataset and Data Acquisition

The dataset used in this project is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. It contains patient health-related records and a target variable indicating whether heart disease is present or absent. The dataset includes attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia.

The data acquisition process is handled through a reproducible script located in the `scripts/` directory. The dataset is downloaded and prepared for analysis, and malformed values such as `?` are converted into missing values. Rows with missing values are handled appropriately before model training. All processed data artifacts are stored in a structured data directory for reuse.

## 4. Data Cleaning and Preprocessing

The data pipeline applies several preprocessing steps to ensure consistency and reproducibility:

- Missing value detection and handling.
- Conversion of categorical or ordinal values to appropriate numeric representations.
- Feature validation to ensure all required columns are available.
- Standardization for models that are sensitive to feature scale.
- Target encoding to a binary class label for classification.

This preprocessing is critical because medical datasets often contain inconsistent values and varying feature scales, which can significantly affect model performance. By validating the feature set before training, the pipeline remains stable and reusable.

## 5. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset structure and identify the most relevant predictive patterns before model training. The analysis included:

- Distribution plots for numerical variables such as age, cholesterol, resting blood pressure, and maximum heart rate.
- Correlation heatmaps to identify relationships among predictor variables.
- Class balance analysis to evaluate whether the target classes are balanced or imbalanced.
- Investigation of outliers and data quality issues.
- Analysis of relationships between patient attributes and disease status.

The EDA output helps validate the assumption that heart disease risk is influenced by multiple physiological indicators. It also serves as a useful artifact for communicating findings to stakeholders or academic reviewers.

## 6. Model Development and Evaluation

To satisfy the assignment requirement, at least two classification models were trained and compared:

- Logistic Regression
- Random Forest Classifier

These models were selected because they represent different learning paradigms: a linear probabilistic model and an ensemble-based tree model. Both were evaluated using consistent preprocessing and cross-validation techniques.

The evaluation process included standard metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-validation performance comparison

The final model selection was based on the most robust trade-off between generalization and predictive performance. The selected model was stored in a reusable format for deployment and inference.

## 7. Experiment Tracking with MLflow

MLflow was used to track model training experiments and preserve reproducibility. Each model run logs:

- Hyperparameters
- Training data version and feature configuration
- Evaluation metrics
- Model artifact references
- Plots and diagnostics when relevant

By storing experiment metadata in a local `mlruns/` directory, the project remains portable and can be reviewed independently of external services. MLflow also allows future experiments to be compared systematically to determine whether new feature engineering or model tuning improves performance.

## 8. Model Packaging and Reproducibility

The best model and associated preprocessing logic are packaged in a reproducible manner. This is important because a model is only useful in production when it can be loaded and executed predictably in the same configuration used during training.

The project includes:

- A versioned environment definition (`requirements.txt`)
- Reusable training logic
- Fixed feature ordering
- Serialized model artifact
- Inference code that loads the model and applies the same preprocessing rules

This design ensures that both training and inference follow the same pipeline, reducing drift and improving deployment safety.

## 9. API Development and FastAPI Service

The trained model is deployed through a FastAPI application that exposes an inference endpoint. The API accepts patient features as JSON and returns a prediction result together with the model confidence score and risk label.

Key API components:

- `/health` endpoint for service health checks
- `/` endpoint to verify the service is running
- `/predict` endpoint for inference requests

The service structure is organized into a dedicated `app/` package and is designed to be easy to run locally or inside a container.

## 10. Docker and Containerization

Containerization is a critical part of the MLOps workflow because it simplifies deployment and ensures that the application runs consistently across environments. A Dockerfile is included in the project and packages the FastAPI application and its dependencies into a single container image.

The containerized service can be built and run locally using Docker commands, allowing developers to validate the API before deployment. This also ensures that runtime dependencies and versions remain consistent with the trained model environment.

## 11. CI/CD Automation and Testing

The project includes a CI/CD pipeline defined in the GitHub Actions workflow. This automated pipeline is used to validate the repository on each push or pull request. It performs essential checks such as:

- Dependency installation
- Linting or static validation
- Unit test execution
- Model training smoke tests
- Basic verification of service functionality

The automation improves software quality by detecting regressions early and ensuring the project remains maintainable over time.

## 12. Deployment Manifest and Kubernetes Workflow

A Kubernetes deployment manifest is included in the repository. The deployment is designed to expose the FastAPI service through a cluster service and make it accessible in a production-like infrastructure environment.

This manifests the practical deployment step expected in the assignment by showing how the application can be scaled and managed in container orchestration. The structure supports future extensions such as ingress, load balancing, autoscaling, and environment-specific configuration.

## 13. Monitoring and Logging

Monitoring is an essential component of operational ML systems. The FastAPI service includes health-check endpoints and logs inference activities. This allows the application to be observed for operational readiness and helps diagnose failures or latency issues.

The project can be extended with more advanced monitoring tools such as Prometheus and Grafana. In a real production setup, this would provide dashboards for request counts, latency, error rates, and model serving health.

## 14. Architecture Overview

```mermaid
flowchart LR
    A[UCI Heart Disease Dataset] --> B[Data Acquisition Script]
    B --> C[Data Cleaning and Preprocessing]
    C --> D[EDA and Analysis]
    D --> E[Model Training]
    E --> F[MLflow Experiment Tracking]
    E --> G[Selected Model]
    G --> H[FastAPI Prediction Service]
    H --> I[Docker Container]
    I --> J[Kubernetes Deployment]
    J --> K[Monitoring and Health Checks]
```

## 15. Setup and Execution Instructions

To run the project locally, use the following steps:

```bash
pip install -r requirements.txt
python scripts/download_dataset.py
python src/eda.py
python src/train.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

## 16. Results and Conclusion

This project successfully demonstrates a complete MLOps pipeline for a binary medical classification problem. It covers the major phases expected in a real-world ML deployment: data acquisition, cleaning, analysis, model development, evaluation, tracking, packaging, API exposure, containerization, and deployment automation.

The repository is organized, reproducible, and suitable for academic submission or extension into a more advanced production system. The project also highlights the practical value of combining ML workflows with DevOps practices to ensure that models are not only accurate but also scalable, monitorable, and deployable.

## 17. Project Deliverables Summary

The final submission includes:

- GitHub repository with project code and configuration files
- Dataset and preprocessing pipeline
- EDA and model development work
- MLflow experiment tracking
- Serialized model artifact
- FastAPI prediction service
- Unit tests and CI workflow
- Dockerfile and deployment manifest
- Monitoring and health-check implementation
- Final report and relevant screenshots

This project demonstrates both technical implementation and MLOps best practices in a realistic, healthcare-oriented ML use case.
