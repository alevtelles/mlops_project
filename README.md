<<<<<<< HEAD
# ML Classifier - Breast Cancer Prediction

A machine learning project that builds and deploys a neural network classifier for breast cancer prediction using the scikit-learn breast cancer dataset.

## Project Structure

```
/mlops_project
├── app/                          # Web application
│   ├── __init__.py               # Package marker
│   ├── main.py                   # Flask app with prediction API
│   └── templates/                # HTML templates for web UI
│       └── index.html            # Web interface for predictions
├── artifacts/                    # Preprocessing artifacts
├── data/                         # Data storage
│   ├── preprocessed/             # Cleaned data
│   ├── processed/                # Feature-engineered data
│   └── raw/                      # Raw dataset
├── metrics/                      # Model performance metrics
├── models/                       # Trained model
├── src/                          # Source code modules
│   ├── __init__.py               # Package marker
│   ├── data_loading/             # Data loading utilities
│   │   ├── __init__.py           # Package marker
│   │   └── load_data.py          # Dataset loading and preparation
│   ├── data_preprocessing/       # Data cleaning and splitting
│   │   ├── __init__.py           # Package marker
│   │   └── preprocess_data.py    # Data cleaning and imputation
│   ├── feature_engineering/      # Feature transformation utilities
│   │   ├── __init__.py           # Package marker
│   │   └── engineer_features.py  # Feature scaling and transformation
│   ├── model_evaluation/         # Model evaluation scripts
│   │   ├── __init__.py           # Package marker
│   │   └── evaluate_model.py     # Model performance evaluation
│   └── model_training/           # Model training scripts
│       ├── __init__.py           # Package marker
│       └── train_model.py        # Neural network training
├── .dockerignore                 # Docker ignore rules
├── Dockerfile                    # Docker build instructions
├── params.yaml                   # Configuration parameters
├── pyproject.toml                # Python dependencies and project metadata
└── README.md                     # Project documentation
```

## Features

- **Data Pipeline**: Complete ETL pipeline from raw data to model-ready features
- **Neural Network**: TensorFlow/Keras deep learning model with configurable architecture
- **Web Interface**: Flask-based web application for making predictions
- **Artifact Management**: Serialized models and preprocessors for deployment
- **Evaluation Metrics**: Comprehensive model performance analysis

## Dependencies

The project requires Python 3.12+ and the packages informed in `pyproject.toml`.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops_project
```

2. Install dependencies:
```bash
pip install -e .
```

## Configuration

Model hyperparameters and data processing settings are configured in `params.yaml`.

## Model Architecture

The neural network consists of a multilayer perceptron with 2 hidden layers.

## Artifacts

The training process generates the following files:

In the `models/` directory:
- `model.keras`: Trained TensorFlow model

In the `artifacts/` directory:
- `[features]_mean_imputer.joblib`: Feature imputer for missing values
- `[features]_scaler.joblib`: Standard scaler for feature normalization
- `[target]_one_hot_encoder.joblib`: One-hot encoder for target labels

## Metrics

Model performance metrics are saved to:
- `metrics/training.json`: Training history and validation metrics
- `metrics/evaluation.json`: Test set performance and confusion matrix

## Development

The project follows a modular structure with separate concerns:
- **Data Loading**: Fetches and saves raw breast cancer dataset
- **Preprocessing**: Handles missing values and data splitting
- **Feature Engineering**: Applies scaling transformations
- **Model Training**: Builds and trains the neural network
- **Model Evaluation**: Generates performance metrics
- **Web Application**: Provides prediction interface

Each module can be run independently and saves its outputs for the next stage in the pipeline.

## Usage

### Training the Model

Run the complete ML pipeline (for proper logging to the terminal, run as modules with `python -m`):

```bash
# 1. Load and prepare raw data
python -m src.data_loading.load_data

# 2. Preprocess data (imputation, train/test split)
python -m src.data_preprocessing.preprocess_data

# 3. Engineer features (scaling)
python -m src.feature_engineering.engineer_features

# 4. Train the neural network model
python -m src.model_training.train_model

# 5. Evaluate model performance
python -m src.model_evaluation.evaluate_model
```

### Running the Web Application

#### Flask

After training the model, start the Flask web server:
=======
# 🧠 Breast Cancer Prediction — Machine Learning & MLOps Pipeline

Projeto de **Machine Learning com abordagem end-to-end**, cobrindo **engenharia de dados, modelagem, avaliação, versionamento de artefatos e serving**, estruturado segundo **boas práticas de MLOps** adotadas em ambientes regulados (bancos, fintechs e big techs).

O objetivo do projeto é demonstrar **capacidade técnica em arquitetura de ML, qualidade de código, reprodutibilidade e deploy**, e não apenas a acurácia do modelo.

---

## 🎯 Objetivos Técnicos

- Construir um **pipeline de ML reproduzível e modular**
- Demonstrar **separação clara de responsabilidades**
- Garantir **rastreabilidade de dados, modelos e métricas**
- Disponibilizar o modelo via **serving HTTP**
- Tornar o projeto **pronto para CI/CD e evolução MLOps**

---

## 🧱 Arquitetura Geral

```text
Raw Data
   ↓
Data Loading
   ↓
Preprocessing (imputação + split)
   ↓
Feature Engineering (scaling)
   ↓
Model Training (Neural Network)
   ↓
Evaluation & Metrics
   ↓
Artifact Persistence
   ↓
Model Serving (Flask API)
```

**Decisões arquiteturais:**

- Pipeline desacoplado por estágio
- Artefatos persistidos entre etapas
- Execução determinística via configuração externa
- Pronto para orquestração (Airflow, Dagster, Prefect)

---

## 📁 Estrutura do Projeto

```text
/mlops_project
├── app/                          # Model Serving (Flask)
│   └── main.py                   # API HTTP para inferência
├── artifacts/                    # Artefatos de pré-processamento
├── data/
│   ├── raw/                      # Dados brutos
│   ├── preprocessed/             # Dados limpos
│   └── processed/                # Dados prontos para treino
├── metrics/                      # Métricas versionadas (JSON)
├── models/                       # Modelo treinado (.keras)
├── src/
│   ├── data_loading/
│   ├── data_preprocessing/
│   ├── feature_engineering/
│   ├── model_training/
│   └── model_evaluation/
├── params.yaml                   # Configuração desacoplada
├── Dockerfile
└── pyproject.toml
```

---

## ⚙️ Pipeline de Dados e ML

### 1️⃣ Data Loading

- Fonte: `sklearn.datasets.load_breast_cancer`
- Persistência do dataset bruto para reprodutibilidade

### 2️⃣ Pré-processamento

- Imputação de valores ausentes (Mean Imputer)
- Split treino/teste
- Persistência dos dados intermediários

### 3️⃣ Feature Engineering

- Normalização com `StandardScaler`
- Salvamento do scaler como artefato

### 4️⃣ Treinamento do Modelo

- Rede Neural (MLP – Multilayer Perceptron)
- TensorFlow/Keras
- Arquitetura configurável via `params.yaml`

### 5️⃣ Avaliação

- Acurácia, Precision, Recall, F1-score
- Matriz de confusão
- Métricas persistidas em JSON

---

## 🧪 Modelo

- Tipo: **Classificação Binária**
- Features: 30 atributos numéricos
- Saída: Benigno vs. Maligno
- Estratégia:

  - Simplicidade intencional para foco em **engenharia**
  - Fácil extensão para outros modelos

---

## 📦 Artefatos Versionados

| Tipo      | Arquivo                              |
| --------- | ------------------------------------ |
| Modelo    | `models/model.keras`                 |
| Imputador | `artifacts/*_mean_imputer.joblib`    |
| Scaler    | `artifacts/*_scaler.joblib`          |
| Encoder   | `artifacts/*_one_hot_encoder.joblib` |
| Métricas  | `metrics/*.json`                     |

Todos os artefatos são **determinísticos e reutilizáveis** no serving.

---

## 🌐 Model Serving

### API Flask
>>>>>>> a5a29ea (Projeto de MLOPs)

```bash
python app/main.py
```

<<<<<<< HEAD
The application will be available at `http://localhost:5001`

### Docker

You can instead build and run the application using Docker:

#### Build the Docker image

```bash
docker build -t ml-classifier .
```

#### Run the Docker container

```bash
docker run -p 5001:5001 ml-classifier
```

The web application will be available at `http://localhost:5001`.

### Making Predictions

1. **Web Interface**: Upload a CSV file with breast cancer features through the web interface
2. **API**: The `/upload` endpoint accepts CSV files and returns predictions

#### Required CSV Format

Your CSV file must contain all 30 breast cancer features with exact column names:
- mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
- See `sklearn.datasets.load_breast_cancer().feature_names` for the complete list
=======
Endpoint principal:

- `POST /upload` — recebe CSV e retorna predições

**Motivação técnica:**

- Flask usado para simplicidade e clareza
- Arquitetura facilmente migrável para **FastAPI + Uvicorn**

---

## 🐳 Containerização

O projeto inclui **Dockerfile** para padronização de ambiente:

```bash
docker build -t ml-classifier .
docker run -p 5001:5001 ml-classifier
```

Pronto para execução em:

- Kubernetes
- ECS
- Cloud Run
- Infra corporativa

---

## 🧠 Qualidade de Código & MLOps

- Código modular e legível
- Configuração externa (`params.yaml`)
- Separação treino × inferência
- Persistência explícita de artefatos
- Logs e métricas estruturadas
- Pipeline reexecutável de ponta a ponta

---

## 🔒 Considerações de Governança (Ambientes Financeiros)

- Dados controlados e rastreáveis
- Modelo reproduzível
- Artefatos versionáveis
- Pronto para integração com:

  - MLflow
  - Feature Store
  - Model Registry
  - Monitoramento de drift

---

## 🚀 Evoluções Planejadas

- CI com GitHub Actions (lint + tests)
- MLflow para tracking
- FastAPI para serving
- Testes automatizados de dados e modelos
- Monitoramento de inferência
- Detecção de data/model drift
>>>>>>> a5a29ea (Projeto de MLOPs)
