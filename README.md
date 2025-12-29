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

> > > > > > > a5a29ea (Projeto de MLOPs)

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
- # See `sklearn.datasets.load_breast_cancer().feature_names` for the complete list

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
