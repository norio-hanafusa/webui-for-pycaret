# PyCaret WebUI

No-code machine learning platform built on [PyCaret](https://pycaret.org/) with a Streamlit web interface.

---

## Features

- **Data Management** -- CSV / Excel upload, interactive data editing, column operations, missing value imputation
- **Machine Learning** -- Classification, Regression, Clustering, Anomaly Detection
- **Model Comparison** -- Compare multiple algorithms with per-model progress tracking
- **Hyperparameter Tuning** -- Baseline vs tuned comparison table with multiple search libraries (scikit-learn, Optuna, scikit-optimize)
- **SHAP Analysis** -- 11 plot types: Summary Bar, Summary Dot, Waterfall, Beeswarm, Bar, Scatter, Dependence, Force, Decision, Violin, Heatmap
- **LIME Analysis** -- Local feature contribution visualization with interpretability metrics
- **Prediction** -- Predict on new data using trained or loaded models
- **Model Management** -- Save, load, rename, delete, download models (pickle / PyCaret format)
- **Preprocessing** -- SMOTE (5 methods), normalization, outlier removal
- **Multi-language** -- Japanese / English UI switching
- **Localhost only** -- Port binding restricted to 127.0.0.1

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Recommended: 8GB+ RAM, 10GB+ free disk space

### Start

```bash
git clone https://github.com/norio-hanafusa/pycaret-webui.git
cd pycaret-webui
docker compose up --build -d
```

Open http://localhost:8501 in your browser.

### Stop

```bash
docker compose down
```

---

## Project Structure

```
pycaret-webui/
  app.py              -- Main application
  i18n.py             -- Internationalization (Japanese / English)
  requirements.txt    -- Additional Python packages
  Dockerfile          -- Container build definition
  docker-compose.yml  -- Docker Compose configuration
  MANUAL.md           -- User manual (Japanese / English)
  data/               -- Volume mount for data files
  models/             -- Volume mount for saved models
  exports/            -- Volume mount for exported files
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Image | `pycaret/full:latest` |
| Web Framework | Streamlit |
| ML Library | PyCaret |
| Explainability | SHAP, LIME |
| Data Editor | streamlit-aggrid |
| Container | Docker / Docker Compose |

---

## Pages

| Page | Description |
|---|---|
| Data Loading | Upload CSV / Excel files, preview data, view statistics and types |
| Data View / Edit | Edit cells, drop columns, fill missing values |
| Machine Learning | Setup preprocessing, compare models, tune hyperparameters, predict |
| Results / Visualization | PyCaret plots, SHAP analysis (11 types), LIME analysis |
| New Data Prediction | Upload new data and predict using trained models |
| Load Model | Load pickle / PyCaret models from upload or server |
| Save / Export | Export data and predictions (CSV / Excel), save and manage models |

---

## SHAP Plot Types

| Plot | Description |
|---|---|
| Summary Plot (Bar) | Feature importance as bar chart |
| Summary Plot (Dot) | Feature impact distribution |
| Waterfall Plot | Single prediction breakdown |
| Beeswarm Plot | Overall feature impact distribution |
| Bar Plot | Mean absolute SHAP values |
| Scatter Plot | Feature value vs SHAP value |
| Dependence Plot | Feature interaction visualization |
| Force Plot | Individual or global prediction forces |
| Decision Plot | Cumulative feature contributions |
| Violin Plot | SHAP value distribution per feature |
| Heatmap Plot | SHAP values across samples and features |

---

## Preprocessing Options

- **SMOTE** -- Default SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
- **Normalization** -- Z-score, Min-Max, MaxAbs, Robust
- **Outlier Removal** -- Configurable threshold (0.01 - 0.10)

---

## Documentation

See [MANUAL.md](MANUAL.md) for the full user manual in Japanese and English.

---

## License

This project uses PyCaret and its dependencies. Please refer to the respective licenses of each library.
