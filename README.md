# WebUI for PyCaret

No-code machine learning platform built on [PyCaret](https://pycaret.org/) with a Streamlit web interface.

---

## Features

- **Data Management** -- CSV / Excel upload, interactive data editing, column operations, missing value imputation
- **Machine Learning** -- Classification, Regression, Clustering, Anomaly Detection
- **Model Comparison** -- Compare multiple algorithms with per-model progress tracking
- **Hyperparameter Tuning** -- Baseline vs tuned comparison table with multiple search libraries (scikit-learn, Optuna, scikit-optimize)
- **SHAP Analysis** -- Bulk SHAP computation with caching, configurable sample count slider, 11 plot types: Summary Bar, Summary Dot, Waterfall, Beeswarm, Bar, Scatter, Dependence, Force, Decision, Violin, Heatmap
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
git clone https://github.com/norio-hanafusa/webui-for-pycaret.git
cd webui-for-pycaret
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
webui-for-pycaret/
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
| Results / Visualization | PyCaret plots, SHAP analysis (bulk compute + cache, 11 plot types), LIME analysis |
| New Data Prediction | Upload new data and predict using trained models |
| Load Model | Load pickle / PyCaret models from upload or server |
| Save / Export | Export data and predictions (CSV / Excel), save and manage models |

---

## SHAP Analysis

### Bulk Computation & Caching

SHAP values are computed once in bulk and cached in session state. Switching between plot types does not trigger recomputation, enabling fast interactive exploration.

- **Sample count slider** -- Select the number of samples for SHAP computation (50 to actual X_train size). TreeExplainer handles full datasets efficiently; KernelExplainer is recommended with 200 or fewer samples.
- **Progress indicator** -- Real-time progress bar during computation (batch processing for KernelExplainer).
- **Cache reset** -- One-click button to clear cached SHAP values and recompute.

### Explainer Priority

1. **TreeExplainer** -- Fast, supports full data for tree-based models
2. **KernelExplainer** -- Model-agnostic, batch processing with progress bar
3. **PermutationExplainer** -- Fallback for unsupported models

### Plot Types (11 types)

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

## Known Issues

| Issue | Description |
|---|---|
| Clustering -- Plot display | Some PyCaret visualization plots (e.g., Cluster Distribution) may fail to render due to incompatibilities between PyCaret's internal plotting backend (plotly/yellowbrick) and Streamlit's image capture. Elbow and Silhouette plots work correctly. |
| Anomaly Detection -- Plot display | Similar to clustering, certain anomaly detection visualization plots may produce errors when PyCaret uses a non-matplotlib backend internally. |

---

## Documentation

See [MANUAL.md](MANUAL.md) for the full user manual in Japanese and English.

---

## Author / Development Process

This project was developed by [norio-hanafusa](https://github.com/norio-hanafusa).

All requirements definition, architectural decisions, feature specifications,
testing, bug reporting, and iterative refinement instructions were provided
by the author. Code generation was assisted by Claude Code (Anthropic),
functioning as an AI-powered development tool under the author's direction.

The development followed an iterative process:
1. The author defined feature requirements and specifications
2. Claude Code generated code based on those instructions
3. The author tested the output and identified issues
4. The author provided correction instructions and additional requirements
5. Steps 2-4 were repeated until each feature met the author's standards

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 norio-hanafusa

### Third-Party Licenses

This project depends on the following open-source libraries.
Users and redistributors must comply with each library's license terms.

| Library | License | URL |
|---|---|---|
| PyCaret | MIT | https://github.com/pycaret/pycaret |
| Streamlit | Apache 2.0 | https://github.com/streamlit/streamlit |
| SHAP | MIT | https://github.com/shap/shap |
| LIME | BSD 2-Clause | https://github.com/marcotcr/lime |
| scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| LightGBM | MIT | https://github.com/microsoft/LightGBM |
| XGBoost | Apache 2.0 | https://github.com/dmlc/xgboost |
| CatBoost | Apache 2.0 | https://github.com/catboost/catboost |
| pandas | BSD 3-Clause | https://github.com/pandas-dev/pandas |
| NumPy | BSD 3-Clause | https://github.com/numpy/numpy |
| matplotlib | PSF-based | https://github.com/matplotlib/matplotlib |
| streamlit-aggrid | MIT | https://github.com/PablocFonseca/streamlit-aggrid |
| openpyxl | MIT | https://github.com/theorchard/openpyxl |
| xlsxwriter | BSD 2-Clause | https://github.com/jmcnamara/XlsxWriter |
