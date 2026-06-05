# Car Insurance Pricing MVP Plan

This document outlines the goal, scope, and implementation plan for the Minimum Viable Product (MVP) of the Car Insurance Pricing data science project.

---

## 1. Goal of the MVP
Create a first working end-to-end version of a car insurance pricing data science pipeline. The MVP focuses on policy claim frequency, loading the French Motor Third-Party Liability datasets, performing initial exploration, building a Poisson GLM actuarial baseline, training a Random Forest comparison model, evaluating both models using standard regression metrics, and documenting findings.

---

## 2. Scope

### In-Scope (Phase 1 to 6)
- **Phase 1: Basic Project Setup**: Clean structure, basic package structure under `src/`, configuration files (`requirements.txt`, `.gitignore`).
- **Phase 2: Data Loading & Validation**: A download script to fetch data from OpenML and save it locally in `data/`, a loading function with validation checks, and a preprocessing function to aggregate and merge frequency and severity datasets.
- **Phase 3: Basic EDA**: A notebook to check missing values, distributions of key variables, and exposure-adjusted frequency metrics.
- **Phase 4: Actuarial Baseline**: A Poisson GLM model with log exposure offset, fitted using `statsmodels` on an 80-20 train-test split.
- **Phase 5: Machine Learning Comparison**: A Random Forest Regressor trained with the `poisson` criterion, using `Exposure` as a sample weight, and comparing performance against the GLM using Mean Squared Error (MSE) and R-squared.
- **Phase 6: Workflow Integration & README Update**: An end-to-end script (`run_pipeline.py`) executing the entire flow, and an updated `README.md` showing final results.

### Out of Scope for MVP
- Claim severity modeling (beyond data loading and aggregation)
- Advanced feature engineering or feature selection
- Hyperparameter tuning or optimization
- Complex validation strategies (e.g., cross-validation, time-based splits)
- Model deployment, APIs, or interactive dashboards (e.g., Plotly dashboards)
- Deep actuarial theory or complex monitoring frameworks
- Detailed styling/tuning of the EDA notebook

---

## 3. Design Decisions Resolved
1. **Dataset Location**: Data is fetched from OpenML and stored locally in a git-ignored `data/` directory using a script (`src/data/download_data.py`).
2. **Directory Structure**: 3-part clean layout:
   - `src/data/` for data retrieval, loading, and validation.
   - `src/models/` for baseline modeling, ML modeling, and evaluation.
   - Package `__init__.py` files for importing modules.
3. **ML Model**: `RandomForestRegressor` with `criterion='poisson'`, predicting claim frequency (`ClaimNb / Exposure`) with `Exposure` as a sample weight.
4. **Comparison Metrics**: Mean Squared Error (MSE) and R-squared on the test set.
5. **Data Processing Separation**:
   - `load_raw_data()` loads the CSVs and validates structures.
   - `preprocess_data()` handles aggregation, left-joins, filling NaN values, and string formatting.

---

## 4. Recommended Implementation Order
1. **Setup Project Structure**: Create folders, confirm dependency packages, verify `.gitignore`.
2. **Implement Data Retrieval**: Write the OpenML downloader script.
3. **Build Data Loader & Preprocessing**: Implement `loader.py` with validation and cleaning.
4. **Complete EDA Notebook**: Link notebook to the loader and generate primary plots.
5. **Implement Poisson GLM Baseline**: Build the statistical baseline script using `statsmodels`.
6. **Implement Random Forest ML Model**: Create the ML training script using `scikit-learn`.
7. **Create Model Evaluation Utility**: Implement MSE & R-squared comparison logic.
8. **Build Pipeline Script**: Create `run_pipeline.py` to automate execution and save results.
9. **Finalize README.md**: Summarize findings and document results.
