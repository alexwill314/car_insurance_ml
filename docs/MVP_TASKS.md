# Car Insurance Pricing MVP Tasks

This document contains step-by-step implementation tasks for Kilo to build the car insurance pricing MVP.

---

## Task 1: Basic Project Structure Setup
- **Goal**: Initialize the project directory structure, package configuration, and package entry points.
- **Affected files**:
  - `requirements.txt` (verify/update)
  - `.gitignore` (update)
  - `src/__init__.py` (create empty)
  - `src/data/__init__.py` (create empty)
  - `src/models/__init__.py` (create empty)
- **Implementation steps**:
  1. Inspect the existing `requirements.txt` file and make sure the packages `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `plotly`, `matplotlib`, `seaborn`, `jupyter` are specified (they are currently listed).
  2. Create the directories: `src/`, `src/data/`, and `src/models/` under the project root.
  3. Create an empty `__init__.py` file in each of these three directories to register them as Python modules.
  4. Edit `.gitignore` to ensure that the local data folder (`data/`) and any virtual environments or cache folders are ignored. E.g., add `data/` and `__pycache__/` to the file.
- **Acceptance criteria**:
  - The directories `src/`, `src/data/`, and `src/models/` exist.
  - Empty `__init__.py` files exist in each of these folders.
  - Running `python -c "import src.data; import src.models"` returns no errors.
  - The `data/` folder is listed in `.gitignore`.
- **Suggested checks**:
  - Execute: `python -c "import src.data; import src.models"` in the terminal.
- **Definition of Done**: Project folder structure created, verified, and committed.

---

## Task 2: Data Downloader Script
- **Goal**: Create a script that downloads the French Motor Third-Party Liability datasets from OpenML and saves them locally as CSVs.
- **Affected files**:
  - `src/data/download_data.py` (create new)
- **Implementation steps**:
  1. Create `src/data/download_data.py`.
  2. Use `sklearn.datasets.fetch_openml` to download:
     - Frequency dataset: OpenML data ID `41214`.
     - Severity dataset: OpenML data ID `41215`.
  3. Construct a local folder named `data/` in the project root if it does not exist.
  4. Extract the `.data` attribute from the fetch response (ensure it's a pandas DataFrame) and save both to CSV files:
     - Save frequency to `data/freMTPL2freq.csv`.
     - Save severity to `data/freMTPL2sev.csv`.
  5. Include basic logging/print output indicating when files are downloaded and saved.
- **Acceptance criteria**:
  - Running `python src/data/download_data.py` completes without error.
  - The files `data/freMTPL2freq.csv` and `data/freMTPL2sev.csv` are created and contain data.
- **Suggested checks**:
  - Run `python src/data/download_data.py` and inspect the `data/` directory to see if both CSVs are present and populated.
- **Definition of Done**: Download script implemented, successfully executed, files generated, and code committed.

---

## Task 3: Data Loader and Preprocessing Module
- **Goal**: Write the functions for loading raw CSVs with data validation, and for merging and preprocessing.
- **Affected files**:
  - `src/data/loader.py` (create new)
- **Implementation steps**:
  1. Create `src/data/loader.py`.
  2. Implement `load_raw_data(data_dir)`:
     - Read `freMTPL2freq.csv` and `freMTPL2sev.csv` using pandas.
     - Validate that both DataFrames are not empty.
     - Validate that the frequency dataset contains columns: `IDpol`, `ClaimNb`, `Exposure`, `VehGas`.
     - Validate that the severity dataset contains columns: `IDpol`, `ClaimAmount`.
     - Check that data types are correct: `ClaimNb` should be integer/numeric, `Exposure` and `ClaimAmount` should be float/numeric.
     - Raise a `ValueError` if validation checks fail.
     - Return `(freq_df, sev_df)`.
  3. Implement `preprocess_data(freq_df, sev_df)`:
     - Group severity dataset `sev_df` by `IDpol` and calculate the sum of `ClaimAmount` (to handle multiple claims per policy).
     - Merge `freq_df` and the aggregated severity DataFrame using a left-join on `IDpol`.
     - Fill missing `ClaimAmount` values with `0.0`.
     - Format categories: Strip single quotes from entries in the `VehGas` column (e.g., convert `'Regular'` to `Regular`, `'Diesel'` to `Diesel`) and cast it and other string columns to category type.
     - Return the merged DataFrame `df`.
- **Acceptance criteria**:
  - `load_raw_data()` raises ValueError if columns are missing or type validations fail.
  - `preprocess_data()` handles aggregation, left-joins, fills NaNs with 0, cleans categories, and returns a single DataFrame.
- **Suggested checks**:
  - Open a python terminal and run:
    ```python
    from src.data.loader import load_raw_data, preprocess_data
    freq, sev = load_raw_data("data")
    df = preprocess_data(freq, sev)
    print(df.shape, df["ClaimAmount"].isnull().sum())
    ```
- **Definition of Done**: Loader and preprocess functions implemented, tested via terminal, and committed.

---

## Task 4: Complete Initial EDA Notebook
- **Goal**: Hook the notebook into the data loader, complete visual checks, and document exploratory statistics.
- **Affected files**:
  - `notebooks/01_EDA.ipynb` (update)
- **Implementation steps**:
  1. Modify `notebooks/01_EDA.ipynb` to import and use the data loader functions from `src.data.loader`.
  2. Replace inline data downloading with:
     ```python
     from src.data.loader import load_raw_data, preprocess_data
     freq, sev = load_raw_data("../data")
     df = preprocess_data(freq, sev)
     ```
  3. Create standard EDA visualizations:
     - Plot distribution of `ClaimNb` and `Exposure`.
     - Plot frequency rates (Total ClaimNb / Total Exposure) aggregated by categories (e.g. `VehGas`, `Area`).
     - Check distributions of key variables (`VehPower`, `VehAge`, `DrivAge`, `BonusMalus`).
     - Plot missing values overview.
  4. Write short, clear text interpretations of the distributions and data quality observations in markdown cells.
- **Acceptance criteria**:
  - The notebook runs top-to-bottom without error.
  - Displays visualizations of distributions, frequency rates, and text interpretations.
- **Suggested checks**:
  - Run all cells in the notebook using Jupyter.
- **Definition of Done**: Notebook updated, run completely, saved with outputs, and committed.

---

## Task 5: Actuarial Baseline Model (Poisson GLM)
- **Goal**: Build and train the Poisson GLM baseline model using `statsmodels`.
- **Affected files**:
  - `src/models/baseline.py` (create new)
- **Implementation steps**:
  1. Create `src/models/baseline.py`.
  2. Implement an 80/20 train-test split function (random state = 42).
  3. Prepare features: Categorical variables (`Area`, `VehGas`, `Region`, `VehBrand`) should be one-hot/dummy encoded. Ensure the column names are consistent.
  4. Fit a Poisson GLM on the training set using `statsmodels.api.GLM`:
     - Target: `ClaimNb`
     - Exposure offset: `numpy.log(df['Exposure'])`
     - Independent variables: `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, plus dummy-encoded categorical columns.
     - Use `family=statsmodels.api.families.Poisson()`.
  5. Print the model summary showing parameters, standard errors, and p-values.
  6. Return the trained model object and the train/test datasets.
- **Acceptance criteria**:
  - Script successfully splits data and fits a Poisson GLM.
  - The model summary prints successfully showing statistical outputs.
- **Suggested checks**:
  - Run `python src/models/baseline.py` to verify it runs, fits, and displays coefficients.
- **Definition of Done**: Baseline Poisson GLM script written, tested, and committed.

---

## Task 6: Machine Learning Model (Random Forest with Poisson Criterion)
- **Goal**: Implement and train a Random Forest comparison model with Poisson criterion using `scikit-learn`.
- **Affected files**:
  - `src/models/ml_model.py` (create new)
- **Implementation steps**:
  1. Create `src/models/ml_model.py`.
  2. Implement a function to train the ML model on the train-test split:
     - Use `sklearn.ensemble.RandomForestRegressor` with `criterion='poisson'`.
     - Target variable: `ClaimNb` / `Exposure` (representing claim frequency).
     - Sample weight: `Exposure` (to weigh by duration of observation).
     - Set hyperparameters to ensure fast, stable training: `n_estimators=50`, `max_depth=8`, `min_samples_leaf=20`, `random_state=42`.
  3. One-hot encode the categorical variables consistently with the GLM baseline data preparation.
  4. Fit the model and save the fitted model as well as predictions.
- **Acceptance criteria**:
  - Script successfully prepares the dataset, trains the Random Forest model, and runs without memory or timeout issues.
- **Suggested checks**:
  - Execute the script and verify that the Random Forest trains within 2 minutes.
- **Definition of Done**: Random Forest script created, training verified, and code committed.

---

## Task 7: Model Evaluation and Comparison
- **Goal**: Calculate MSE and R-squared on the test set for both models, and format a comparison.
- **Affected files**:
  - `src/models/evaluation.py` (create new)
- **Implementation steps**:
  1. Create `src/models/evaluation.py`.
  2. Write functions to evaluate predictions against actual targets on the test set:
     - Test Target: `ClaimNb` / `Exposure` (claim frequency).
     - Calculate Mean Squared Error (MSE) and R-squared ($R^2$).
  3. Load predictions from the baseline Poisson GLM (calculate predicted frequency as GLM predictions divided by exposure, or use GLM's raw rate prediction).
  4. Load predictions from the Random Forest model (already outputs predicted frequency).
  5. Compute both metrics (MSE, R-squared) for both models.
  6. Output a formatted markdown table comparing the performance of the baseline GLM vs. the Random Forest ML model.
- **Acceptance criteria**:
  - Evaluation functions compute MSE and R-squared on test set for both models.
  - The script prints a clean markdown comparison table.
- **Suggested checks**:
  - Execute the script and verify that valid float values for MSE and R-squared are outputted.
- **Definition of Done**: Evaluation module implemented, tested, and committed.

---

## Task 8: Pipeline Runner Script
- **Goal**: Create an integration script `run_pipeline.py` that runs the entire workflow end-to-end and saves the comparison output.
- **Affected files**:
  - `run_pipeline.py` (create new in project root)
- **Implementation steps**:
  1. Create `run_pipeline.py` in the project root directory.
  2. Implement pipeline execution logic:
     - Step 1: Check if CSV files exist in `data/`. If not, call `download_data`.
     - Step 2: Load and preprocess raw data using `loader.py`.
     - Step 3: Perform train-test split.
     - Step 4: Fit the baseline Poisson GLM (Task 5) and generate test set predictions.
     - Step 5: Fit the Random Forest model (Task 6) and generate test set predictions.
     - Step 6: Evaluate both models and compute metrics (Task 7).
     - Step 7: Print the comparison table and save it to `data/results_comparison.md`.
- **Acceptance criteria**:
  - Running `python run_pipeline.py` executes the entire process from download to evaluation.
  - Generates the `data/results_comparison.md` file containing the comparison table.
- **Suggested checks**:
  - Run the entire script: `python run_pipeline.py` and inspect both the terminal output and the created results file.
- **Definition of Done**: Pipeline runner script completed, fully executed, output file verified, and code committed.

---

## Task 9: README Update
- **Goal**: Update the main repository README with business details, dataset info, MVP workflow, and results.
- **Affected files**:
  - `README.md` (update)
- **Implementation steps**:
  1. Open the existing `README.md`.
  2. Update Section 1 & 2 to details of the MVP scope.
  3. Update Section 5 ("Key Findings") with the output comparison table generated in `data/results_comparison.md`.
  4. Write a paragraph explaining the findings (e.g. which model performed better, R-squared values).
  5. Add a section describing Limitations of the MVP (e.g., no severity model, basic features, no hyperparameter tuning) and future steps.
- **Acceptance criteria**:
  - `README.md` updated with the actual results and limitations.
- **Suggested checks**:
  - Read through `README.md` to ensure it is clean and complete.
- **Definition of Done**: README updated with results, finalized, and committed.
