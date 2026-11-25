# Sales Prediction

**Sales Prediction** is a small Python machine-learning project that trains an XGBoost regression model to predict item outlet sales from a tabular sales dataset.

The current implementation is a single-script, reproducible experiment:

1. Load `sales_data.csv` from an absolute Windows path.
2. Encode two identifier columns as integers.
3. Separate the target and an unused row-index column from the feature matrix.
4. Split the data into training and testing subsets.
5. Train `XGBRegressor`.
6. Print training and testing R-squared (`R²`) scores.

> **Dataset status:** `sales_data.csv` is referenced by the project but is not present in the repository at the time of this documentation. The program cannot run until a compatible CSV is placed at `D:\sales_prediction\sales_data.csv`.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Contents](#repository-contents)
- [Machine Learning Objective](#machine-learning-objective)
- [Data Contract](#data-contract)
- [Application Flow](#application-flow)
- [Evaluation](#evaluation)
- [Environment and Dependencies](#environment-and-dependencies)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Example Console Output](#example-console-output)
- [Known Behavior and Limitations](#known-behavior-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Testing and Validation](#testing-and-validation)
- [Recommended Improvements](#recommended-improvements)
- [Development Notes](#development-notes)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project uses supervised regression. Each row represents an observation containing item/outlet information and associated features. The model learns to predict the continuous value in `Item_Outlet_Sales`.

The implementation deliberately uses default XGBoost hyperparameters. It does not perform:

- Hyperparameter tuning
- Cross-validation
- Feature scaling
- Imputation
- One-hot encoding
- Missing-value validation
- Outlier removal
- Train-only feature engineering
- Model persistence
- Batch or real-time prediction
- A command-line interface or web/API interface
- Automated tests

The script imports Matplotlib and Seaborn, but its current active code does not create or display plots.

## Repository Contents

| Path | Purpose |
|---|---|
| `main.py` | End-to-end data loading, preprocessing, model training, prediction, and R² reporting |
| `requirements.txt` | Pinned Python package versions used by the project |
| `.venv/` | Optional local Python virtual environment; ignored by Git through `.venv/.gitignore` |
| `sales_data.csv` | Required input dataset; currently missing |
| `README.md` | Project documentation |

The repository does not currently contain source tests, a `.gitignore` at the repository root, a packaging configuration, or license files.

## Machine Learning Objective

The modeled relationship is:

```text
Item_Outlet_Sales = f(dataset features, with identifier encoding)
```

- **Task type:** Supervised machine-learning regression
- **Model:** XGBoost `XGBRegressor`
- **Target:** `Item_Outlet_Sales`
- **Features:** Every CSV column except `Item_Outlet_Sales` and `s_no`
- **Identifier preprocessing:** Label encoding for `Item_Identifier` and `Outlet_Identifier`
- **Train/test split:** 80% training, 20% testing
- **Random seed:** `2`
- **Primary metric:** R-squared

R-squared is calculated with `sklearn.metrics.r2_score`. It measures the proportion of target variance explained by predictions:

- `1.0` indicates a perfect fit for the available data.
- `0.0` indicates performance no better than predicting the target mean.
- Negative values indicate predictions worse than the target mean.

## Data Contract

Before running the script, provide `D:\sales_prediction\sales_data.csv`.

The dataset must include at least these columns:

| Column | Required by | Role | Preprocessing |
|---|---|---|---|
| `Item_Identifier` | `main.py` | Categorical identifier | `LabelEncoder.fit_transform` |
| `Outlet_Identifier` | `main.py` | Categorical identifier | `LabelEncoder.fit_transform` |
| `Item_Outlet_Sales` | `main.py` | Continuous regression target | No transformation |
| `s_no` | `main.py` | Row identifier/unused field | Dropped from features |

All remaining columns are retained as model features. XGBoost must receive numeric-compatible values after loading. Categorical columns other than the two identifier columns are not automatically encoded by this implementation.

### Important data assumptions

- `Item_Identifier` and `Outlet_Identifier` exist exactly with the capitalization shown above.
- `Item_Outlet_Sales` exists exactly with the capitalization shown above.
- `s_no` exists exactly with the capitalization shown above.
- CSV parsing uses the pandas default comma delimiter.
- The header row exists and is used as column names.
- Numeric feature columns are numeric in the CSV or can be interpreted as numeric by pandas.
- The target contains no missing values that would prevent metric calculation.
- The identifier columns contain values that `LabelEncoder` can process.
- The file path is valid on the Windows system used to run the script.

Missing values are not removed or imputed. Incompatible string categories or missing values may cause model training or pandas/scikit-learn/XGBoost compatibility errors.

## Application Flow

The script is executed from top to bottom.

```text
Import libraries
      |
      v
Read D:\sales_prediction\sales_data.csv
      |
      v
Encode Item_Identifier and Outlet_Identifier
      |
      v
Create X = all columns except Item_Outlet_Sales and s_no
Create y = Item_Outlet_Sales
      |
      v
Split 80% train / 20% test (random_state=2)
      |
      v
Instantiate XGBRegressor with default settings
      |
      v
Fit model on training data
      |
      v
Predict training and test targets
      |
      v
Print two R-squared values
```

### Identifier encoding

`LabelEncoder` assigns an integer to every unique value:

```python
encoder = LabelEncoder()
dataframe["Item_Identifier"] = encoder.fit_transform(
    dataframe["Item_Identifier"]
)
dataframe["Outlet_Identifier"] = encoder.fit_transform(
    dataframe["Outlet_Identifier"]
)
```

The same encoder instance is reused twice. There is no encoding leakage check, no stored encoder, and no support for unseen identifiers. Encoding is fitted on the complete dataset before the split, so category mappings are influenced by all rows. For an inference API or saved model, the encoder should be fitted on training data only and persisted with the model.

Note that `LabelEncoder` imposes an ordinal integer representation. It does not add the semantic meaning of category values, and XGBoost may interpret relationships as though some integer values are closer than others.

### Feature matrix and target

```python
x = dataframe.drop(["Item_Outlet_Sales", "s_no"], axis=1)
y = dataframe["Item_Outlet_Sales"]
```

- `axis=1` means pandas drops columns.
- `x` is the complete remaining feature matrix.
- `y` is the one-dimensional sales target.
- `s_no` is removed because the script treats it as an unused identifier.

### Train/test split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=2,
)
```

- Training features: approximately 80% of rows
- Testing features: approximately 20% of rows
- Training target: matching 80% of target rows
- Testing target: matching 20% of target rows
- Row selection: randomized
- Reproducibility: `random_state=2`

The split is not stratified because this is a regression task.

### Model training

The script creates `XGBRegressor()` with all library defaults. The model receives `x_train` and `y_train` through:

```python
model.fit(x_train, y_train)
```

No custom learning rate, tree depth, number of estimators, subsampling value, regularization parameter, early stopping configuration, or objective is specified.

### Predictions

The model generates continuous predicted sales values:

```python
train_data_score = model.predict(x_train)
test_data_score = model.predict(x_test)
```

The output names `train_data_score` and `test_data_score` represent predicted values, not scores. The resulting values are passed to `r2_score` for evaluation.

## Evaluation

The script prints exactly two values:

```python
print(metrics.r2_score(y_train, train_data_score))
print(metrics.r2_score(y_test, test_data_score))
```

No label identifies them in the source output. In code order:

1. Training R²
2. Testing R²

### Interpreting overfitting

The source comment states that the model may be heavily overfit. Overfitting is likely when:

- Training R² is substantially higher than test R².
- Performance is strong on familiar rows but weak on unseen rows.

This is a hypothesis to verify by comparing the actual metrics. R² alone cannot establish overfitting conclusively, and it does not measure absolute sales error. A comprehensive evaluation should include MAE, RMSE, residual plots, cross-validation, and held-out validation.

## Environment and Dependencies

The repository includes a local virtual environment created with Python 3.13.7. The `.venv/pyvenv.cfg` record identifies:

- Python executable: `C:\Users\praty\AppData\Local\Programs\Python\Python313\python.exe`
- Virtual environment: `D:\sales_prediction\.venv`
- `include-system-site-packages`: `false`

The project has been provisioned in a Python 3.13 environment, but the CSV and runtime package installation should be verified before use.

The pinned dependencies are:

| Package | Purpose |
|---|---|
| `pandas` | CSV loading and tabular data operations |
| `numpy` | Numerical array support |
| `matplotlib` | Visualization library dependency |
| `seaborn` | Statistical visualization library dependency |
| `scikit-learn` | Train/test splitting, label encoding, and R² metric |
| `xgboost` | Gradient-boosted regression model |
| `scipy` | Scientific and numerical support used by the ecosystem |
| `cloudpickle`, `joblib`, `threadpoolctl` | Serialization/job support and parallel tooling |
| `cycler`, `kiwisolver`, `fonttools`, `pyparsing`, `packaging` | Plotting and packaging support |
| `pillow` | Image support for plotting |
| `python-dateutil`, `tzdata` | Date and time support |
| `narwhals` | DataFrame compatibility support |

Exact pinned package versions are maintained in `requirements.txt`.

## Installation

Run commands in PowerShell from the repository root.

### Option 1: Use the existing virtual environment

```powershell
cd D:\sales_prediction
.\.venv\Scripts\Activate.ps1
python --version
python -m pip --version
python main.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This execution-policy change applies only to the current PowerShell process.

### Option 2: Create a new virtual environment

```powershell
cd D:\sales_prediction
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned versions may not support every Python version or operating-system architecture. Check package compatibility if an installation is moved to another machine.

For a virtual environment located elsewhere, use its own activation command and set `DATA_PATH` only if the script is updated to use environment-based configuration.

## Running the Project

1. Place the CSV at:

   ```text
   D:\sales_prediction\sales_data.csv
   ```

2. Confirm required columns are present.
3. Activate the environment.
4. Run:

   ```powershell
   python main.py
   ```

The program prints two R² values and then exits successfully if data loading, preprocessing, training, and prediction all succeed.

The script has no CLI arguments. Do not pass a filename, test size, or random seed on the command line; these values are fixed in `main.py`.

## Configuration

All configuration is hard-coded in `main.py`.

| Setting | Current value | Location |
|---|---|---|
| Dataset path | `D:\sales_prediction\sales_data.csv` | `pd.read_csv(...)` |
| Encoded columns | `Item_Identifier`, `Outlet_Identifier` | `LabelEncoder` calls |
| Removed features | `s_no` | `dataframe.drop(...)` |
| Target | `Item_Outlet_Sales` | `y = ...` |
| Test size | `0.2` | `train_test_split(...)` |
| Random state | `2` | `train_test_split(...)` |
| Model | `XGBRegressor()` | Default constructor |
| Metric | R-squared | `metrics.r2_score(...)` |

The absolute path is machine-specific. Moving the project to another directory or operating system requires updating the path unless the script is refactored to use a relative path or environment variable.

## Example Console Output

```text
0.9142
0.5897
```

The values above are illustrative only. The first value is training R² and the second is testing R². Actual values depend on the dataset, software versions, system configuration, and XGBoost behavior.

The source does not print:

- Dataset dimensions
- Column names
- Missing-value counts
- Feature names
- Model parameters
- Training duration
- MAE or RMSE
- Predictions
- Feature importance
- Residual statistics

Some of these operations exist as commented examples in `main.py`.

## Known Behavior and Limitations

### Dataset is missing

There is no `sales_data.csv` in the repository. The code has a hard-coded path and cannot run in a fresh clone until the CSV is supplied.

### Absolute Windows path

The path `D:\sales_prediction\sales_data.csv` will not work unchanged on another drive, another user account, or a non-Windows system.

### Encoding all categories before splitting

Label encoders are fitted on the entire dataframe before train/test division. This is not a strict leakage of target values, but category information from the test data participates in identifier encoding.

### No missing-value handling

The script does not inspect or fill null values in active code.

### No categorical handling beyond identifiers

Additional text columns in `x` are passed to XGBoost or the underlying data conversion layer without explicit encoding.

### Potentially high overfitting risk

Default XGBoost settings can memorize training patterns. The comment in `main.py` explicitly identifies possible overfitting, but the script does not calculate the difference or stop execution based on it.

### R² is not sufficient for production validation

R² does not expose absolute error, residual distribution, prediction bias, or outlier performance.

### No model saving

The `model` variable is in memory only. It is discarded when the process ends.

### No repeatable prediction interface

The script cannot load or transform external new records with separate saved preprocessing objects.

### No visualizations

Although plotting packages are imported, no active plotting code is present.

### No tests

There is no test suite or CI configuration.

### Reproducibility

The split is reproducible through `random_state=2`, but full reproducibility can still require controlling thread counts and library/model parameters. The model constructor uses default XGBoost settings.

## Troubleshooting

### `FileNotFoundError`

Confirm the data file exists exactly at `D:\sales_prediction\sales_data.csv`. If located elsewhere, place it at that path or update `pd.read_csv(...)`.

### `KeyError: 'Item_Outlet_Sales'`

The CSV does not contain the target column with the exact expected name. Inspect headers and rename or update the source.

### `KeyError: 'Item_Identifier'`, `KeyError: 'Outlet_Identifier'`, or `KeyError: 's_no'`

A required column is missing or has different capitalization/spacing. Rename the CSV column or update the script consistently.

### Encoding error

`LabelEncoder.fit_transform` requires a one-dimensional column containing compatible values. Check for malformed CSV parsing or mixed structures.

### Training error caused by strings or objects

XGBoost/model input must be numeric-compatible. Explicitly encode all remaining categorical feature columns or drop unsupported columns and retrain.

### Missing package

Activate `.venv` or install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If the pinned package set fails on a new Python version, create a compatible environment instead of mixing packages across versions.

### `ModuleNotFoundError`

Confirm the virtual environment is active. In PowerShell, the prompt should show `(.venv)`. Alternatively run:

```powershell
.\.venv\Scripts\python.exe main.py
```

### Test R² is much lower than training R²

This indicates possible overfitting or distribution differences between subsets. Add cross-validation, regularization, controlled hyperparameters, and stronger data preprocessing before reporting production readiness.

## Testing and Validation

No automated tests are included. A manual validation sequence is:

1. Verify `sales_data.csv` exists.
2. Run `python main.py`.
3. Confirm it prints exactly two numeric R² values.
4. Run it again and verify the split/reported training result is consistent.
5. Compare training and test R² values.
6. Inspect missing values and feature types before trusting the metric.

Useful diagnostic commands in a Python or notebook session:

```python
print(dataframe.head())
print(dataframe.tail())
print(dataframe.describe())
print(dataframe.shape)
print(dataframe.info())
print(dataframe.isnull().sum())
print(dataframe.isnull().sum().sum())
```

These statements are commented out in `main.py` and can be restored temporarily for debugging.

## Recommended Improvements

A production-oriented implementation should add:

- Relative or command-line dataset path
- Explicit input and target schema validation
- Missing-value reporting and imputation
- One-hot or target-safe encoding for categorical features
- Fit encoders only on training data
- Pipeline-based preprocessing plus model persistence
- Cross-validation
- Hyperparameter tuning
- Learning curves
- MAE, RMSE, and residual analysis
- Feature-importance reporting
- Reproducible random-number controls
- Model and encoder serialization
- A prediction function that validates unseen input
- Unit and integration tests
- Requirements grouped by runtime and development usage
- Optional argument parsing
- Logging instead of raw print statements
- Visualization of residuals and feature importance
- Dataset documentation and provenance
- A license and contribution guide

## Development Notes

The source is intentionally small and procedural. It can be useful as a learning example for:

- Loading CSV data with pandas
- Encoding categorical identifier columns
- Splitting a regression dataset
- Training XGBoost
- Calculating R² with scikit-learn

If the script is expanded, keep preprocessing and model steps in a scikit-learn `Pipeline` or equivalent abstraction so training and prediction use identical transformations.

Recommended repository layout for future growth:

```text
sales_prediction/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- src/
|   |-- preprocessing.py
|   |-- train.py
|   `-- predict.py
|-- tests/
|-- main.py
|-- requirements.txt
`-- README.md
```

This layout is a recommendation, not the current implementation.

## Contributing

Contributions should preserve deterministic behavior unless the change explicitly improves it.

Before opening a change:

1. Document the dataset requirements.
2. Add or update tests for preprocessing and model behavior.
3. Add command-line or environment-based configuration instead of another absolute path.
4. Avoid fitting preprocessors on test data.
5. Report both statistical metrics and the data split used.
6. Pin dependency changes in `requirements.txt`.
7. Do not commit credentials, private datasets, or large generated artifacts.

## License

No license file is currently included. No license terms are stated for reuse or distribution. Add an appropriate `LICENSE` file before treating the repository as publicly reusable.
