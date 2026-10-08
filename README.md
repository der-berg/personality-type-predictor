# Personality Type Predictor

## Overview

This project predicts one of four personality-type labels—`Moderate`,
`Resilient`, `Overcontroller`, or `Undercontroller`—from 19 questionnaire
responses and three demographic fields (`age`, `gender`, and `hand`). It
demonstrates a supervised-learning workflow from exploratory data analysis
(EDA) through model comparison, MLflow tracking, and a local Streamlit app.

The supplied, reduced Big-Five dataset contains one row per respondent:
19 response scores, age, gender, writing hand, and the target label. Obtain
it from the linked data folder in the setup instructions below.

The selected champion is Logistic Regression. Its saved holdout macro-F1
is **0.784** (accuracy **0.824**, 3,943 test rows). Macro-F1 is the primary
comparison metric because it gives every class equal weight. The model
selection section below explains the trade-off against Random Forest.

To use the app, review the 19 response sliders and enter your personal
details, then select **Show result** to see the predicted label and its limits.

The prediction is an output of a model trained on this course dataset. It is
not a clinical assessment or a diagnosis. The target labels are derived from
personality-questionnaire information; strong performance on this dataset
does not establish validity for other populations or uses.

The 19 questionnaire statements used by the app match English items from the
[IPIP 50-item Big-Five Factor Markers](https://ipip.ori.org/ChineseAB5CFactorMarkers.htm).
IPIP [permits public use of its items](https://ipip.ori.org/newPermission.htm).
This attribution applies to the item text, **not** to redistribution of the
course dataset or its codebook. The app uses a selected subset of items and
must not be described as the full 50-item instrument.

The app's brief descriptions explain the four labels used in this dataset;
they are not independently validated interpretations of an individual person.
The dataset uses a reduced set of questionnaire items, so the app does not
implement a complete standard Big-Five assessment.

## Setup

The instructions below target Windows PowerShell and Python 3.12. Run commands
from the repository root, the folder containing `eda.ipynb`, `modeling.ipynb`,
and `app.py`.

### Clone the repository

```powershell
git clone https://github.com/der-berg/personality-type-predictor.git
cd personality-type-predictor
```

### 1. Obtain the course data

The repository does not contain the raw or processed dataset, the course
codebook, MLflow tracking files, or trained models. Download `data.csv` and
`codebook.txt` from the [course data folder](https://drive.google.com/drive/folders/1KhwTPAG07EdaENW_XX9nVvKhC-DP1Ags?usp=sharing), then place them at
exactly these paths:

```text
data/data.csv
data/codebook.txt
```

Create the `data` folder first if it does not exist:

```powershell
New-Item -ItemType Directory -Force data
```

Access to the course folder and permission to redistribute its contents are
different questions. This repository links to the source; it does not grant
redistribution rights. Before submission, verify that reviewers can open the
data link without relying on the author's session.

### 2. Create the environment and install dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Using `.venv\Scripts\python.exe` directly avoids relying on whichever global
Python version happens to be selected in the terminal. In VS Code, select
this same interpreter as the Python interpreter and Jupyter kernel.

**Windows setup troubleshooting:** Use a short project path. A deeply nested
path can exceed Windows file-path limits while installing dependencies.
If `py -3.12` cannot find an already installed Python 3.12 interpreter, use
its verified executable path instead; do not substitute another Python version:

```powershell
& '<PATH_TO_PYTHON_3_12_EXE>' -m venv .venv
```

If installation stalls while writing the default pip cache, retry with a
project-local cache (this does not change the required package versions):

```powershell
.\.venv\Scripts\python.exe -m pip install --cache-dir .\pip-cache --disable-pip-version-check --no-input --timeout 15 --retries 0 -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Keep `pip-cache/` local and out of Git. Do not disable certificate checks
or change system-wide settings to bypass an unexplained installation error.

### 3. Run EDA and create the processed dataset

Open `eda.ipynb` in VS Code, select the `.venv` kernel, and use **Restart & Run
All**. The notebook reads `data/data.csv`, documents the raw-data checks and
cleaning decisions, and writes `data/data_processed.csv`. Confirm that the
final validation cell succeeds and the processed file exists.

The EDA notebook does not fit the model's imputation, scaling, or categorical
encoding on the full dataset. Those learned transformations belong to the
training pipeline.

### 4. Train and compare models

Open `modeling.ipynb` with the same kernel and run all cells from the top. It
reads **only** `data/data_processed.csv`, creates a stratified holdout split,
and compares a dummy baseline with Logistic Regression, Random Forest, and
K-Nearest Neighbors. The three candidate model families each receive a
limited 20-trial Hyperopt search with stratified five-fold cross-validation
on the training data. The primary comparison metric is macro-averaged F1.

Each tuned run logs its parameters, scores, and fitted preprocessing-plus-
classifier pipeline to the local MLflow store. The notebook evaluates the
already selected champion once on the reserved holdout set. It does not use
that set to choose among candidate models.

This run creates a local `mlflow.db` and model-artifact directories; they are
not supplied by Git and must remain outside the public repository.

### 5. Inspect MLflow and connect the app to the new run

After the modeling notebook completes, open a new PowerShell terminal in the
repository root and start the MLflow UI against **the same database**:

```powershell
$mlflowDb = (Resolve-Path .\mlflow.db).Path.Replace('\', '/')
.\.venv\Scripts\mlflow.exe ui --backend-store-uri "sqlite:///$mlflowDb" --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and inspect the
`personality-type-predictor` experiment. Compare the training-CV macro-F1
scores and the trade-offs described in `modeling.ipynb`. This project uses the
previously justified Logistic Regression configuration as its champion;
running the notebook again does not silently select a different model.

MLflow assigns new run IDs on a new machine. Copy the **current champion run
ID printed by the completed notebook** into `CHAMPION_RUN_ID` in
`model_loader.py`. Verify that this ID refers to the intended run in the
MLflow UI before launching the app. The source file deliberately starts with
an empty ID and refuses to load a model until this handoff is completed.
Changing to a different champion requires a new model-selection decision and
corresponding notebook and loader changes; replacing the ID alone is not a
general model-selection mechanism.

### 6. Run the Streamlit app

In another PowerShell terminal at the repository root, run:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open the local URL reported by Streamlit. The app collects the 19 raw response
values and the three demographic inputs, loads the fitted pipeline from
MLflow, and displays a predicted type. It does not train a model or duplicate
the pipeline's imputation, scaling, or encoding. The app limits accepted ages
to 13–100 and rejects invalid entries rather than predicting from a hidden
replacement value.

## Model selection and current local evidence

The saved notebook outputs from the controlled local reproduction report
the following **training cross-validation** macro-F1 scores for the best
of the 20 trials per family:

| Candidate | Mean CV macro-F1 |
|---|---:|
| Random Forest | 0.785807 |
| Logistic Regression | 0.784427 |
| K-Nearest Neighbors | 0.744169 |

Logistic Regression was selected despite the small CV-score advantage for
Random Forest. The documented reasons include explainability and the
class-specific out-of-fold comparison, especially for `Undercontroller`.
The fixed champion's holdout macro-F1 was 0.784 on 3,943 rows in that local
run. These results should be compared with the outputs of a new run when
reproducing the project.

The saved notebook outputs show a verified local execution. Its MLflow run
IDs are historical evidence, not models supplied by this repository. A new
execution creates its own tracking store and IDs; complete the loader handoff
described above. Diagnostic and artifact-download messages are omitted from
the saved presentation outputs, without changing the code or result tables.

Class performance is uneven. In the controlled local reproduction,
`Resilient` reached holdout F1 of 1.000, whereas `Undercontroller` reached
0.551. The target labels were derived from questionnaire information in
this course dataset. These results do not independently validate the
questionnaire or establish reliability for other populations. The app is a
course demonstration, not an instrument for consequential decisions.

## Documentation

The [project report](docs/executive-report.md) explains the main findings,
decisions, and limitations. The [executive summary](docs/executive-summary.md)
and its [printable PDF](docs/executive-summary.pdf) provide a compact overview.
The notebooks remain the primary evidence for the reported results.

## Project Structure

```text
personality-type-predictor/
├── .gitattributes             # Keep text-file line endings consistent.
├── .gitignore                 # Exclude local data, models, environments, and caches.
├── .streamlit/
│   └── config.toml            # Set the blue input-widget accent; no secrets.
├── README.md                  # Setup, reproduction steps, results, and limitations.
├── app.py                     # Collect questionnaire inputs and display predictions.
├── data/
│   └── .gitkeep               # Keep the local data folder in the repository.
├── eda.ipynb                  # Inspect and clean raw data; create the processed dataset.
├── docs/
│   ├── executive-report.md    # Findings, model selection, and limitations.
│   ├── executive-summary.md   # Compact project overview.
│   └── executive-summary.pdf  # Printable executive summary.
├── modeling.ipynb             # Compare and tune models; track runs in MLflow.
├── model_loader.py            # Load the selected pipeline from the local MLflow store.
├── requirements.txt           # Pin Python package versions used by the project.
└── tests/
    └── test_app_validation.py # Check input validation and safe app behavior.
```

The data folder contains only `.gitkeep` in this repository. After following
the data-download instructions, it also contains the local `data.csv` and
`codebook.txt`; `eda.ipynb` creates `data_processed.csv`. These course files
and the generated MLflow store and artifacts are excluded from Git.
