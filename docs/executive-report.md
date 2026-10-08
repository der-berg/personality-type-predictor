# Personality Type Predictor: Project Report

## Purpose and scope

The project predicts four dataset labels—Moderate, Resilient, Overcontroller,
and Undercontroller—from 19 questionnaire responses and three demographic
inputs. It combines exploratory data analysis, a reproducible training
pipeline, tracked model comparison, and a local Streamlit interface.

This is a supervised-learning demonstration, not a clinical assessment. The
labels are derived from questionnaire information. Predicting them well does
not independently validate the questionnaire or establish usefulness for
other populations. The selected items do not constitute the complete Big-Five
instrument. Item attribution and the external data source are in the [README](../README.md).

## Data inspection and cleaning

The raw dataset contained 19,719 rows and 23 columns, including the target.
Inspection identified three duplicate rows, 83 age values above 100, 24
missing gender values, and 100 missing hand values. One record had zero
responses to all 19 questionnaire items, outside the permitted response scale.

The [EDA notebook](../eda.ipynb) removes the all-zero record and complete
duplicates, leaving 19,715 rows. Age values above 100 are marked missing;
they are not interpreted as birth years without supporting evidence. Other
available fields are retained, but this does not prove that those responses
are correct. Missing demographic values remain missing at this stage.

Imputation is learned inside the training pipeline, not during whole-dataset
cleaning. Numeric features use median imputation and standardization;
categorical features use most-frequent imputation and one-hot encoding.
These transformations are fitted separately in each training fold.

| Target label | Records | Share |
|---|---:|---:|
| Moderate | 8,449 | 42.86% |
| Resilient | 6,162 | 31.26% |
| Overcontroller | 2,829 | 14.35% |
| Undercontroller | 2,275 | 11.54% |

Shares are rounded. Stratification preserves these proportions in the split
and cross-validation folds; it does not balance the classes or reweight errors.
Balanced class weights in Logistic Regression and Random Forest address
training error weights. Macro-F1 gives each class equal weight in evaluation.

## Modeling and champion selection

The [modeling notebook](../modeling.ipynb) reads the processed dataset,
reserves a stratified 20% holdout, and uses stratified five-fold
cross-validation on the remaining training data. A dummy baseline provides
context. Logistic Regression, Random Forest, and K-Nearest Neighbors each
receive a bounded 20-trial Hyperopt search. MLflow records parameters,
metrics, and fitted preprocessing-plus-classifier pipelines.

The saved notebook outputs report:

| Best tuned candidate | Mean CV macro-F1 | Fold standard deviation | Mean fit time (seconds) |
|---|---:|---:|---:|
| Random Forest | 0.786 | 0.009 | 0.476 |
| Logistic Regression | 0.784 | 0.007 | 0.298 |
| K-Nearest Neighbors | 0.744 | 0.008 | 0.046 |

Times are measurements from this notebook execution, not portable performance
guarantees. Fit time is training time, not application loading or inference time.
The standard deviation describes variation across folds, not across classes.

Logistic Regression is the champion. Random Forest's mean CV advantage is
small, while Logistic Regression is simpler to explain and has slightly less
fold variation in this run. Training out-of-fold Undercontroller F1 is 0.547
for Logistic Regression versus 0.535 for Random Forest; this illustrates a
precision–recall trade-off, not superiority on every class or a demonstrated
statistically significant difference. The selected Logistic Regression uses
`C = 1.0984035416423363` and `class_weight = "balanced"`.

The KNN tuning was a later completeness follow-up. The champion decision was
not reopened using the known holdout results. This chronology limits any
claim that the entire comparison was a prospectively sealed experiment.

### MLflow evidence and handoff

The saved tuning table links each best candidate to a logged run:

| Candidate | Historical run ID | Mean CV macro-F1 |
|---|---|---:|
| Random Forest | `0beee6fd95aa49b79915be4854286137` | 0.786 |
| Logistic Regression (champion) | `7d27288ce33e41d3b5490f8b04e9b07d` | 0.784 |
| K-Nearest Neighbors | `85c6a69fac5e4b93bc004cb4bbb74f52` | 0.744 |

These IDs document the saved reproduction, not downloadable models supplied
by this repository. Each run records parameters, CV metrics, and the complete
fitted pipeline. A fresh execution creates new IDs: verify the intended
champion in the local MLflow UI and copy its new ID to `model_loader.py`.
The notebook recovers the ID of the previously selected configuration;
it does not make a new model-selection decision automatically.

## Fixed champion: holdout findings

The reserved set contains 3,943 records. The fixed champion achieved accuracy
0.824, macro-F1 0.784, and balanced accuracy 0.819 in the saved execution.

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Moderate | 0.909 | 0.720 | 0.804 | 1,690 |
| Overcontroller | 0.719 | 0.855 | 0.781 | 566 |
| Resilient | 1.000 | 1.000 | 1.000 | 1,232 |
| Undercontroller | 0.455 | 0.699 | 0.551 | 455 |

Undercontroller predictions require particular caution: 318 of 699 predicted
Undercontrollers are correct, while 318 of 455 actual Undercontrollers are
recognized. Those percentages have different denominators. There are 381
false positives and 137 false negatives for this class. The perfect Resilient
score is a dataset-specific finding, not proof of perfect real-world recognition.

The test results describe limitations of the already selected model. They
must not be used to tune parameters or select another champion while still
calling this same holdout an independent final test. Class imbalance is a
relevant concern, but the observed errors cannot be attributed to it alone.

## Application and reproducibility

The English Streamlit app provides 19 discrete response sliders and three
demographic fields. Neutral questionnaire defaults allow a quick example;
users are told to replace them with their own answers. Accepted ages are
13–100, an application scope restriction rather than a claim that every age
in this interval has been separately validated. Invalid ages stop prediction.

After validation, the app forms one raw input row and loads the saved fitted
pipeline through `model_loader.py`. Cached loading avoids repeated model
retrieval. Prediction applies the learned transformations; it does not
retrain the model, run the modeling notebook, or learn a new scale from an
individual input. A technical view exposes the submitted values for checking.

The repository intentionally excludes datasets, virtual environments,
MLflow databases, and trained artifacts. Follow the README to obtain the
external data, create a Python 3.12 environment, run EDA and modeling, and
set the newly generated champion run ID in the loader. An empty initial ID
is a deliberate safeguard, not an included ready-to-use model.

## Limitations and interpretation

- Performance depends on the supplied dataset and its label-generation process.
- Scores summarize one documented experimental setup; uncertainty across new
  populations or alternative splits has not been comprehensively quantified.
- Imputation retains records but may alter their feature relationships.
- Neutral defaults are examples, not a population-average personality result.
- The interface has usability and accessibility checks, but no complete
  accessibility-conformance claim is made.
- Neither the app nor its brief type descriptions should support diagnostic,
  employment, or other consequential decisions.

A useful next step would be evaluation on an independently collected dataset
with compatible inputs and clearly documented target labels. This would test
transferability rather than repeatedly tuning against the known holdout.
It is a proposal for future work, not an evaluation already performed.

The notebooks are the primary evidence for cleaning and modeling decisions;
this report is a readable synthesis of their saved results.
