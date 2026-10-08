# Personality Type Predictor: Executive Summary

## What the project delivers

A local English Streamlit app predicts one of four dataset personality-type
labels from 19 questionnaire responses and three demographic inputs. The
project includes EDA, leakage-aware preprocessing, model tuning, MLflow
tracking, and an explicit handoff of the selected trained pipeline.
It is a demonstration, not a diagnosis or a complete standard Big-Five test.

## Main findings and decisions

**Cleaning.** The raw data contain 19,719 records. Removing three complete
duplicates and one all-zero questionnaire record leaves 19,715. The 83 age
values above 100 become missing; missing demographic values are retained
for training-pipeline imputation. Retaining a record does not establish the
validity of all its remaining responses.

**Class balance.** Moderate accounts for 42.86% of cleaned records and
Resilient for 31.26%; Overcontroller and Undercontroller account for 14.35%
and 11.54%. Stratification preserves proportions. Balanced class weights
change training error weights; macro-F1 gives each class equal evaluation
weight. These are distinct measures, not interchangeable forms of balancing.

**Model comparison.** Three model families receive 20 Hyperopt trials each
with stratified five-fold training cross-validation. Their best mean CV
macro-F1 scores are Random Forest 0.786, Logistic Regression 0.784, and KNN
0.744. Logistic Regression is selected for explainability and its competitive
CV results, including slightly higher training out-of-fold Undercontroller
F1. The small mean difference does not prove statistical superiority. KNN
tuning was a later completeness follow-up; the known holdout was not used
to reopen the champion decision.

**Final evaluation.** The fixed champion achieves accuracy 0.824 and
macro-F1 0.784 on 3,943 holdout records. Undercontroller precision is 0.455,
recall 0.699, and F1 0.551. Thus fewer than half of predictions of this label
are correct, although approximately 70% of its actual members are recognized.
Resilient F1 is 1.000 in this dataset, not evidence of perfect generalization.
The holdout is used to report limitations, not to tune or select a new model.

**Application.** Inputs remain raw until the fitted pipeline applies its
stored imputation, scaling, and encoding. The app never trains a new model.
Ages outside 13–100 are rejected before prediction. Sliders, clear labels,
explicit errors, and a technical input view support understandable operation.
Neutral defaults are demonstration inputs, not a validated average profile.

## Reproduction and evidence

Datasets, models, MLflow stores, and virtual environments are not published.
Follow the [README](../README.md) for data access, setup, notebook execution,
and the new run-ID handoff. [EDA](../eda.ipynb) and
[modeling](../modeling.ipynb) hold the saved evidence;
the [project report](executive-report.md) explains the decisions in more detail.

**MLflow evidence.** The saved champion run is
`7d27288ce33e41d3b5490f8b04e9b07d` (Logistic Regression). It links the fitted
pipeline and training-CV metrics; the notebook reports holdout macro-F1 0.784.
This is historical evidence, not a supplied model. Reproduction creates a
new ID to verify and hand to the app.

**Next step.** Evaluate transferability on independent data with compatible
inputs and documented labels. This is proposed future work, not a completed test.

For demonstration only, not for clinical or other consequential decisions.
