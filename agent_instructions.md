# Agent Instructions — HR Attrition Predictor

## Project Overview
This is an AI-powered HR Employee Attrition Prediction web application.

- **Dataset**: IBM-style HR Attrition dataset (`data/dataset attrition.csv`) with 1,470 employee records and 31 features.
- **Problem type**: Binary classification — predict whether an employee will leave (`Attrition = Yes`) or stay (`Attrition = No`).
- **Target column**: `Attrition` (values: `Yes` / `No`, encoded as 1 / 0 during training).

## Architecture

```
train_model.py  →  models/model.pkl  ←  app.py
                          ↑
              Pipeline (preprocessor + best_clf)
              + metadata dict (feature lists, stats, options, scores)
```

## Key Design Decisions

1. **Dropped columns**: `Unnamed: 32` (empty), `Date_of_Hire` (string date, low signal), `Date_of_termination` (all NaN), `Status_of_leaving` (post-event leakage — only populated for leavers).
2. **Preprocessing**: `StandardScaler` for numerics, `OrdinalEncoder` for categoricals — both inside a `ColumnTransformer` so the pipeline is a single serialisable object.
3. **Model selection**: 5-fold stratified cross-validation on the train set; best ROC-AUC wins. Typically Gradient Boosting or Random Forest wins.
4. **Class imbalance**: Handled via `class_weight="balanced"` on applicable models and stratified splits.
5. **Gemini integration**: `gemini-3.6-flash` model via the `google-generativeai` SDK. The API key is read from `GEMINI_API_KEY` env var. App is fully functional without a key — AI panel is gracefully disabled.

## Feature Groups

| Group | Features |
|---|---|
| Numerical | Age, DistanceFromHome, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager, Leaves, Absenteeism |
| Categorical | BusinessTravel, Department, Gender, JobRole, MaritalStatus, OverTime, Higher_Education, Mode_of_work, Work_accident, Source_of_Hire, Job_mode |

## File Responsibilities

| File | Responsibility |
|---|---|
| `train_model.py` | Load data, preprocess, train 5 models, evaluate, save best pipeline + metadata |
| `app.py` | Load pipeline, render Streamlit UI, run inference, call Gemini API |
| `models/model.pkl` | joblib-serialised dict: pipeline + metadata (feature lists, stats, options, CV results) |
| `data/dataset attrition.csv` | Source dataset — never modified by the application |

## Extending the Project

- To add a new model: add it to the `MODELS` dict in `train_model.py` and re-run.
- To change the Gemini prompt: edit `get_gemini_explanation()` in `app.py`.
- To add SHAP explanations: install `shap` and call `shap.TreeExplainer` on `pipeline.named_steps["clf"]`.
