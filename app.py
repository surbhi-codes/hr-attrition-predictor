"""
app.py — HR Attrition Predictor
--------------------------------
Streamlit web application that:
  1. Loads the trained ML pipeline from models/model.pkl
  2. Accepts HR employee feature inputs via sidebar widgets
  3. Predicts employee attrition probability
  4. Provides an AI-powered explanation via Google Gemini 3.6 Flash
"""

import os
import time
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ── Environment & config ──────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model metadata ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "model.pkl")

@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Please run `python train_model.py` first.",
            icon="🚨",
        )
        st.stop()
    return joblib.load(MODEL_PATH)

meta = load_model()
pipeline           = meta["pipeline"]
NUM_FEATURES       = meta["numerical_features"]
CAT_FEATURES       = meta["categorical_features"]
CAT_OPTIONS        = meta["categorical_options"]
NUM_STATS          = meta["numerical_stats"]
BEST_MODEL         = meta["best_model_name"]
TEST_AUC           = meta["test_roc_auc"]
TEST_ACC           = meta["test_accuracy"]
ALL_RESULTS        = meta["all_results"]

# ── Helper: Gemini explanation ────────────────────────────────────────────────
def get_gemini_explanation(employee_data: dict, prediction: str, probability: float) -> str:
    """Call Gemini 3.6 Flash to generate a human-readable attrition analysis."""
    if not GEMINI_API_KEY:
        return (
            "**Gemini AI explanation unavailable** — set your `GEMINI_API_KEY` "
            "in the `.env` file or as an environment variable."
        )
    client = genai.Client(api_key=GEMINI_API_KEY)

    emp_details = "\n".join(
        f"  - {k}: {v}" for k, v in employee_data.items()
    )
    prompt = f"""
You are an expert HR analytics consultant. Analyse the following employee profile
and the machine learning model's attrition prediction, then provide a clear,
actionable explanation.

**Employee Profile:**
{emp_details}

**ML Prediction:** {prediction}
**Attrition Probability:** {probability:.1%}

Please provide:
1. **Key Risk Factors** - the top 3-4 factors from this profile most likely driving
   the attrition risk, with a brief explanation for each.
2. **Protective Factors** - any factors that reduce attrition risk.
3. **Recommended HR Actions** - 3 concrete, practical steps HR/management can take
   to reduce attrition risk for this employee.
4. **Summary** - a single concise paragraph summarising the overall situation.

Keep the tone professional and constructive. Use markdown formatting.
"""

    max_retries = 3
    retry_delay = 2  # seconds between attempts

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            error_str = str(exc)
            is_last = attempt == max_retries
            # Retry only on transient server-side errors (5xx / UNAVAILABLE)
            if not is_last and ("503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str):
                time.sleep(retry_delay)
                continue
            # All retries exhausted or non-retryable error
            if "503" in error_str or "UNAVAILABLE" in error_str:
                return (
                    "⚠️ **The Gemini AI service is temporarily unavailable** (model overloaded). "
                    "The analysis could not be generated after 3 attempts. "
                    "Please try again in a few moments — this is usually a transient issue."
                )
            return f"**Gemini API error:** {exc}"


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏢 HR Employee Attrition Predictor")
st.markdown(
    f"AI-powered attrition risk assessment using **{BEST_MODEL}** "
    f"(Test AUC: **{TEST_AUC}**, Accuracy: **{TEST_ACC}**) "
    "combined with **Google Gemini 3.6 Flash** for intelligent HR insights."
)
st.divider()

# ── Sidebar: Employee Input Form ──────────────────────────────────────────────
st.sidebar.header("🧑‍💼 Employee Details")
st.sidebar.caption("Fill in the employee profile to get an attrition prediction.")

def num_input(label, col):
    stats = NUM_STATS[col]
    step  = 1 if stats["max"] - stats["min"] > 10 else 0.1
    return st.sidebar.number_input(
        label,
        min_value=float(stats["min"]),
        max_value=float(stats["max"]),
        value=float(stats["median"]),
        step=float(step),
        key=col,
    )

def cat_input(label, col):
    opts = CAT_OPTIONS[col]
    return st.sidebar.selectbox(label, opts, key=col)


# ── Personal & Job Details ────────────────────────────────────────────────────
st.sidebar.subheader("Personal Information")
age           = num_input("Age", "Age")
gender        = cat_input("Gender", "Gender")
marital       = cat_input("Marital Status", "MaritalStatus")
education     = cat_input("Education Level", "Higher_Education")

st.sidebar.subheader("Job Details")
department    = cat_input("Department", "Department")
job_role      = cat_input("Job Role", "JobRole")
job_level     = num_input("Job Level (1–5)", "JobLevel")
job_mode      = cat_input("Job Mode", "Job_mode")
business_travel = cat_input("Business Travel", "BusinessTravel")
mode_of_work  = cat_input("Work Mode", "Mode_of_work")
overtime      = cat_input("OverTime", "OverTime")
source_hire   = cat_input("Source of Hire", "Source_of_Hire")

st.sidebar.subheader("Compensation & Experience")
monthly_income  = num_input("Monthly Income", "MonthlyIncome")
pct_salary_hike = num_input("% Salary Hike", "PercentSalaryHike")
stock_option    = num_input("Stock Option Level", "StockOptionLevel")
total_wk_years  = num_input("Total Working Years", "TotalWorkingYears")
num_companies   = num_input("Num Companies Worked", "NumCompaniesWorked")

st.sidebar.subheader("Performance & Satisfaction")
job_involvement   = num_input("Job Involvement (1–4)", "JobInvolvement")
job_satisfaction  = num_input("Job Satisfaction (1–4)", "JobSatisfaction")
perf_rating       = num_input("Performance Rating (1–4)", "PerformanceRating")
training_times    = num_input("Training Times Last Year", "TrainingTimesLastYear")

st.sidebar.subheader("Tenure & Wellbeing")
years_at_company  = num_input("Years at Company", "YearsAtCompany")
years_since_promo = num_input("Years Since Last Promotion", "YearsSinceLastPromotion")
years_curr_mgr    = num_input("Years with Current Manager", "YearsWithCurrManager")
distance_home     = num_input("Distance From Home (km)", "DistanceFromHome")
leaves            = num_input("Leaves Taken", "Leaves")
absenteeism       = num_input("Absenteeism Days", "Absenteeism")
work_accident     = cat_input("Work Accident", "Work_accident")

predict_btn = st.sidebar.button("🔍 Predict Attrition", type="primary", use_container_width=True)

# ── Collect inputs into a dict matching feature order ─────────────────────────
INPUT_MAP = {
    "Age":                   age,
    "BusinessTravel":        business_travel,
    "Department":            department,
    "DistanceFromHome":      distance_home,
    "Gender":                gender,
    "JobInvolvement":        job_involvement,
    "JobLevel":              job_level,
    "JobRole":               job_role,
    "JobSatisfaction":       job_satisfaction,
    "MaritalStatus":         marital,
    "MonthlyIncome":         monthly_income,
    "NumCompaniesWorked":    num_companies,
    "OverTime":              overtime,
    "PercentSalaryHike":     pct_salary_hike,
    "PerformanceRating":     perf_rating,
    "StockOptionLevel":      stock_option,
    "TotalWorkingYears":     total_wk_years,
    "TrainingTimesLastYear": training_times,
    "YearsAtCompany":        years_at_company,
    "YearsSinceLastPromotion": years_since_promo,
    "YearsWithCurrManager":  years_curr_mgr,
    "Higher_Education":      education,
    "Mode_of_work":          mode_of_work,
    "Leaves":                leaves,
    "Absenteeism":           absenteeism,
    "Work_accident":         work_accident,
    "Source_of_Hire":        source_hire,
    "Job_mode":              job_mode,
}

# ── Main area — show results ──────────────────────────────────────────────────
col_info, col_chart = st.columns([3, 2])

with col_info:
    st.subheader("📊 Model Performance Summary")
    result_df = pd.DataFrame(
        {"Model": list(ALL_RESULTS.keys()), "ROC-AUC": list(ALL_RESULTS.values())}
    ).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    result_df["ROC-AUC"] = result_df["ROC-AUC"].round(4)
    result_df.index += 1
    st.dataframe(result_df, use_container_width=True)

with col_chart:
    st.subheader("🏆 Best Model Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Best Model", BEST_MODEL)
    m2.metric("ROC-AUC (Test)", TEST_AUC)
    m3, m4 = st.columns(2)
    m3.metric("Accuracy (Test)", TEST_ACC)
    m4.metric("Training Samples", "1470")

st.divider()

# ── Prediction section ────────────────────────────────────────────────────────
if predict_btn:
    # Build DataFrame with correct column order
    all_features = NUM_FEATURES + CAT_FEATURES
    row = {feat: INPUT_MAP.get(feat) for feat in all_features}
    input_df = pd.DataFrame([row])

    # Predict
    pred_label = pipeline.predict(input_df)[0]
    pred_prob  = pipeline.predict_proba(input_df)[0]

    attrition_prob = pred_prob[1]   # probability of Attrition = Yes
    no_attrition_prob = pred_prob[0]
    prediction_text = "YES — Employee Likely to Leave" if pred_label == 1 else "NO — Employee Likely to Stay"

    # ── Result display ────────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Result")

    risk_color = (
        "red"    if attrition_prob >= 0.7
        else "orange" if attrition_prob >= 0.4
        else "green"
    )
    risk_label = (
        "🔴 HIGH RISK"    if attrition_prob >= 0.7
        else "🟡 MEDIUM RISK" if attrition_prob >= 0.4
        else "🟢 LOW RISK"
    )

    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric(
        "Attrition Prediction",
        "Will Leave ⚠️" if pred_label == 1 else "Will Stay ✅",
    )
    res_c2.metric(
        "Attrition Probability",
        f"{attrition_prob:.1%}",
        delta=f"{attrition_prob - 0.5:+.1%} vs 50% baseline",
        delta_color="inverse",
    )
    res_c3.metric("Risk Level", risk_label)

    st.progress(float(attrition_prob), text=f"Attrition Probability: {attrition_prob:.1%}")

    st.divider()

    # ── Gemini AI Explanation ─────────────────────────────────────────────────
    st.subheader("🤖 Gemini 3.6 Flash — AI HR Analysis")

    if not GEMINI_API_KEY:
        st.warning(
            "Gemini API key not configured. Add `GEMINI_API_KEY=your_key` "
            "to your `.env` file to enable AI-powered explanations.",
            icon="⚠️",
        )
    else:
        with st.spinner("Generating AI-powered HR analysis with Gemini 3.6 Flash…"):
            explanation = get_gemini_explanation(INPUT_MAP, prediction_text, attrition_prob)
        st.markdown(explanation)

    st.divider()

    # ── Employee profile summary ──────────────────────────────────────────────
    st.subheader("📋 Submitted Employee Profile")
    profile_df = pd.DataFrame(
        INPUT_MAP.items(), columns=["Feature", "Value"]
    )
    # Convert Value column to str so PyArrow can handle mixed numeric/text types
    profile_df["Value"] = profile_df["Value"].astype(str)
    col_a, col_b = st.columns(2)
    half = len(profile_df) // 2
    col_a.dataframe(profile_df.iloc[:half].reset_index(drop=True), use_container_width=True)
    col_b.dataframe(profile_df.iloc[half:].reset_index(drop=True), use_container_width=True)

else:
    # Default state — instructions
    st.info(
        "👈 Fill in the **Employee Details** in the sidebar and click "
        "**Predict Attrition** to get a prediction and AI-powered HR analysis.",
        icon="ℹ️",
    )
    st.subheader("ℹ️ About This Application")
    st.markdown(
        """
This application uses a **machine learning pipeline** trained on an IBM-style
HR attrition dataset to predict whether an employee is likely to leave the organisation.

**How it works:**
1. Multiple classification models (Logistic Regression, Random Forest, Gradient Boosting,
   Decision Tree, AdaBoost) are trained and compared using 5-fold cross-validation.
2. The best-performing model (by ROC-AUC) is automatically selected and saved.
3. You input an employee's details → the model returns an attrition probability.
4. **Google Gemini 3.6 Flash** analyses the profile and prediction to generate
   personalised HR recommendations.

**Features used:** Age, Department, Job Role, Monthly Income, OverTime status,
Job Satisfaction, Years at Company, and many more HR metrics.
        """
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HR Attrition Predictor · Powered by scikit-learn & Google Gemini 3.6 Flash · "
    "Built with Streamlit"
)
