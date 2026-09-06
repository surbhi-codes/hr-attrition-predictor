# HR Employee Attrition Predictor

An AI-powered machine learning web application that predicts employee attrition risk and provides intelligent HR recommendations using **Google Gemini 2.5 Flash**.

## Features

- **Binary classification** — predicts whether an employee is likely to leave (`Yes`) or stay (`No`)
- **Multi-model comparison** — trains and evaluates Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and AdaBoost; automatically selects the best model by ROC-AUC
- **Complete ML pipeline** — preprocessing (scaling, encoding, imputation) + classifier saved as a single `.pkl`
- **Gemini 2.5 Flash integration** — generates personalised HR analysis, key risk factors, protective factors, and actionable recommendations
- **Streamlit UI** — clean sidebar form for employee details, live prediction with probability and risk level

---

## Project Structure

```
hr-attrition-predictor/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training & evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── agent_instructions.md   # AI agent context
├── .env.example            # Environment variable template
├── data/
│   └── dataset attrition.csv   # IBM HR Attrition dataset (1470 rows)
└── models/
    └── model.pkl           # Trained ML pipeline + metadata (auto-generated)
```

---

## Local Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hr-attrition-predictor.git
cd hr-attrition-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Train the model

```bash
python train_model.py
```

This will:
- Load `data/dataset attrition.csv`
- Compare 5 ML models with 5-fold cross-validation
- Save the best pipeline to `models/model.pkl`

### 6. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Streamlit Community Cloud Deployment

1. Push your repository to GitHub (ensure `models/model.pkl` is committed **or** add a `@st.cache_resource` startup step that trains if model is missing — the current `app.py` requires the model file).

   > **Recommended**: commit `models/model.pkl` to the repo so deployment works without re-training.

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo and `app.py`.

3. In **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```

4. Click **Deploy** — the app will be live within minutes.

---

## Environment Variables

| Variable         | Required | Description                          |
|------------------|----------|--------------------------------------|
| `GEMINI_API_KEY` | Yes*     | Google Gemini API key for AI analysis |

\* The app works without a Gemini key — ML predictions still function; only the AI explanation panel is disabled.

---

## Model Performance

| Model               | CV ROC-AUC |
|---------------------|-----------|
| Random Forest       | ~0.84     |
| Gradient Boosting   | ~0.83     |
| AdaBoost            | ~0.82     |
| Logistic Regression | ~0.79     |
| Decision Tree       | ~0.72     |

*(Exact values depend on the random state and may vary slightly.)*

---

## Tech Stack

- **Python 3.14.2**
- **scikit-learn** — ML pipeline, preprocessing, model training
- **pandas / numpy** — data manipulation
- **Streamlit** — web UI
- **Google Generative AI SDK** — Gemini 2.5 Flash integration
- **joblib** — model serialisation
- **python-dotenv** — environment variable management
