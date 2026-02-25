

# FundSight AI
### Intelligent Mutual Fund Advisory & Investment Simulation Engine

**FundSight AI** is a production-grade machine learning system designed to analyze, recommend, and simulate mutual fund investments. Unlike simple dashboard wrappers, this project implements a **full ML lifecycle pipeline**—handling raw data ingestion, historical sanity checks, time-series feature engineering, and offline model training—before serving predictions via a conversational UI.

The system utilizes **XGBoost** for CAGR prediction and **SHAP** for interpretability, wrapped in a **Streamlit** interface with rule-based conversational agents.


---

##  Live App  
👉 [Click here to try the Streamlit App](https://fundsight-ai.streamlit.app/)

---

## 🏗️ Project Philosophy

This repository mirrors real-world financial ML systems by prioritizing:

1. **Data Integrity**  
   Strict sanity checks reject funds with incomplete histories or data gaps.

2. **Pipeline Separation**  
   Heavy data engineering (Offline) is cleanly decoupled from inference and user interaction (Online).

3. **Explainability**  
   SHAP values explain *why* a fund is recommended, not just *what* to buy.

4. **Simulation**  
   Realistic SIP (Systematic Investment Plan) and Lump Sum calculators based on historical volatility.

---

## 📂 Repository Structure

fundsight/

src/                        # Core ML & Data Pipeline (Offline Logic)
- data_ingestion.py         # Fetches raw NAV data from APIs
- history_sanity_check.py   # Filters out unstable or short-history funds
- nav_preprocessing.py      # Aligns time-series data
- data_preprocessing.py     # Cleans and normalizes datasets
- feature_Engineering.py    # Generates volatility, alpha, beta & rolling metrics
- model_training.py         # Trains XGBoost and saves artifacts
- model_Inference.py        # Logic for generating predictions
- recommender.py            # Ranking logic for user recommendations
- sip_calculator.py         # Investment simulation logic
- shap_explainbility.py     # Generates offline explanation charts

app/                        # Conversational Advisory Layer (Online Logic)
- agents/                   # Advisor and Calculator agents
- router.py                 # Intent detection and routing
- state.py                  # Session state management

data/                       # Pipeline cache & artifact store
                             (Raw NAVs, cleaned datasets, engineered features)

models/                     # Model artifacts
- final_xgb_model.json      # Trained XGBoost model
- feature_columns.json     # Feature schema for inference

shap_exports/               # Pre-computed SHAP explanation visualizations
str_app.py                  # Streamlit application entry point
requirements.txt            # Project dependencies

---

## ⚙️ Execution Flow (Pipeline Architecture)

This project operates in **two distinct phases**.  
You **do NOT** need to retrain the model to run the app.

---

### 🔹 PHASE 1: Data Engineering & Training (Offline)

Run these scripts **only** when updating data or retraining the model.

Ingestion & Validation  
Fetch NAV data and reject unstable or incomplete funds.

python src/data_ingestion.py  
python src/history_sanity_check.py  

Preprocessing  
Align NAV dates and normalize values.

python src/nav_preprocessing.py  
python src/data_preprocessing.py  

Feature Engineering  
Compute financial metrics such as Sharpe Ratio, volatility, and rolling returns.

python src/feature_Engineering.py  
python src/scripts/map_to_features.py  

Model Training  
Train XGBoost and persist model artifacts.

python src/model_training.py  

Explainability (Optional)  
Generate SHAP plots for interpretability and analysis.

python src/shap_explainbility.py  

---

### 🔹 PHASE 2: Application Runtime (User-Facing)

This is the **only command required for end users**.  
The application loads the pre-trained model and cached features to provide real-time advice.

streamlit run str_app.py

---

## 🌟 Key Features

- **Advisory Mode**  
  Context-aware recommendations based on user risk (Low / Medium / High) and investment horizon.

- **Smart Calculator**  
  Simulates SIP (with optional step-up) and one-time lump sum investments.

- **“Why?” Engine**  
  Uses SHAP values to explain the exact factors driving each recommendation  
  (e.g., “Recommended due to high Sharpe Ratio and stable volatility”).

- **Conversational Interface**  
  Rule-based NLP router to handle natural language investment queries.

---

## 🛠️ Tech Stack

- Language: Python 3.9+
- Machine Learning: XGBoost, Scikit-learn
- Explainability: SHAP (SHapley Additive Explanations)
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Plotly
- Interface: Streamlit

---

## 🚀 Setup & Installation

Clone the repository :  git clone https://github.com/your-username/fundsight-ai.git  

cd fundsight-ai  

Create a virtual environment :  python -m venv venv  

Windows :   .\venv\Scripts\activate  
Mac / Linux :   source venv/bin/activate  

Install dependencies : pip install -r requirements.txt  

Run the application : streamlit run str_app.py  

---

## 🔮 Future Roadmap

- [ ] Integration with live market APIs (e.g., AMFI real-time data)
- [ ] Portfolio optimization using Markowitz Efficient Frontier
- [ ] User authentication and portfolio persistence

---

## 📄 License

This project is open-source and available under the **MIT License**.


