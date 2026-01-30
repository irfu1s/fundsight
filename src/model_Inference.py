import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from xgboost import XGBRegressor

# ================= CONFIGURATION ================= #
FEATURE_PATH = Path("data/sip_features_model_ready.csv")
MODEL_PATH = Path("models/final_xgb_model.json")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")
PREDICTIONS_OUT_PATH = Path("data/sip_predictions.csv")

# ================= HELPER: NORMALIZE NAME ================= #

def normalize_scheme_name(s: str) -> str:
    """
    Cleans up fund names so we can group duplicates.
    Example: "Nippon Small Cap - Direct Growth" -> "NIPPON INDIA SMALL CAP FUND"
    """
    if not isinstance(s, str): return ""
    x = s.upper()
    junk_words = [
        "DIRECT PLAN", "REGULAR PLAN", "GROWTH OPTION", "GROWTH",
        "IDCW", "DIVIDEND", "REINVESTMENT", "BONUS",
        "MONTHLY", "QUARTERLY", "ANNUAL", "PAYOUT", "PAY OUT",
        "OPTION", "PLAN", "-", "  ",
    ]
    for w in junk_words:
        x = x.replace(w, " ")
    return " ".join(x.split())

# ================= CORE INFERENCE LOGIC ================= #

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("❌ Model not found. Run model_training.py first.")
        
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)
        
    print(f"✅ Artifacts loaded. Expecting {len(feature_cols)} features.")
    return model, scaler, feature_cols

def predict_for_dataframe(df: pd.DataFrame, model, scaler, feature_cols):
    # 1. Align Columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # 2. Sanitize
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # 3. Scale & Predict
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print(f"❌ Scaler Failed: {e}")
        raise e

    preds = model.predict(X_scaled)
    preds = np.clip(preds, -99, 200) # Safety clip
    return preds

# ================= MAIN PIPELINE ================= #

def predict_all_schemes(save_to_csv=True):
    print("🚀 Starting Inference Engine...")
    
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature file missing: {FEATURE_PATH}")
    
    df = pd.read_csv(FEATURE_PATH)
    print(f"   Loaded {len(df)} schemes (raw).")
    
    # 1. Generate Predictions
    model, scaler, feature_cols = load_artifacts()
    preds = predict_for_dataframe(df, model, scaler, feature_cols)
    df["predicted_cagr_percent"] = preds.round(2)
    
    # 2. CLEAN & DEDUPLICATE (The Fix!)
    print("🧹 Cleaning and merging duplicate funds...")
    
    # Create Base Name column
    df["base_name"] = df["scheme_name"].apply(normalize_scheme_name)
    
    # Sort by Prediction (Best returns on top)
    df = df.sort_values("predicted_cagr_percent", ascending=False)
    
    # Group by Base Name and take the first one (The Winner)
    # This automatically discards duplicates like "Regular Plan" if "Direct" is better
    df_clean = df.drop_duplicates(subset=["base_name"]).copy()
    
    print(f"   Reduced from {len(df)} variants to {len(df_clean)} unique funds.")

    # 3. Select Final Columns
    output_cols = ["scheme_code", "scheme_name", "base_name", "category", "predicted_cagr_percent", "category_risk_tier"]
    
    # Ensure all output columns exist
    for c in output_cols:
        if c not in df_clean.columns:
            df_clean[c] = np.nan
            
    final_df = df_clean[output_cols]
    
    if save_to_csv:
        PREDICTIONS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(PREDICTIONS_OUT_PATH, index=False)
        print(f"✅ Clean Predictions saved to: {PREDICTIONS_OUT_PATH}")
        print("-" * 30)
        print(final_df[["base_name", "predicted_cagr_percent", "category"]].head())
        print("-" * 30)

    return final_df

def main():
    try:
        predict_all_schemes()
    except Exception as e:
        print(f"❌ Inference Failed: {e}")

if __name__ == "__main__":
    main()